"""Cache management utilities using SQLite for incremental updates."""
import sqlite3
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

class SQLiteCache:
    def __init__(self, db_path: Path):
        """Initialize DB connection and create tables if they don't exist."""
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        """Create the necessary tables for caching images and faces."""
        with self.conn:
            # Table for global metadata
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            # Table for tracking photos and their modification times
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS photos (
                    path TEXT PRIMARY KEY,
                    mtime REAL,
                    size INTEGER
                )
            """)

            # Table for storing detected faces and encodings
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS faces (
                    uuid TEXT PRIMARY KEY,
                    photo_path TEXT,
                    face_id INTEGER,
                    bbox TEXT,
                    encoding BLOB,
                    FOREIGN KEY (photo_path) REFERENCES photos (path) ON DELETE CASCADE
                )
            """)

            # Migration: Ensure 'size' column exists for existing databases
            cursor = self.conn.execute("PRAGMA table_info(photos)")
            columns = [row['name'] for row in cursor.fetchall()]
            if 'size' not in columns:
                self.conn.execute("ALTER TABLE photos ADD COLUMN size INTEGER")

    def get_global_mtime(self) -> Optional[float]:
        """Retrieve the last stored global directory modification time."""
        cursor = self.conn.execute("SELECT value FROM meta WHERE key = 'last_global_mtime'")
        row = cursor.fetchone()
        return float(row['value']) if row else None

    def set_global_mtime(self, mtime: float):
        """Update the meta table with the latest directory mtime."""
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
                ("last_global_mtime", str(mtime))
            )

    def get_incremental_updates(self, photos_path: Path) -> Dict[str, List[Path]]:
        """
        Implements Incremental Scan to detect new, modified, and deleted photos.
        Returns a dict: {'new': [], 'modified': [], 'deleted': [], 'none': []}
        """

        # Incremental Scan
        # 1. Fetch all cached paths, mtimes, and sizes
        cursor = self.conn.execute("SELECT path, mtime, size FROM photos")
        cached_data = {row['path']: (row['mtime'], row['size']) for row in cursor.fetchall()}

        # 2. Scan filesystem
        current_files = {}
        for file in photos_path.rglob("*"):
            if file.suffix.lower() in (".jpg", ".jpeg", ".png", ".heic"):
                stat = file.stat()
                # Use .resolve() for consistent absolute paths across Windows/Linux
                current_files[str(file.resolve())] = (stat.st_mtime, stat.st_size)

        # 3. Compute differences
        new = []
        modified = []
        deleted = []

        current_paths = set(current_files.keys())
        cached_paths = set(cached_data.keys())

        # New files
        for path in (current_paths - cached_paths):
            new.append(Path(path))

        # Modified files
        for path in (current_paths & cached_paths):
            current_mtime, current_size = current_files[path]
            cached_mtime, cached_size = cached_data[path]

            # Robust modification check:
            # 1. Size changed (almost certain modification)
            # 2. Mtime increased beyond a 1-second threshold (to handle filesystem precision/drift)
            # 3. cached_size is None (migration case)
            if (cached_size is None or
                current_size != cached_size or
                current_mtime > (cached_mtime + 1.0)):
                modified.append(Path(path))

        # Deleted files
        for path in (cached_paths - current_paths):
            deleted.append(Path(path))

        return {
            'new': new,
            'modified': modified,
            'deleted': deleted,
            'all_current': list(current_paths)
        }

    def update_photo_data(self, path: Path, mtime: float, faces: List[Dict]):
        """
        Updates the DB for a specific photo.
        faces list contains {'uuid', 'face_id', 'bbox', 'encoding'}.
        """
        path_str = str(path.resolve())
        size = path.stat().st_size
        with self.conn:
            # Remove existing faces for this photo to avoid duplicates on modification
            self.conn.execute("DELETE FROM faces WHERE photo_path = ?", (path_str,))
            # Update or insert photo record
            self.conn.execute(
                "INSERT OR REPLACE INTO photos (path, mtime, size) VALUES (?, ?, ?)",
                (path_str, mtime, size)
            )
            # Insert faces
            for face in faces:
                # encoding is a numpy array, convert to bytes
                encoding_bytes = face['encoding'].tobytes() if face['encoding'] is not None else None
                bbox_json = json.dumps(face['bbox'])

                self.conn.execute(
                    "INSERT INTO faces (uuid, photo_path, face_id, bbox, encoding) VALUES (?, ?, ?, ?, ?)",
                    (face['uuid'], path_str, face['face_id'], bbox_json, encoding_bytes)
                )

    def remove_photo_data(self, path: Path):
        """Remove a photo and its associated faces from the cache."""
        path_str = str(path.resolve())
        with self.conn:
            self.conn.execute("DELETE FROM photos WHERE path = ?", (path_str,))

    def reconstruct_face_data(self) -> Dict:
        """
        Queries the DB and reconstructs the face_data dictionary.
        Format: { "path": [ { "uuid": ..., "bbox": ..., "encoding": ... }, ... ], ... }
        """
        face_data = {}
        cursor = self.conn.execute("SELECT photo_path, uuid, face_id, bbox, encoding FROM faces")

        for row in cursor.fetchall():
            path = row['photo_path']
            if path not in face_data:
                face_data[path] = []

            # Convert bytes back to numpy array
            encoding_blob = row['encoding']
            encoding = None
            if encoding_blob is not None:
                encoding = np.frombuffer(encoding_blob, dtype=np.float64)

            bbox = json.loads(row['bbox'])

            face_data[path].append({
                "face_id": f"photo_{Path(path).stem}_image_{row['face_id']}",
                "uuid": row['uuid'],
                "bbox": bbox,
                "encoding": encoding
            })

        return face_data

    def close(self):
        """Close the DB connection."""
        self.conn.close()
