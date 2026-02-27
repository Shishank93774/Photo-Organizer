"""Cache management utilities."""
import re
import pickle
from pathlib import Path
from typing import Dict, Optional

from src.encoder import ENCODING_PATTERN


def save_cache(cache_path: Path, face_data: Dict, now: str) -> None:
    """
    Save face data with encodings to cache file.

    Args:
        cache_path: Directory path to cache storage
        face_data: Dictionary with face detections and encodings
        now: Denotes the current timestamp for encodings versioning
    """

    cache_path.mkdir(exist_ok=True)

    cache_file = cache_path / f"encodings_{now}.pkl"

    try:
        with open(cache_file, "wb") as file:
            pickle.dump(face_data, file)  # type: ignore
        print("Encodings cached...")
    except Exception as e:
        print(f"Error saving encodings into cache, Error- {e}")


def _get_latest_encodings(cache_path: Path) -> Path|None:
    """
    Find the most recent encodings file in the cache directory.

    Returns:
        Path to the most recent encodings file, or None if no matching files found
    """

    # Create cache directory if it doesn't exist
    cache_path.mkdir(exist_ok=True)

    # Get all matching files
    matching_files = [f for f in cache_path.iterdir() if f.is_file() and re.match(ENCODING_PATTERN, f.name)]

    # Return None if no matching files found
    if not matching_files:
        return None

    # Return the most recent file (based on filename timestamp)
    return max(matching_files)


def _clear_cached_encodings(cache_path: Path) -> None:

    print("Clearing up cache files...")

    if Path.exists(cache_path):
        encodings_files = [f for f in cache_path.iterdir() if f.is_file() and re.match(ENCODING_PATTERN, f.name)]
        try:
            for file in encodings_files:
                file.unlink()
        except Exception as e:
            print(f"Error while clearing up cache, Error- {e}")


def load_cache(cache_path: Path, last_modified_dir_ts: str) -> Optional[Dict]:
    """
    Load face data from cache file if it exists.

    Returns:
        Cached face data, or None if cache doesn't exist
    """
    # Your implementation:
    # 1. Check if cache file exists
    # 2. If exists, load with pickle.load()
    # 3. If doesn't exist, return None
    # 4. Handle corrupted cache files

    cache_file = _get_latest_encodings(cache_path)

    if cache_file is None:
        return None

    last_modified_cache_ts = cache_file.stem.split("encodings_")[1]

    if last_modified_dir_ts > last_modified_cache_ts:
        _clear_cached_encodings(cache_path)
        print("Photo directory was modified, cache invalidated!")
        return None

    if Path.exists(Path(cache_file)):
        try:
            with open(cache_file, "rb") as file:
                face_data = pickle.load(file)
            print(f"Encoding cache loaded successfully from {cache_file}!")

            return face_data
        except Exception as e:
            print(f"Error loading encoding cache, Error- {e}")

    return None