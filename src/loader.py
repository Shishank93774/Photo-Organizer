"""Photo loading utilities."""
from pathlib import Path
from typing import List, Optional
from datetime import datetime


def load_photos(directory_path: Path, verbose: bool = False) -> List[Path]:
    """
    Recursively load all image file paths from directory.

    Args:
        directory_path: Root directory to search
        verbose: Print progress messages if True

    Returns:
        List of absolute paths to image files
    """
    photos = []
    directory = str(directory_path)
    if verbose:
        print(f"Scanning {directory}")

    try:
        for item in directory_path.iterdir():
            if item.is_dir():
                # Recursive call for subdirectories
                photos.extend(load_photos(item, verbose))
            elif item.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                # Only add image files
                photos.append(item)
    except PermissionError as e:
        if verbose:
            print(f"Permission denied: {directory}")
    except Exception as e:
        if verbose:
            print(f"Error scanning {directory}: {e}")

    return photos


def get_latest_photo_modification_time(directory_path: Path) -> Optional[str]:
    """
    Get the most recent modification time of any photo in the directory tree.

    Args:
        directory_path: Root directory to check

    Returns:
        Timestamp string of most recent photo, or None if no photos found
    """
    if not directory_path.exists() or not directory_path.is_dir():
        return None

    latest_time = 0

    # Check all image files recursively
    for ext in ['.jpg', '.jpeg', '.png', '.HEIC']:
        for photo_path in directory_path.rglob(f'*{ext}'):
            try:
                mod_time = photo_path.stat().st_mtime
                latest_time = max(latest_time, mod_time)
            except Exception as e:
                continue

    if latest_time == 0:
        return None

    return datetime.fromtimestamp(latest_time).strftime("%Y_%m_%d_%H_%M_%S")
