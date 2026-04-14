"""Photo loading utilities."""
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import numpy as np
from PIL import Image
import pillow_heif
import logging

# Register HEIF opener to allow PIL to open .heic files
pillow_heif.register_heif_opener()


# Register HEIF opener to allow PIL to open .heic files
pillow_heif.register_heif_opener()

def load_image_as_array(path: Path) -> np.ndarray:
    """
    Loads an image file and returns it as an RGB numpy array, supporting HEIC.

    Args:
        path: Path to the image file

    Returns:
        RGB numpy array of the image
    """
    img = Image.open(path)
    img = img.convert("RGB")
    return np.array(img)

def load_photos(directory_path: Path, verbose: bool = True, target_photos: Optional[List[Path]] = None) -> List[Path]:
    """
    Recursively load image file paths from directory.

    Args:
        directory_path: Root directory to search
        verbose: Print progress messages if True
        target_photos: If provided, only load these specific photos (must be within directory_path)

    Returns:
        List of absolute paths to image files
    """
    logger = logging.getLogger()
    if target_photos is not None:
        # Filter target_photos to ensure they are actually under directory_path and still exist
        valid_photos = []
        for p in target_photos:
            if p.is_relative_to(directory_path) and p.exists():
                valid_photos.append(p)
        return valid_photos

    photos = []
    directory = str(directory_path)
    if verbose:
        logger.info(f"Scanning {directory}")

    try:
        for item in directory_path.iterdir():
            if item.is_dir():
                # Recursive call for subdirectories
                photos.extend(load_photos(item, verbose))
            elif item.suffix.lower() in {'.jpg', '.jpeg', '.png', '.heic'}:
                # Only add image files
                photos.append(item)
    except PermissionError as e:
        logger.warning(f"Permission denied: {directory}")
    except Exception as e:
        logger.error(f"Error scanning {directory}: {e}")

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
    for ext in ['.jpg', '.jpeg', '.png', '.heic']:
        for photo_path in directory_path.rglob(f'*{ext}'):
            try:
                mod_time = photo_path.stat().st_mtime
                latest_time = max(latest_time, mod_time)
            except Exception as e:
                continue

    if latest_time == 0:
        return None

    return datetime.fromtimestamp(latest_time).strftime("%Y_%m_%d_%H_%M_%S")
