"""Main entry point for photo organization tool."""
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional

from src.loader import load_photos
from src.detector import detect_faces_batch, print_detection_summary
from src.encoder import generate_face_encodings, validate_encodings
from src.cache import save_cache, load_cache


def get_latest_photo_modification_time(directory_path: str) -> Optional[str]:
    """
    Get the most recent modification time of any photo in the directory tree.

    Args:
        directory_path: Root directory to check

    Returns:
        Timestamp string of most recent photo, or None if no photos found
    """
    directory_path = Path(directory_path)
    if not directory_path.exists() or not directory_path.is_dir():
        return None

    latest_time = 0

    # Check all image files recursively
    for ext in ['.jpg', '.jpeg', '.png']:
        for photo_path in directory_path.rglob(f'*{ext}'):
            try:
                mod_time = photo_path.stat().st_mtime
                latest_time = max(latest_time, mod_time)
            except FileExistsError as fee:
                print(f"File {photo_path} seems to not exist, Error- {fee}")
                continue

    if latest_time == 0:
        return None

    return datetime.fromtimestamp(latest_time).strftime("%Y_%m_%d_%H_%M_%S")


def main(photos_directory: str, force_cache_recompute: bool = False) -> dict:
    """"
    Main pipeline for face detection and encoding.

    Args:
        photos_directory: Path to folder containing photos
        force_cache_recompute: If True, ignore cache and regenerate

    Returns:
        Dictionary with face detections and encodings
    """

    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") # current date and time

    photos_path = Path(photos_directory)
    cache_path = photos_path.parent / "cache"

    if not force_cache_recompute:
        print("Checking for cached data...")
        last_modified_dir_ts = get_latest_photo_modification_time(photos_directory)
        cached_data = load_cache(cache_path, last_modified_dir_ts)
        if cached_data is not None:
            print("✓ Using cached data (skip detection and encoding)")
            print_detection_summary(cached_data)
            return cached_data
        else:
            print("No cache found. Processing from scratch...")
    else:
        print("Force recompute enabled. Processing from scratch...")

    # Process from scratch
    print("[1/4] Loading photos...")
    photos = load_photos(photos_path, verbose=True)

    print("\n[2/4] Detecting faces...")
    face_data = detect_faces_batch(photos, verbose=True)
    print_detection_summary(face_data)

    print("\n[3/4] Generating encodings...")
    generate_face_encodings(face_data, verbose=True)

    print("\n[4/4] Validating and saving...")
    # stats = validate_encodings(face_data)
    # print(f"Encoding quality: {stats['valid']}/{stats['total']} valid")

    save_cache(cache_path=cache_path, face_data=face_data, now=now)
    print("✓ Pipeline complete!")

    return face_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Photo organization tool")
    parser.add_argument("photos_directory", help="Directory containing photos")
    parser.add_argument("--force-recompute", action="store_true")

    args = parser.parse_args()
    result = main(args.photos_directory, args.force_recompute)