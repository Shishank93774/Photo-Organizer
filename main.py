"""Main entry point for photo organization tool."""
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from src.loader import load_photos
from src.detector import detect_faces_batch, print_detection_summary
from src.encoder import generate_face_encodings, validate_encodings
from src.cache import save_cache, load_cache


def get_directory_modified_time(directory_path: Path): # captures any add or deletes in the photo_directory folder
    """Get the directory's and it's sub-directory's  last modified time"""

    if not directory_path.exists() or not directory_path.is_dir():
        return None

    max_mod_time = directory_path.stat().st_mtime

    for dir in directory_path.iterdir():
        if dir.is_dir():
            max_mod_time = max(max_mod_time, get_directory_modified_time(dir))

    return max_mod_time


def main(photos_directory: str, force_cache_recompute: bool = False, verbose: bool = True) -> dict:
    """"
    Main pipeline for face detection and encoding.

    Args:
        photos_directory: Path to folder containing photos
        force_cache_recompute: If True, ignore cache and regenerate
        verbose: If true, provide verbose output

    Returns:
        Dictionary with face detections and encodings
    """

    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") # current date and time

    photos_path = Path(photos_directory)
    cache_path = photos_path.parent / "cache"

    if not force_cache_recompute:
        print("Checking for cached data...")
        last_modified_dir_ts = datetime.fromtimestamp(get_directory_modified_time(photos_path)).strftime("%Y_%m_%d_%H_%M_%S")
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
    photos = load_photos(photos_path, verbose=verbose)

    print("\n[2/4] Detecting faces...")
    face_data = detect_faces_batch(photos, verbose=verbose)
    print_detection_summary(face_data)

    print("\n[3/4] Generating encodings...")
    generate_face_encodings(face_data, verbose=verbose)

    print("\n[4/4] Validating and saving...")
    stats = validate_encodings(face_data)
    print(f"\nEncoding quality: {stats['valid']}/{stats['total']} valid")

    save_cache(cache_path=cache_path, face_data=face_data, now=now)
    print("\n\n✓ Pipeline complete!")

    return face_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Photo organization tool")
    parser.add_argument("photos_directory", help="Directory containing photos")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--force-recompute", action="store_true")

    args = parser.parse_args()
    result = main(args.photos_directory, args.force_recompute, args.verbose)