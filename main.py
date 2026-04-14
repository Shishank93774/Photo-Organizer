"""Main entry point for photo organization tool."""
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import argparse
import time
import logging
from pathlib import Path
from typing import Any

from src.loader import load_photos
from src.detector import detect_faces_batch, print_detection_summary
from src.encoder import generate_face_encodings
from src.cache import SQLiteCache
from src.clustering import extract_encodings_for_clustering, cluster_faces, save_cluster_summary
from src.organizer import name_clusters_interactive, organize_photos
from src.logger import setup_logging


def get_directory_modified_time(directory_path: Path):
    """Get the directory's and it's sub-directory's last modified time"""

    if not directory_path.exists() or not directory_path.is_dir():
        return None

    max_mod_time = directory_path.stat().st_mtime

    for dir in directory_path.iterdir():
        if dir.is_dir():
            max_mod_time = max(max_mod_time, get_directory_modified_time(dir))

    return max_mod_time



def load_face_data(
        photos_directory: str,
        force_cache_recompute: bool = False,
        verbose: bool = True,
        use_cnn: bool = False,
        parallel: bool = False,
        downscale: bool = False,
        log_queue: Any = None
) -> dict:
    """
    Loads faces from given photo directory and returns face_data with encodings
    Using SQLite incremental caching.
    """
    photos_path = Path(photos_directory)
    cache_dir = photos_path.parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    db_path = cache_dir / "cache.db"

    cache = SQLiteCache(db_path)

    if not force_cache_recompute:
        print("Checking for cached data...")

        # Pass 1: Global Fast-Path Check
        current_max_mtime = get_directory_modified_time(photos_path)
        cached_max_mtime = cache.get_global_mtime()

        if cached_max_mtime is not None and current_max_mtime <= cached_max_mtime:
            print("✓ Global cache valid (no changes detected)")
            return cache.reconstruct_face_data()

        # Pass 2: Incremental Scan
        updates = cache.get_incremental_updates(photos_path)
        photos_to_process = updates['new'] + updates['modified']

        if not photos_to_process:
            print("No individual files changed. Using cached data.")
            # Update global mtime since we just verified everything is up-to-date
            cache.set_global_mtime(current_max_mtime)
            return cache.reconstruct_face_data()

        print(f"Incremental update: {len(updates['new'])} new, {len(updates['modified'])} modified photos.")

        # Handle deleted files
        for deleted_path in updates['deleted']:
            cache.remove_photo_data(deleted_path)
    else:
        print("Force recompute enabled. Processing from scratch...")
        # In force recompute, we treat everything as new
        photos_to_process = load_photos(photos_path, verbose=verbose)

    # Process only necessary photos
    if photos_to_process:
        print(f"[1/4] Loading {len(photos_to_process)} photos...")
        # Use the target_photos argument we added to load_photos
        photos = load_photos(photos_path, verbose=verbose, target_photos=photos_to_process)

        print("\n[2/4] Detecting faces...")
        if use_cnn:
            print("Using CNN detector")
        face_data_incremental = detect_faces_batch(photos, verbose=verbose, use_cnn=use_cnn, parallel=parallel, downscale=downscale, log_queue=log_queue)

        print("\n[3/4] Generating encodings...")
        generate_face_encodings(face_data_incremental, verbose=verbose, parallel=parallel, log_queue=log_queue)

        print("\n[4/4] Updating cache...")
        # Update cache for ALL processed photos, even those with no faces
        for path_str in photos_to_process:
            p = Path(path_str)
            # Get faces for this photo if found, otherwise empty list
            faces = face_data_incremental.get(str(p.resolve()), [])
            cache.update_photo_data(p, p.stat().st_mtime, faces)

        # Update global marker after successful processing
        current_max_mtime = get_directory_modified_time(photos_path)

        if current_max_mtime:
            cache.set_global_mtime(current_max_mtime)

    # Final step: reconstruct full face_data from DB
    full_face_data = cache.reconstruct_face_data()
    print_detection_summary(full_face_data)
    print("\n✓ Pipeline complete!")

    return full_face_data



def main(
        photos_directory: str,
        force_cache_recompute: bool = False,
        verbose: bool = True,
        use_cnn: bool = False,
        downscale: bool = False,
        parallel: bool = False
) -> None:
    """
    The main pipeline for Photo Organizer tool

    Args:
        photos_directory: Path to folder containing photos
        force_cache_recompute: If True, ignore cache and regenerate
        verbose: If true, provide verbose output
        use_cnn: If true, use CNN detector
        downscale: If true, downscales the image for faster processing.
        parallel: If true, use ProcessPoolExecutor for detection and encoding
    """
    # Setup Logging
    log_queue, listener = setup_logging(verbose)

    try:
        print("\n" + "=" * 60)
        print(" PROGRAM CONFIGURATION")
        print("=" * 60)
        print(f"  Photos Directory : {photos_directory}")
        print(f"  Force Recompute  : {'YES' if force_cache_recompute else 'NO'}")
        print(f"  Verbose Mode     : {'ON' if verbose else 'OFF'}")
        print(f"  Use CNN Detector : {'YES' if use_cnn else 'NO'}")
        print(f"  Downscale Photos : {'YES' if downscale else 'NO'}")
        print(f"  Parallel Mode    : {'ON' if parallel else 'OFF'}")
        print("=" * 60)
        print("\nStarting pipeline in 4 seconds... (Press Ctrl+C to cancel)")
        time.sleep(4)
        print("\n")

        # Pass log_queue to load_face_data so it can pass it to batch functions
        face_data = load_face_data(photos_directory, force_cache_recompute, verbose, use_cnn, parallel, downscale, log_queue=log_queue)

        # Phase 4: Clustering
        print("\n" + "=" * 60)
        print("PHASE 4: CLUSTERING")
        print("=" * 60)

        # Extract encodings
        encodings_matrix, face_uuids, face_uuid_to_path_map = extract_encodings_for_clustering(face_data)

        # Cluster
        cluster_labels = cluster_faces(encodings_matrix, eps=0.35, min_samples=2)

        # Save text summary
        save_cluster_summary(face_data, face_uuids, cluster_labels)

        # Ask if user wants to visualize clusters first (optional)
        print("\nWould you like to visualize the clusters? (y/n)")
        response = input().strip().lower()

        if response == 'y':
            from src.clustering import visualize_clusters
            visualize_clusters(face_data, face_uuids, face_uuid_to_path_map, cluster_labels, max_faces_per_cluster=5)

        # Interactive naming and organization
        print("\n" + "=" * 60)
        print("PHOTO ORGANIZATION")
        print("=" * 60)
        print("\nWould you like to organize photos into named folders? (y/n)")
        response = input().strip().lower()

        if response == 'y':
            # Ask user to name clusters
            cluster_names = name_clusters_interactive(
                face_data, face_uuids, cluster_labels, face_uuid_to_path_map
            )

            # Organize photos into folders
            if cluster_names:
                organize_photos(
                    face_data, cluster_labels, face_uuids,
                    face_uuid_to_path_map, cluster_names,
                    output_dir="output"
                )
            else:
                print("\nNo clusters were named. Skipping organization.")
        else:
            print("\nSkipping photo organization.")

        print("\n✓ Pipeline complete!")
    finally:
        listener.stop()


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Photo organization tool")
    parser.add_argument("photos_directory", help="Directory containing photos")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--use-cnn", action="store_true", help="Use CNN detector (slower but more accurate)")
    parser.add_argument("--downscale", action="store_true", help="Downscale images for faster detection")
    parser.add_argument("--parallel", action="store_true", help="Use ProcessPoolExecutor for parallel detection and encoding")

    args = parser.parse_args()

    if args.parallel and args.use_cnn:
        print("\nProgram not yet configured to run CNN parallely due to logical & physical constraints.")
        exit()

    try:
        main(args.photos_directory, args.force_recompute, args.verbose, args.use_cnn, args.downscale, args.parallel)
    except KeyboardInterrupt as ke:
        print("\nProgram exited because of keyboard interrupt!")
    except Exception as e:
        print("\nUnexpected error!", e)
