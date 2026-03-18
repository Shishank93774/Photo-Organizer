"""Main entry point for photo organization tool."""
import argparse
from pathlib import Path
from datetime import datetime

from src.loader import load_photos
from src.detector import detect_faces_batch, print_detection_summary
from src.encoder import generate_face_encodings, validate_encodings
from src.cache import save_cache, load_cache
from src.clustering import extract_encodings_for_clustering, cluster_faces, save_cluster_summary
from src.test_clustering import test_clustering_parameters
from src.diagnosis import analyze_encoding_quality, analyze_detection_quality


def get_directory_modified_time(directory_path: Path): # captures any add or deletes in the photo_directory folder
    """Get the directory's and it's sub-directory's  last modified time"""

    if not directory_path.exists() or not directory_path.is_dir():
        return None

    max_mod_time = directory_path.stat().st_mtime

    for dir in directory_path.iterdir():
        if dir.is_dir():
            max_mod_time = max(max_mod_time, get_directory_modified_time(dir))

    return max_mod_time


def load_face_data(photos_directory: str, force_cache_recompute: bool = False, verbose: bool = True, use_cnn: bool = False) -> dict:
    """"
    Loads faces from given photo directory and returns face_data with encodings

    Args:
        photos_directory: Path to folder containing photos
        force_cache_recompute: If True, ignore cache and regenerate
        verbose: If true, provide verbose output
        use_cnn: If True, use CNN detector

    Returns:
        face_data: Dictionary with face detections and encodings
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
    if use_cnn:
        print("Using CNN detector")
    face_data = detect_faces_batch(photos, verbose=verbose, use_cnn=use_cnn)
    print_detection_summary(face_data)

    print("\n[3/4] Generating encodings...")
    generate_face_encodings(face_data, verbose=verbose)

    print("\n[4/4] Validating and saving...")
    stats = validate_encodings(face_data)
    print(f"\nEncoding quality: {stats['valid']}/{stats['total']} valid")

    save_cache(cache_path=cache_path, face_data=face_data, now=now)
    print("\n\n✓ Pipeline complete!")

    return face_data


def main(photos_directory: str, force_cache_recompute: bool = False, verbose: bool = True, use_cnn: bool = False) -> None:
    """
    The main pipeline for Photo Organizer tool

    Args:
        photos_directory: Path to folder containing photos
        force_cache_recompute: If True, ignore cache and regenerate
        verbose: If true, provide verbose output
        use_cnn: If true, use CNN detector
    """
    # Phase 1-3: Load, detect, encode
    face_data = load_face_data(photos_directory, force_cache_recompute, verbose)

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

    # Visualize clusters
    print("\nWould you like to visualize the clusters? (y/n)")
    response = input().strip().lower()

    if response == 'y':
        from src.clustering import visualize_clusters
        visualize_clusters(face_data, face_uuids, face_uuid_to_path_map, cluster_labels, max_faces_per_cluster=10)

    print("\n✓ Clustering complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Photo organization tool")
    parser.add_argument("photos_directory", help="Directory containing photos")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--force-recompute", action="store_true")
    parser.add_argument("--use-cnn", action="store_true", help="Use CNN detector (slower but more accurate)")

    args = parser.parse_args()
    try:
        main(args.photos_directory, args.force_recompute, args.verbose, args.use_cnn)
    except KeyboardInterrupt as ke:
        print("\nProgram exited because of keyboard interrupt!")
    except Exception as e:
        print("\nUnexpected error!", e)
