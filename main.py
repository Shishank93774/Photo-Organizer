"""Main entry point for photo organization tool."""
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

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


def load_face_data(photos_directory: str, force_cache_recompute: bool = False, verbose: bool = True) -> dict:
    """"
    Loads faces from given photo directory and returns face_data with encodings

    Args:
        photos_directory: Path to folder containing photos
        force_cache_recompute: If True, ignore cache and regenerate
        verbose: If true, provide verbose output

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


def extract_encodings_for_clustering(face_data: Dict[str, List[Dict]]) -> Tuple[np.ndarray, Dict]:
    """
    Extract face encodings and IDs for clustering.

    Args:
        face_data: Dictionary with face detections and encodings

    Returns:
        Tuple of:
            - encodings_matrix: np.ndarray of shape (N, 128)
            - face_ids: List of face IDs corresponding to each row

    Note:
        Skips faces with encoding=None
    """

    uuid_to_photo_map = {}
    encoding_mat = []

    for photo_path, face_list in face_data.items():
        for face in face_list:
            uuid = face['uuid']
            uuid_to_photo_map[uuid] = photo_path
            encoding_mat.append(np.append(np.array(face['encoding']), [uuid]))

    encoding_mat = np.array(encoding_mat)

    return encoding_mat, uuid_to_photo_map


def cluster_faces(encodings_matrix: np.ndarray, eps: float = 15, min_samples: int = 2) -> np.ndarray:
    """
    Cluster face encodings using DBSCAN.

    Args:
        encodings_matrix: np.ndarray of shape (N, 128)
        eps: Maximum distance between neighbors (default 0.6 for faces)
        min_samples: Minimum cluster size

    Returns:
        Array of cluster labels, one per face
        Label -1 indicates noise/outlier
    """
    X, y = encodings_matrix[:,:-1], encodings_matrix[:,-1:]

    X_scaled = StandardScaler().fit_transform(X)

    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = clusterer.fit_predict(X_scaled)

    # How many clusters found?
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    print(f"Found {n_clusters} distinct people")
    print(f"Noise points (strangers): {n_noise}")


def main(photos_directory: str, force_cache_recompute: bool = False, verbose: bool = True) -> None:
    """
    The main pipeline for Photo Organizer tool

    Args:
        photos_directory: Path to folder containing photos
        force_cache_recompute: If True, ignore cache and regenerate
        verbose: If true, provide verbose output

    """
    face_data = load_face_data(photos_directory, force_cache_recompute, verbose)

    encodings, id_to_photo_map = extract_encodings_for_clustering(face_data=face_data)

    cluster_faces(encodings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Photo organization tool")
    parser.add_argument("photos_directory", help="Directory containing photos")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--force-recompute", action="store_true")

    args = parser.parse_args()
    main(args.photos_directory, args.force_recompute, args.verbose)
