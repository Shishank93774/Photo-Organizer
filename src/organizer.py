"""Photo organization utilities for naming clusters and organizing photos."""
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np

from src.detector import show_face


def name_clusters_interactive(
    face_data: Dict[str, List[Dict]],
    face_uuids: List[str],
    cluster_labels: np.ndarray,
    face_uuid_to_path_map: Dict[str, str],
    max_faces_per_cluster: int = 3
) -> Dict[int, str]:
    """
    Interactively ask user to name each cluster by showing sample faces.

    Args:
        face_data: Dictionary mapping photo paths to face data
        face_uuids: List of face UUIDs (parallel to cluster_labels)
        cluster_labels: Cluster assignments from DBSCAN
        face_uuid_to_path_map: Mapping of UUIDs to photo paths
        max_faces_per_cluster: Number of sample faces to show per cluster

    Returns:
        Dictionary mapping cluster_id to user-provided name
    """
    # Group faces by cluster
    clusters = defaultdict(list)
    for uuid, cluster_id in zip(face_uuids, cluster_labels):
        clusters[cluster_id].append(uuid)

    # Sort cluster IDs (regular clusters first, noise last)
    cluster_ids = sorted([cid for cid in clusters.keys() if cid != -1])
    if -1 in clusters:
        cluster_ids.append(-1)

    cluster_names = {}

    print("\n" + "=" * 60)
    print("NAMING CLUSTERS")
    print("=" * 60)
    print("\nI'll show you sample faces from each cluster.")
    print("Enter a name for each person, or press Enter to skip.\n")

    for cluster_id in cluster_ids:
        # Skip noise cluster
        if cluster_id == -1:
            print(f"\n[SKIPPING] Cluster -1 (Noise/Strangers) - {len(clusters[cluster_id])} faces")
            print("  These faces will be placed in 'unorganized' folder.")
            continue

        cluster_uuids = clusters[cluster_id]
        print(f"\n{'=' * 60}")
        print(f"CLUSTER {cluster_id} - {len(cluster_uuids)} faces total")
        print(f"{'=' * 60}")

        # Show sample faces
        sample_uuids = cluster_uuids[:max_faces_per_cluster]
        print(f"Showing {len(sample_uuids)} sample faces:\n")

        for i, uuid in enumerate(sample_uuids, 1):
            photo_path = face_uuid_to_path_map[uuid]
            faces = face_data[photo_path]
            for face in faces:
                if face['uuid'] == uuid:
                    bbox = face['bbox']
                    photo_name = Path(photo_path).stem
                    title = f"Cluster {cluster_id} | Face {i}/{len(sample_uuids)} | Photo: {photo_name}"
                    print(f"  [{i}] {photo_name}")
                    show_face(photo_path, title=title, box=bbox, gray=False)

        # Prompt for name
        while True:
            name = input(f"\nEnter name for Person {cluster_id + 1} (or press Enter to skip): ").strip()
            if name == "":
                print(f"  [SKIPPED] Cluster {cluster_id} will go to 'unorganized'")
                break
            # Sanitize name for folder creation
            safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip().capitalize()
            if safe_name:
                cluster_names[cluster_id] = safe_name
                print(f"  [NAMED] Cluster {cluster_id} -> '{safe_name}'")
                break
            else:
                print("  Invalid name. Please use alphanumeric characters.")

        # Wait before next cluster (unless it's the last one)
        if cluster_id != cluster_ids[-1] and cluster_ids[-1] == -1:
            remaining = len([cid for cid in cluster_ids if cid > cluster_id and cid != -1])
            if remaining > 0:
                input("\nPress Enter to see next cluster...")

    print("\n" + "=" * 60)
    print("NAMING COMPLETE")
    print("=" * 60)
    print(f"\nNamed clusters: {len(cluster_names)}")
    print(f"Skipped clusters: {len(cluster_ids) - len(cluster_names) - (1 if -1 in clusters else 0)}")

    return cluster_names


def organize_photos(
    face_data: Dict[str, List[Dict]],
    cluster_labels: np.ndarray,
    face_uuids: List[str],
    face_uuid_to_path_map: Dict[str, str],
    cluster_names: Dict[int, str],
    output_dir: str = "output"
) -> None:
    """
    Organize photos into folders based on cluster names.

    Args:
        face_data: Dictionary mapping photo paths to face data
        cluster_labels: Cluster assignments from DBSCAN
        face_uuids: List of face UUIDs (parallel to cluster_labels)
        face_uuid_to_path_map: Mapping of UUIDs to photo paths
        cluster_names: Dictionary mapping cluster_id to folder name
        output_dir: Base output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Build mapping: photo_path -> set of cluster_ids
    photo_to_clusters: Dict[str, Set[int]] = defaultdict(set)

    for uuid, cluster_id in zip(face_uuids, cluster_labels):
        photo_path = face_uuid_to_path_map[uuid]
        photo_to_clusters[photo_path].add(cluster_id)

    # Track which photos go to which folders
    folder_photos: Dict[str, Set[str]] = defaultdict(set)  # folder_name -> set of photo paths
    unorganized_photos: Set[str] = set()

    for photo_path, cluster_ids in photo_to_clusters.items():
        # Check if any cluster is named
        named_clusters = [cid for cid in cluster_ids if cid in cluster_names]

        if named_clusters:
            # Photo goes to all named cluster folders
            for cluster_id in named_clusters:
                folder_name = cluster_names[cluster_id]
                folder_photos[folder_name].add(photo_path)
        else:
            # Photo goes to unorganized
            unorganized_photos.add(photo_path)

    # Create folders and copy photos
    print("\n" + "=" * 60)
    print("ORGANIZING PHOTOS")
    print("=" * 60)

    total_copied = 0

    # Copy photos to named folders
    for folder_name, photo_paths in sorted(folder_photos.items()):
        folder_path = output_path / folder_name

        # Handle duplicate folder names
        counter = 1
        while folder_path.exists():
            folder_path = output_path / f"{folder_name}_{counter}"
            counter += 1

        folder_path.mkdir(exist_ok=True)

        for photo_path in photo_paths:
            src = Path(photo_path)
            dst = folder_path / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                total_copied += 1

        print(f"\n{folder_name}/")
        print(f"  {len(photo_paths)} photo(s) copied")

    # Create unorganized folder if needed
    if unorganized_photos:
        unorganized_path = output_path / "unorganized"
        unorganized_path.mkdir(exist_ok=True)

        for photo_path in unorganized_photos:
            src = Path(photo_path)
            dst = unorganized_path / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
                total_copied += 1

        print(f"\nunorganized/")
        print(f"  {len(unorganized_photos)} photo(s) copied")

    # Save summary
    save_organization_summary(cluster_names, folder_photos, unorganized_photos, output_path)

    print(f"\n✓ Total photos copied: {total_copied}")
    print(f"✓ Output directory: {output_path.absolute()}")


def save_organization_summary(
    cluster_names: Dict[int, str],
    folder_photos: Dict[str, Set[str]],
    unorganized_photos: Set[str],
    output_path: Path
) -> None:
    """
    Save a summary file of the organization.

    Args:
        cluster_names: Dictionary mapping cluster_id to folder name
        folder_photos: Dictionary mapping folder name to set of photo paths
        unorganized_photos: Set of photo paths in unorganized folder
        output_path: Output directory path
    """
    summary_file = output_path / "organization_summary.txt"

    with open(summary_file, 'w') as f:
        f.write("PHOTO ORGANIZATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        f.write("NAMED FOLDERS\n")
        f.write("-" * 40 + "\n\n")

        for folder_name in sorted(folder_photos.keys()):
            photo_count = len(folder_photos[folder_name])
            f.write(f"{folder_name}/\n")
            f.write(f"  {photo_count} photo(s)\n\n")

        if unorganized_photos:
            f.write("UNORGANIZED\n")
            f.write("-" * 40 + "\n\n")
            f.write(f"unorganized/\n")
            f.write(f"  {len(unorganized_photos)} photo(s)\n\n")

        f.write("=" * 60 + "\n")

    print(f"\n✓ Summary saved to {summary_file}")