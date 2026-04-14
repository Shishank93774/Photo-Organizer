import numpy as np
from typing import Dict, Tuple, List
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import DBSCAN
import logging

from src.detector import show_face

def extract_encodings_for_clustering(face_data: Dict[Path, List[Dict]]) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Extract face encodings and IDs for clustering.

    Args:
        face_data: Dictionary with face detections and encodings

    Returns:
        Tuple of:
            - encodings_matrix: np.ndarray of shape (N, 128)
            - face_uuids: List of UUIDs corresponding to each row
            - face_uuid_to_path_map: Dict of uuid and it's corresponding photo path

    Note:
        Skips faces with encoding=None
    """
    encodings_list = []
    face_uuids = []

    face_uuid_to_path_map = {}
    logger = logging.getLogger()

    for photo_path, face_list in face_data.items():
        for face in face_list:
            encoding = face.get('encoding')

            # Skip faces without valid encodings
            if encoding is None or not isinstance(encoding, np.ndarray):
                continue

            encodings_list.append(encoding)
            face_uuids.append(face['uuid'])

            face_uuid_to_path_map[face['uuid']] = photo_path # Storing mapping for quicker lookup later

    # Convert list of arrays to 2D matrix
    encodings_matrix = np.array(encodings_list)

    logger.info(f"Extracted {len(face_uuids)} valid encodings for clustering")
    logger.debug(f"Matrix shape: {encodings_matrix.shape}")

    return encodings_matrix, face_uuids, face_uuid_to_path_map


def cluster_faces(encodings_matrix: np.ndarray, eps: float = 0.4, min_samples: int = 2) -> np.ndarray:
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
    logger = logging.getLogger()
    # Handle cases where no faces were detected
    if encodings_matrix.size == 0:
        print("\nNo faces found to cluster.")
        return np.array([], dtype=int)

    print(f"\nClustering {encodings_matrix.shape[0]} faces...")
    logger.debug(f"Parameters: eps={eps}, min_samples={min_samples}")

    clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    cluster_labels = clusterer.fit_predict(encodings_matrix)

    # Analyze results
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)

    print("\nClustering Results:")
    print(f"  Distinct people found: {n_clusters}")
    print(f"  Noise points (strangers): {n_noise}")
    if len(cluster_labels) > 0:
        print(f"  Noise percentage: {n_noise / len(cluster_labels) * 100:.1f}%")

    # Show cluster sizes
    if n_clusters > 0:
        print("\nCluster sizes:")
        for cluster_id in range(n_clusters):
            count = list(cluster_labels).count(cluster_id)
            print(f"  Cluster {cluster_id}: {count} faces")

    return cluster_labels


def visualize_clusters(face_data: Dict[Path, List[Dict]],
                       face_uuids: List[str],
                       face_uuid_to_path_map: Dict,
                       cluster_labels: np.ndarray,
                       max_faces_per_cluster: int = 5):
    """
    Visualize sample faces from each cluster.

    Args:
        face_data: Original face data with bbox info
        face_uuids: List of face UUIDs (parallel to cluster_labels)
        face_uuid_to_path_map: Mapping of UUIDs to actual photo Path for quicker lookup
        cluster_labels: Cluster assignments from DBSCAN
        max_faces_per_cluster: Maximum number of sample faces to show per cluster
    """
    print("\n" + "=" * 60)
    print("VISUALIZING CLUSTERS")
    print("=" * 60)

    # Group faces by cluster
    clusters = defaultdict(list)

    for uuid, cluster_id in zip(face_uuids, cluster_labels):
        clusters[cluster_id].append(uuid)

    # Sort cluster IDs (show regular clusters first, noise last)
    cluster_ids = sorted([cid for cid in clusters.keys() if cid != -1])
    if -1 in clusters:
        cluster_ids.append(-1)

    # Visualize each cluster
    for cluster_id in cluster_ids:
        cluster_uuids = clusters[cluster_id]

        if cluster_id == -1:
            print(f"\n{'=' * 60}")
            print(f"CLUSTER {cluster_id} (NOISE/STRANGERS) - {len(cluster_uuids)} faces")
            print(f"{'=' * 60}")
        else:
            print(f"\n{'=' * 60}")
            print(f"CLUSTER {cluster_id} (PERSON {cluster_id + 1}) - {len(cluster_uuids)} faces")
            print(f"{'=' * 60}")

        # Show sample faces
        sample_uuids = cluster_uuids[:max_faces_per_cluster]

        print(f"Showing {len(sample_uuids)} sample faces:")

        for i, uuid in enumerate(sample_uuids, 1):
            photo_path = face_uuid_to_path_map[uuid]
            faces = face_data[photo_path]
            for face in faces:
                if face['uuid'] == uuid:
                    bbox = face['bbox']
                    photo_name = Path(photo_path).stem

                    title = f"Cluster {cluster_id} | Face {i}/{len(sample_uuids)} | Photo: {photo_name}"

                    print(f"  [{i}] {photo_name} - {uuid}")
                    show_face(photo_path, title=title, box=bbox, gray=False)

        if len(cluster_uuids) > max_faces_per_cluster:
            print(f"  ... and {len(cluster_uuids) - max_faces_per_cluster} more faces in this cluster")

        # Wait for user before showing next cluster
        if cluster_id != cluster_ids[-1]:
            input("\nPress Enter to see next cluster...")


def save_cluster_summary(face_data: Dict[Path, List[Dict]],
                         face_uuids: List[str],
                         cluster_labels: np.ndarray,
                         output_file: str = "cluster_summary.txt"):
    """
    Save a text summary of clustering results.

    Args:
        face_data: Original face data
        face_uuids: List of face UUIDs
        cluster_labels: Cluster assignments
        output_file: Path to output file
    """
    clusters = defaultdict(list)

    for uuid, cluster_id in zip(face_uuids, cluster_labels):
        # Find photo path for this uuid
        for photo_path, faces in face_data.items():
            for face in faces:
                if face['uuid'] == uuid:
                    clusters[cluster_id].append({
                        'uuid': uuid,
                        'photo': Path(photo_path).name
                    })
                    break

    with open(output_file, 'w') as f:
        f.write("CLUSTERING SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        # Regular clusters
        regular_clusters = sorted([cid for cid in clusters.keys() if cid != -1])

        for cluster_id in regular_clusters:
            faces = clusters[cluster_id]
            f.write(f"CLUSTER {cluster_id} (PERSON {cluster_id + 1})\n")
            f.write(f"Total faces: {len(faces)}\n")
            f.write(f"Photos:\n")

            # Group by photo
            photos = defaultdict(list)
            for face_info in faces:
                photos[face_info['photo']].append(face_info['uuid'])

            for photo, uuids in sorted(photos.items()):
                f.write(f"  {photo}: {len(uuids)} face(s)\n")

            f.write("\n")

        # Noise cluster
        if -1 in clusters:
            faces = clusters[-1]
            f.write(f"CLUSTER -1 (NOISE/STRANGERS)\n")
            f.write(f"Total faces: {len(faces)}\n")
            f.write(f"Photos:\n")

            for face_info in faces:
                f.write(f"  {face_info['photo']}: {face_info['uuid']}\n")

    logging.getLogger().info(f"Cluster summary saved to {output_file}")
