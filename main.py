"""
Photo organization tool for detecting and clustering faces.
Phase 2: Face detection implementation.
"""

import face_recognition
from PIL import Image, ImageShow
from PIL.Image import Resampling
import pathlib
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def load_photos(directory: pathlib.Path, verbose: bool = False) -> List[str]:
    """
    Recursively load all image file paths from directory.

    Args:
        directory: Root directory to search
        verbose: Print progress messages if True

    Returns:
        List of absolute paths to image files
    """
    photos = []

    if verbose:
        print(f"Scanning {directory}")

    try:
        for item in directory.iterdir():
            if item.is_dir():
                # Recursive call for subdirectories
                photos.extend(load_photos(item, verbose))
            elif item.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                # Only add image files
                photos.append(str(item))
    except PermissionError as e:
        print(f"Permission denied: {directory}")
    except Exception as e:
        print(f"Error scanning {directory}: {e}")

    return photos


def detect_faces_in_image(image_path: str) -> Optional[List[Tuple[int, int, int, int]]]:
    """
    Detect faces in a single image.

    Args:
        image_path: Path to image file

    Returns:
        List of face bounding boxes as (top, right, bottom, left) tuples,
        or None if image loading/detection fails

    Note:
        Empty list means no faces found (successful detection, zero faces)
        None means detection couldn't run (file error, corrupted image)
    """
    try:
        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        return face_locations

    except FileNotFoundError:
        print(f"File not found: {image_path}")
        return None

    except Exception as e:
        print(f"Error detecting faces in {image_path}: {type(e).__name__}: {e}")
        return None

def show_face(path: str, title: str = None, box: Optional[Tuple[float, float, float, float]] = None, scale: Tuple[int, int] = (300, 300), gray: bool = True):
    face = Image.open(path)
    if box:
        upper, right, lower, left = box
        face = face.crop((left, upper, right, lower))
    face = face.resize(scale)
    if gray:
        face = face.convert("L")

    ImageShow.show(face, title=title)


def detect_faces_batch(photo_paths: List[str], verbose: bool = True, show_photos: bool = False) -> Dict[str, List[Tuple[int, int, int, int]]]:
    """
    Detect faces in multiple photos.

    Args:
        photo_paths: List of image file paths
        verbose: Show progress messages
        show_photos: Show photos

    Returns:
        Dictionary mapping photo path -> list of face bounding boxes
        Only includes photos where at least one face was detected
    """
    face_data = {}

    for i, photo_path in enumerate(photo_paths, 1):
        if verbose:
            print(f"Processing {i}/{len(photo_paths)}: {pathlib.Path(photo_path).name}")

        face_locations = detect_faces_in_image(photo_path)

        # Skip if detection failed (None) or no faces found (empty list)
        if face_locations is None:
            if verbose:
                print(f"  ⚠️ Detection failed")
            continue

        if len(face_locations) == 0:
            if verbose:
                print(f"  ℹ️ No faces found")
            continue

        face_data[photo_path] = face_locations

        if verbose:
            print(f"  ✓ Found {len(face_locations)} face(s)")

        if show_photos:
            for face in face_locations:
                show_face(photo_path, box=face)

    return face_data


# def print_detection_summary(face_data: Dict[str, List]) -> None:
#     """
#     Print statistics about face detection results.
#
#     Args:
#         face_data: Dictionary from detect_faces_batch()
#     """
#     for i, (path, photo_locations) in enumerate(face_data.items()):
#         image = Image.open(path)
#         print(f"Photo {i + 1}: {path} has {len(photo_locations)}", "photos" if len(photo_locations) > 1 else "photo")
#         for photo_location in photo_locations:
#             top, right, bottom, left = photo_location
#             cropped_img = image.crop((left, top, right, bottom))
#             resized_img = cropped_img.resize((300, 300), resample=Resampling.LANCZOS)
#             grayscale_img = resized_img.convert("L")
#             ImageShow.show(grayscale_img)

def print_detection_summary(face_data: Dict[str, List]) -> None:
    """
    Print statistics about face detection results.

    Args:
        face_data: Dictionary from detect_faces_batch()
    """
    total_photos = len(face_data)
    total_faces = sum(len(faces) for faces in face_data.values())

    print("\n" + "=" * 50)
    print("FACE DETECTION SUMMARY")
    print("=" * 50)
    print()
    print(f"Photos with faces: {total_photos}")
    print(f"Total faces detected: {total_faces}")
    print(f"Average faces per photo: {total_faces / total_photos:.1f}")
    print("=" * 50 + "\n")

    # Optionally show sample results
    print("Sample results:")
    for i, (photo_path, faces) in enumerate(list(face_data.items())[:5]):
        print(f"  {pathlib.Path(photo_path).name}: {len(faces)} face(s)")

    if total_photos > 5:
        print(f"  ... and {total_photos - 5} more photos")


def main(photos_directory: str) -> None:
    """
    Main pipeline for face detection.

    Args:
        photos_directory: Path to folder containing photos
    """
    # 1. Load photos
    photos = load_photos(pathlib.Path(photos_directory))
    # 2. Detect faces
    faces = detect_faces_batch(photos, show_photos=False)
    # faces = detect_faces_batch(photos, show_photos=True)
    # 3. Print summary
    print_detection_summary(faces)
    # TODO: Return or save results




if __name__ == "__main__":
    # This only runs when script is executed directly
    # Not when imported as a module
    import sys

    if len(sys.argv) < 2:
        print("Usage: python main.py <photos_directory>")
        sys.exit(1)

    main(sys.argv[1])
