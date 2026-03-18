"""Face detection utilities."""
import uuid
from pathlib import Path
import face_recognition
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image, ImageShow
from PIL.Image import Resampling


def detect_faces_in_image(image_path: Path, use_cnn: bool = False) -> Optional[List[Tuple[int, int, int, int]]]:
    """
    Detect faces in a single image.

    Args:
        image_path: Path to image file
        use_cnn: Use CNN model for face detection (slower but more accurate)

    Returns:
        List of face bounding boxes as (top, right, bottom, left) tuples,
        or None if image loading/detection fails

    Note:
        Empty list means no faces found (successful detection, zero faces)
        None means detection couldn't run (file error, corrupted image)
    """
    try:
        image = face_recognition.load_image_file(str(image_path))

        model = 'cnn' if use_cnn else 'hog'
        face_locations = face_recognition.face_locations(image, model=model)

        return face_locations

    except FileNotFoundError:
        print(f"File not found: {str(image_path)}")
        return None

    except Exception as e:
        print(f"Error detecting faces in {str(image_path)}: {type(e).__name__}: {e}")
        return None


def show_face(path: Path, title: str = None, box: Optional[Tuple[float, float, float, float]] = None,
              scale: Tuple[int, int] = (300, 300), gray: bool = True):
    """Display a face image."""
    face = Image.open(str(path))

    if box:
        upper, right, lower, left = box
        face = face.crop((left, upper, right, lower))

    face = face.resize(scale, resample=Resampling.LANCZOS)

    if gray:
        face = face.convert("L")

    if title:
        print(f"\n>>> Showing: {title}")

    ImageShow.show(face)


def detect_faces_batch(photo_paths: List[Path], verbose: bool = True, show_photos: bool = False, use_cnn: bool = False) -> Dict[Path, List[Dict[str, Any]]]:
    """
    Detect faces in multiple photos.

    Args:
        photo_paths: List of image file paths
        verbose: Show progress messages
        show_photos: Show photos
        use_cnn: Use CNN model for face detection (slower but more accurate)

    Returns:
        Dictionary mapping photo path -> list of face bounding boxes
        Only includes photos where at least one face was detected
    """
    face_data = {}

    for i, photo_path in enumerate(photo_paths, 1):
        if verbose:
            print(f"Processing {i}/{len(photo_paths)}: {photo_path.name}")

        face_locations = detect_faces_in_image(photo_path, use_cnn=use_cnn)

        # Skip if detection failed (None) or no faces found (empty list)
        if face_locations is None:
            if verbose:
                print(f"  ⚠️ Detection failed")
            continue

        if len(face_locations) == 0:
            if verbose:
                print(f"  ℹ️ No faces found")
            continue

        face_data_list = []
        photo_name = photo_path.stem

        for j, loc in enumerate(face_locations):
             face_data_list.append({
                "face_id": f"photo_{photo_name}_image_{j}",
                "uuid": str(uuid.uuid4()),
                "bbox": loc,
                "encoding": None,
            })
        face_data[str(photo_path)] = face_data_list

        if verbose:
            print(f"  ✓ Found {len(face_locations)} face(s)")

        if show_photos:
            for face in face_locations:
                show_face(photo_path, box=face, title=str(i))

    return face_data


def print_detection_summary(face_data: Dict[Path, List]) -> None:
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
        print(f"  {Path(photo_path).name}: {len(faces)} face(s)")

    if total_photos > 5:
        print(f"  ... and {total_photos - 5} more photos")
