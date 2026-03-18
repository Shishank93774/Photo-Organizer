"""Face encoding utilities."""
import numpy as np
import face_recognition
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Pattern for encodings files: encodings_YYYY_MM_DD_HH_MM_SS.pkl
ENCODING_PATTERN = r"^encodings_\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}\.pkl$"


def generate_face_encoding(photo_path: Path, bbox: Tuple[int, int, int, int],
                           num_jitters: int = 10) -> Optional[np.ndarray]:
    """
    Generate 128-dimensional encoding for a single face.

    Args:
        photo_path: Path to image file
        bbox: Face bounding box as (top, right, bottom, left)

    Returns:
        128-d numpy array encoding, or None if encoding fails
    """
    try:
        image = face_recognition.load_image_file(str(photo_path))
        encodings = face_recognition.face_encodings(
            image,
            known_face_locations=[bbox],
            num_jitters=num_jitters
        )

        # face_encodings returns a list - take first element
        if len(encodings) > 0:
            return encodings[0]  # ✅ Return the array, not the list
        else:
            # Sometimes encoding fails (blurry face, weird angle)
            return None

    except Exception as e:
        print(f"Error generating encoding: {e}")
        return None


def generate_face_encodings(face_data: Dict[Path, List[Dict[str, Any]]], verbose: bool = True) -> None:
    """
    Generate face encodings for all detected faces. Modifies face_data in place.

    Args:
        face_data: Dictionary (photo paths -> face bboxes)
        verbose: Show progress messages

    Returns:
        None (modifies face_data in place)
    """
    total_faces = sum(len(faces) for faces in face_data.values())
    successful = 0
    failed = 0

    for i, (path, faces) in enumerate(face_data.items(), 1):
        if verbose:
            print(f"\nEncoding faces in photo {i}/{len(face_data)}: {Path(path).name}")

        for face in faces:
            face_id = face['face_id']
            bbox = face['bbox']

            encoding = generate_face_encoding(path, bbox)

            if encoding is not None:
                face['encoding'] = encoding
                successful += 1
                if verbose:
                    print(f"  ✓ {face_id}: encoding generated")
            else:
                failed += 1
                if verbose:
                    print(f"  ⚠️ {face_id}: encoding failed")

    print(f"\nEncoding complete: {successful} successful, {failed} failed out of {total_faces} total\nn")


def validate_encodings(face_data: Dict) -> Dict[str, int]:
    """Check encoding quality."""
    total = valid = invalid = 0

    for photo, faces in face_data.items():
        for face in faces:
            total += 1
            encoding = face.get('encoding')

            if encoding is None or not isinstance(encoding, np.ndarray):
                invalid += 1
            elif encoding.shape != (128,):
                invalid += 1
            else:
                valid += 1

    return {'total': total, 'valid': valid, 'invalid': invalid}
