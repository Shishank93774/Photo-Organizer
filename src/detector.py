"""Face detection utilities."""
import uuid
import logging
import face_recognition
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image, ImageShow
from PIL.Image import Resampling
from tqdm import tqdm

from src.loader import load_image_as_array
from src.logger import worker_logging_config


def detect_faces_in_image(
        image_path: Path,
        use_cnn: bool = False,
        downscale: bool = False
) -> Optional[List[Tuple[int, int, int, int]]]:
    """
    Detect faces in a single image.

    Args:
        image_path: Path to image file
        use_cnn: Use CNN model for face detection (slower but more accurate)
        downscale: If True, downscale the image for faster detection (especially for CNN)

    Returns:
        List of face bounding boxes as (top, right, bottom, left) tuples,
        or None if image loading/detection fails

    Note:
        Empty list means no faces found (successful detection, zero faces)
        None means detection couldn't run (file error, corrupted image)
    """
    try:
        image = load_image_as_array(image_path)

        scale_factor = 1.0
        if downscale:
            # Downscale to a maximum width of 1000px for faster detection
            h, w = image.shape[:2]
            if w > 1000:
                scale_factor = 1000 / w
                from PIL import Image
                pil_img = Image.fromarray(image)
                new_h = int(h * scale_factor)
                pil_img = pil_img.resize((1000, new_h), resample=Image.Resampling.LANCZOS)
                image = np.array(pil_img)

        model = 'cnn' if use_cnn else 'hog'
        face_locations = face_recognition.face_locations(image, model=model)

        # Scale bounding boxes back up if we downscaled
        if scale_factor != 1.0:
            face_locations = [
                (int(top / scale_factor), int(right / scale_factor), int(bottom / scale_factor), int(left / scale_factor))
                for (top, right, bottom, left) in face_locations
            ]

        return face_locations

    except FileNotFoundError:
        return None

    except Exception:
        return None


def _detect_faces_worker(args: Tuple[Path, bool]) -> Tuple[Path, Optional[List[Dict[str, Any]]]]:
    """
    Worker function for parallel face detection. Processes a single photo.

    Args:
        args: Tuple of (photo_path, use_cnn)

    Returns:
        Tuple of (photo_path, face_data_list or None)
        face_data_list is None if detection failed
    """
    photo_path, use_cnn = args
    logger = logging.getLogger()

    logger.debug(f"Processing {photo_path.name} ({round(photo_path.stat().st_size / 1048576, 1)} MB)")
    face_locations = detect_faces_in_image(photo_path, use_cnn=use_cnn)

    # Skip if detection failed (None) or no faces found (empty list)
    if face_locations is None:
        logger.warning(f"Detection failed for {photo_path.name}")
        return photo_path, None

    if len(face_locations) == 0:
        logger.debug(f"No faces found in {photo_path.name}")
        return photo_path, []

    face_data_list = []
    photo_name = photo_path.stem

    for j, loc in enumerate(face_locations):
        face_data_list.append({
            "face_id": f"photo_{photo_name}_image_{j}",
            "uuid": str(uuid.uuid4()),
            "bbox": loc,
            "encoding": None,
        })

    logger.debug(f"Found {len(face_locations)} face(s) in {photo_path.name}")
    return photo_path, face_data_list


def detect_faces_batch(
    photo_paths: List[Path],
    show_photos: bool = False,
    use_cnn: bool = False,
    parallel: bool = False,
    downscale: bool = False,
    log_queue: Any = None,
) -> Dict[Path, List[Dict[str, Any]]]:
    """
    Detect faces in multiple photos.

    Args:
        photo_paths: List of image file paths
        show_photos: Show photos
        use_cnn: Use CNN model for face detection (slower but more accurate)
        parallel: If True, use multiprocessing for parallel detection
        downscale: If True, downscale images for faster detection
        log_queue: Queue for logging messages
    """

    if parallel:
        # Parallel mode: use ProcessPoolExecutor
        face_data = _detect_faces_parallel(photo_paths, use_cnn, log_queue=log_queue)
    else:
        # Sequential mode
        face_data = _detect_faces_sequential(photo_paths, use_cnn, downscale=downscale)

    if show_photos:
        for photo_path, faces in face_data.items():
            for face in faces:
                show_face(photo_path, box=face['bbox'], title=photo_path.name)

    return face_data


def _detect_faces_sequential(
    photo_paths: List[Path],
    use_cnn: bool,
    downscale: bool = False,
) -> Dict[Path, List[Dict[str, Any]]]:
    """Sequential face detection with tqdm progress bar."""
    face_data = {}
    logger = logging.getLogger()

    with tqdm(total=len(photo_paths), desc="Detecting faces", ascii=True, leave=True) as pbar:
        for i, photo_path in enumerate(photo_paths, 1):
            logger.debug(f"Processing {i}/{len(photo_paths)}: {photo_path.name} ({round(photo_path.stat().st_size/1048576, 1)} MB)")

            face_locations = detect_faces_in_image(photo_path, use_cnn=use_cnn, downscale=downscale)

            # Skip if detection failed (None) or no faces found (empty list)
            if face_locations is None:
                logger.warning(f"Detection failed for {photo_path.name}")
                pbar.update(1)
                continue

            if len(face_locations) == 0:
                logger.debug(f"No faces found in {photo_path.name}")
                pbar.update(1)
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
            face_data[str(photo_path.resolve())] = face_data_list

            logger.info(f"✓ Found {len(face_locations)} face(s) in {photo_path.name}")
            pbar.update(1)

    return face_data


def _detect_faces_parallel(
    photo_paths: List[Path],
    use_cnn: bool,
    log_queue: Any = None,
) -> Dict[Path, List[Dict[str, Any]]]:
    """Parallel face detection using ProcessPoolExecutor."""
    import os

    face_data = {}
    num_workers = min(os.cpu_count() or 4, 6)

    # Prepare work items
    work_items = [(photo_path, use_cnn) for photo_path in photo_paths]

    # Parallel initialization
    initializer = None
    if log_queue is not None:
        initializer = worker_logging_config
        init_args = (log_queue,)
    else:
        init_args = ()

    # Use tqdm to show progress
    desc = "Detecting faces"
    with tqdm(total=len(work_items), desc=desc, ascii=True, leave=False) as pbar:
        with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=initializer,
                initargs=init_args
        ) as executor:
            futures = {executor.submit(_detect_faces_worker, item): item[0] for item in work_items}

            for future in as_completed(futures):
                photo_path, face_data_list = future.result()

                if face_data_list is not None and len(face_data_list) > 0:
                    face_data[str(photo_path.resolve())] = face_data_list

                pbar.update(1)

    return face_data


def show_face(
        path: Path,
        title: str = None,
        box: Optional[Tuple[float, float, float, float]] = None,
        scale: Tuple[int, int] = (300, 300), gray: bool = True
):
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
    if total_photos > 0:
        print(f"Average faces per photo: {total_faces / total_photos:.1f}")
    print("=" * 50 + "\n")

    # Optionally show sample results
    print("Sample results:")
    for i, (photo_path, faces) in enumerate(list(face_data.items())[:5]):
        print(f"  {Path(photo_path).name}: {len(faces)} face(s)")

    if total_photos > 5:
        print(f"  ... and {total_photos - 5} more photos")
