"""Face encoding utilities."""
import numpy as np
import face_recognition
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from src.loader import load_image_as_array
from src.logger import worker_logging_config


def generate_face_encoding(
        photo_path: Path,
        bbox: Tuple[int, int, int, int],
        num_jitters: int = 10
) -> Optional[np.ndarray]:
    """
    Generate 128-dimensional encoding for a single face.

    Args:
        photo_path: Path to image file
        bbox: Face bounding box as (top, right, bottom, left)
        num_jitters: Number of times to re-sample the face

    Returns:
        128-d numpy array encoding, or None if encoding fails
    """
    try:
        image = load_image_as_array(photo_path)
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

    except Exception:
        return None


def _encode_face_worker(args: Tuple[str, Tuple[int, int, int, int], str]) -> Tuple[str, str, Optional[np.ndarray]]:
    """
    Worker function for parallel face encoding.

    Args:
        args: Tuple of (photo_path, bbox, face_id)

    Returns:
        Tuple of (photo_path, face_id, encoding or None)
    """
    photo_path, bbox, face_id = args
    logger = logging.getLogger()

    logger.debug(f"Encoding face {face_id} in {Path(photo_path).name}")
    encoding = generate_face_encoding(Path(photo_path), bbox)

    if encoding is None:
        logger.warning(f"Encoding failed for face {face_id} in {Path(photo_path).name}")

    return photo_path, face_id, encoding


def generate_face_encodings(
    face_data: Dict[Path, List[Dict[str, Any]]],
    verbose: bool = True,
    parallel: bool = False,
    log_queue: Any = None,
) -> None:
    """
    Generate face encodings for all detected faces. Modifies face_data in place.

    Args:
        face_data: Dictionary (photo paths -> face bboxes)
        verbose: Show progress messages
        parallel: If True, use multiprocessing for parallel encoding
        log_queue: Queue for logging messages

    Returns:
        None (modifies face_data in place)
    """
    if parallel:
        _generate_face_encodings_parallel(face_data, verbose, log_queue=log_queue)
    else:
        _generate_face_encodings_sequential(face_data, verbose)


def _generate_face_encodings_sequential(
    face_data: Dict[Path, List[Dict[str, Any]]],
    verbose: bool,
) -> None:
    """Sequential face encoding with tqdm progress bar."""
    total_faces = sum(len(faces) for faces in face_data.values())
    successful = 0
    failed = 0
    logger = logging.getLogger()

    with tqdm(total=total_faces, desc="Encoding faces", ascii=True, leave=True) as pbar:
        for i, (path, faces) in enumerate(face_data.items(), 1):
            logger.debug(f"Encoding faces in photo {i}/{len(face_data)}: {Path(path).name}")

            for face in faces:
                face_id = face['face_id']
                bbox = face['bbox']

                encoding = generate_face_encoding(path, bbox)

                if encoding is not None:
                    face['encoding'] = encoding
                    successful += 1
                    logger.info(f"  ✓ {face_id}: encoding generated")
                else:
                    failed += 1
                    logger.warning(f"  ⚠️ {face_id}: encoding failed")

                pbar.update(1)

    logger.info(f"Encoding complete: {successful} successful, {failed} failed out of {total_faces} total\n")


def _generate_face_encodings_parallel(
    face_data: Dict[Path, List[Dict[str, Any]]],
    verbose: bool,
    log_queue: Any = None,
) -> None:
    """Parallel face encoding using ProcessPoolExecutor."""
    import os

    # Collect all faces that need encoding
    work_items = []
    for path, faces in face_data.items():
        for face in faces:
            work_items.append((str(path), face['bbox'], face['face_id']))

    total_faces = len(work_items)
    successful = 0
    failed = 0
    num_workers = min(os.cpu_count() or 4, 8)

    # Parallel initialization
    initializer = None
    if log_queue is not None:
        initializer = worker_logging_config
        init_args = (log_queue,)
    else:
        init_args = ()

    # Process with progress bar
    desc = "Encoding faces"
    if verbose:
        with tqdm(total=len(work_items), desc=desc, ascii=True, leave=True) as pbar:
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=initializer,
                initargs=init_args
            ) as executor:
                # Submit all work
                future_to_item = {
                    executor.submit(_encode_face_worker, item): item
                    for item in work_items
                }

                # Process results as they complete
                for future in as_completed(future_to_item):
                    photo_path, face_id, encoding = future.result()

                    if encoding is not None:
                        successful += 1
                        _update_face_encoding(face_data, photo_path, face_id, encoding)
                    else:
                        failed += 1

                    pbar.update(1)
    else:
        # Non-verbose: clean progress bar only, update face_data after
        results = []
        with tqdm(total=len(work_items), desc=desc, ascii=True, leave=False) as pbar:
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=initializer,
                initargs=init_args
            ) as executor:
                future_to_item = {
                    executor.submit(_encode_face_worker, item): item
                    for item in work_items
                }

                for future in as_completed(future_to_item):
                    photo_path, face_id, encoding = future.result()
                    results.append((photo_path, face_id, encoding))
                    pbar.update(1)

        # Update face_data with results
        for photo_path, face_id, encoding in results:
            if encoding is not None:
                successful += 1
                _update_face_encoding(face_data, photo_path, face_id, encoding)
            else:
                failed += 1

    logging.getLogger().info(f"Encoding complete: {successful} successful, {failed} failed out of {total_faces} total\n")


def _update_face_encoding(
    face_data: Dict[Path, List[Dict[str, Any]]],
    photo_path: str,
    face_id: str,
    encoding: np.ndarray,
) -> None:
    """Update a single face's encoding in face_data."""
    for path_key, faces in face_data.items():
        if str(path_key) == photo_path:
            for face in faces:
                if face['face_id'] == face_id:
                    face['encoding'] = encoding
                    return


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
