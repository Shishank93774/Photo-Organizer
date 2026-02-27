import face_recognition
from PIL import Image, ImageShow
import numpy as np
from PIL.Image import Resampling
from PIL.ImageOps import grayscale
from sklearn.cluster import DBSCAN
import cv2
import pathlib
from collections import defaultdict

photo_map = defaultdict()


def photo_loader(path_to_images_dir: pathlib.Path) -> list:

    photos = []
    print(f"Loading files in {path_to_images_dir}")
    try:
        for obj in path_to_images_dir.iterdir():
            if obj.is_dir():
                photos_from_subfolder = photo_loader(obj.absolute())
                for photo in photos_from_subfolder:
                    photos.append(photo)
            elif obj.name.endswith(".jpg") or obj.name.endswith(".png") or obj.name.endswith(".jpeg"):
                photos.append(str(obj.absolute()))
    except Exception as e:
        print("Error-", e)

    return photos

photos = photo_loader(pathlib.Path("D:\\DesktopD\\Projects\\PhotoOrganizer\\dlib"))


def detect_faces(img_path: str) -> list|None:
    try:
        img = face_recognition.load_image_file(img_path)
        return face_recognition.face_locations(img=img)
    except Exception as e:
        print("Error-", e)

for photo in photos:
    if photo is None:
        continue
    face_locations = detect_faces(photo)
    if face_locations is None or len(face_locations) == 0:
        continue

    photo_map[photo] = face_locations


print("Processing photos:...")
for i, (path, photo_locations) in enumerate(photo_map.items()):
    image = Image.open(path)
    print(f"Photo {i+1}: {path} has {len(photo_locations)}", "photos" if len(photo_locations) > 1 else "photo")
    for photo_location in photo_locations:
        top, right, bottom, left = photo_location
        cropped_img = image.crop((left, top, right, bottom))
        resized_img = cropped_img.resize((300, 300), resample=Resampling.LANCZOS)
        grayscale_img = resized_img.convert("L")
        ImageShow.show(grayscale_img)
