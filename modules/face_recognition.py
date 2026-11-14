
import json
import re
from pathlib import Path

import cv2
import numpy as np

from config import (
    KNOWN_FACES_FOLDER,
    MODEL_FOLDER,
    FACE_MATCH_THRESHOLD
)

from database import Database


class FaceRecognition:

    def __init__(self):
        self.database = Database()

        self.known_faces_folder = Path(KNOWN_FACES_FOLDER)
        self.model_folder = Path(MODEL_FOLDER)

        self.known_faces_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        self.model_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        self.model_path = self.model_folder / "face_model.yml"
        self.labels_path = self.model_folder / "face_labels.json"

        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        if self.face_detector.empty():
            raise RuntimeError(
                "OpenCV face detector could not be loaded."
            )

        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "OpenCV face module is missing. "
                "Install opencv-contrib-python."
            )

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

        self.labels = {}
        self.is_trained = False

        self.load_model()

    # ======================================
    # VALIDATE PERSON NAME
    # ======================================

    @staticmethod
    def validate_person_name(person_name):

        if not isinstance(person_name, str):
            raise ValueError("Person name must be text.")

        person_name = person_name.strip()

        if not person_name:
            raise ValueError("Person name cannot be empty.")

        if len(person_name) > 50:
            raise ValueError("Person name is too long.")

        safe_name = re.sub(
            r"[^a-zA-Z0-9_-]",
            "_",
            person_name
        )

        return safe_name

    # ======================================
    # DETECT FACES
    # ======================================

    def detect_faces(self, image_path):

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError("Unable to read the image.")

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        return image, gray, faces

    # ======================================
    # REGISTER FACE
    # ======================================

    def register_face(self, person_name, image_path):

        safe_name = self.validate_person_name(person_name)

        image, gray, faces = self.detect_faces(image_path)

        if len(faces) == 0:
            raise ValueError("No face detected in the image.")

        person_folder = (
            self.known_faces_folder / safe_name
        )

        person_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        registered_count = 0

        for face_box in faces:

            x, y, width, height = face_box

            face = gray[
                y:y + height,
                x:x + width
            ]

            face = cv2.resize(
                face,
                (200, 200)
            )

            existing_files = list(
                person_folder.glob("face_*.jpg")
            )

            image_number = len(existing_files) + 1

            output_path = (
                person_folder /
                f"face_{image_number}.jpg"
            )

            if not cv2.imwrite(
                str(output_path),
                face
            ):
                raise RuntimeError(
                    "Could not save registered face."
                )

            self.database.add_face(
                person_name=person_name.strip(),
                image_path=str(output_path)
            )

            registered_count += 1

        return {
            "person_name": person_name.strip(),
            "registered_faces": registered_count,
            "message": "Face registration completed."
        }

    # ======================================
    # TRAIN FACE RECOGNIZER
    # ======================================

    def train_model(self):

        face_images = []
        face_labels = []

        self.labels = {}

        person_folders = sorted(
            folder
            for folder in self.known_faces_folder.iterdir()
            if folder.is_dir()
        )

        label_id = 0

        for person_folder in person_folders:

            image_files = sorted(
                person_folder.glob("face_*.jpg")
            )

            valid_images = []

            for image_path in image_files:

                image = cv2.imread(
                    str(image_path),
                    cv2.IMREAD_GRAYSCALE
                )

                if image is None:
                    continue

                image = cv2.resize(
                    image,
                    (200, 200)
                )

                valid_images.append(image)

            if not valid_images:
                continue

            self.labels[str(label_id)] = person_folder.name

            for image in valid_images:
                face_images.append(image)
                face_labels.append(label_id)

            label_id += 1

        if len(face_images) < 2:
            raise ValueError(
                "Register at least two face images before training."
            )

        self.recognizer.train(
            face_images,
            np.array(face_labels, dtype=np.int32)
        )

        self.recognizer.save(
            str(self.model_path)
        )

        with open(
            self.labels_path,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                self.labels,
                file,
                indent=4
            )

        self.is_trained = True

        return {
            "success": True,
            "registered_images": len(face_images),
            "registered_people": len(self.labels),
            "message": "Face recognition model trained."
        }

    # ======================================
    # LOAD TRAINED MODEL
    # ======================================

    def load_model(self):

        if not self.model_path.exists():
            return False

        if not self.labels_path.exists():
            return False

        try:
            self.recognizer.read(
                str(self.model_path)
            )

            with open(
                self.labels_path,
                "r",
                encoding="utf-8"
            ) as file:
                self.labels = json.load(file)

            self.is_trained = True

            return True

        except (cv2.error, OSError, ValueError, json.JSONDecodeError):
            self.is_trained = False
            self.labels = {}

            return False

    # ======================================
    # RECOGNIZE FACE
    # ======================================

    def recognize_face(self, image_path):

        if not self.is_trained:
            self.load_model()

        if not self.is_trained:
            raise ValueError(
                "Face model is not trained. "
                "Register faces and train the model first."
            )

        image, gray, faces = self.detect_faces(image_path)

        if len(faces) == 0:
            raise ValueError("No face detected.")

        results = []

        for face_box in faces:

            x, y, width, height = face_box

            face = gray[
                y:y + height,
                x:x + width
            ]

            face = cv2.resize(
                face,
                (200, 200)
            )

            label_id, distance = self.recognizer.predict(
                face
            )

            person_name = self.labels.get(
                str(label_id),
                "Unknown"
            )

            # LBPH returns a distance, not a probability.
            # A lower distance means a closer match.
            is_match = (
                distance <= 70
                and person_name != "Unknown"
            )

            results.append({
                "person_name": (
                    person_name if is_match else "Unknown"
                ),
                "distance": round(float(distance), 2),
                "box": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(width),
                    "height": int(height)
                }
            })

        return {
            "faces_detected": len(results),
            "results": results,
            "warning": (
                "Experimental face matching only. "
                "A match is not proof of identity."
            )
        }
