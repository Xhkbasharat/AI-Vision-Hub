
import json
import logging
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


logger = logging.getLogger("AI-Vision-Hub")


class FaceRecognition:
    """
    Face registration, model training, and recognition
    using OpenCV Haar Cascade and LBPH.
    """

    def __init__(self):

        self.known_faces_folder = Path(
            KNOWN_FACES_FOLDER
        )

        self.model_folder = Path(
            MODEL_FOLDER
        )

        self.model_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        self.known_faces_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        self.model_path = (
            self.model_folder / "face_model.yml"
        )

        self.labels_path = (
            self.model_folder / "face_labels.json"
        )

        self.database = Database()

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        if self.face_cascade.empty():

            raise RuntimeError(
                "Unable to load Haar Cascade classifier."
            )

        if not hasattr(cv2, "face"):

            raise RuntimeError(
                "OpenCV contrib is required for LBPH recognition."
            )

        self.recognizer = (
            cv2.face.LBPHFaceRecognizer_create()
        )

        self.labels = {}

        self.load_model()

    # ======================================
    # VALIDATE IMAGE
    # ======================================

    @staticmethod
    def validate_image(image_path):

        image_path = Path(image_path)

        if not image_path.exists():

            raise FileNotFoundError(
                "Image file does not exist."
            )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise ValueError(
                "Unable to read the image."
            )

        return image

    # ======================================
    # NORMALIZE PERSON NAME
    # ======================================

    @staticmethod
    def normalize_person_name(person_name):

        if not isinstance(person_name, str):

            raise ValueError(
                "Person name must be text."
            )

        person_name = person_name.strip()

        if not person_name:

            raise ValueError(
                "Person name cannot be empty."
            )

        if len(person_name) > 80:

            raise ValueError(
                "Person name is too long."
            )

        if any(
            ord(character) < 32
            for character in person_name
        ):

            raise ValueError(
                "Person name contains invalid characters."
            )

        return person_name

    # ======================================
    # FIND FACES
    # ======================================

    def detect_faces(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40)
        )

        return gray, faces

    # ======================================
    # REGISTER FACE
    # ======================================

    def register_face(
        self,
        person_name,
        image_path
    ):

        person_name = self.normalize_person_name(
            person_name
        )

        image = self.validate_image(
            image_path
        )

        gray, faces = self.detect_faces(
            image
        )

        if len(faces) == 0:

            raise ValueError(
                "No face detected in the supplied image."
            )

        safe_name = re.sub(
            r"[^a-zA-Z0-9_-]",
            "_",
            person_name
        ).strip("_")

        if not safe_name:

            raise ValueError(
                "Person name cannot be converted to a valid folder name."
            )

        person_folder = (
            self.known_faces_folder / safe_name
        )

        person_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        registered_files = []

        existing_files = list(
            person_folder.glob("face_*.jpg")
        )

        next_index = len(existing_files) + 1

        for index, (x, y, width, height) in enumerate(faces):

            face = gray[
                y:y + height,
                x:x + width
            ]

            face = cv2.resize(
                face,
                (200, 200)
            )

            filename = (
                f"face_{next_index + index}.jpg"
            )

            output_path = (
                person_folder / filename
            )

            saved = cv2.imwrite(
                str(output_path),
                face
            )

            if not saved:

                raise RuntimeError(
                    "Unable to save registered face image."
                )

            registered_files.append(
                str(output_path)
            )

        self.database.add_face(
            person_name,
            registered_files[0]
        )

        logger.info(
            "Registered %s face image(s) for %s.",
            len(registered_files),
            person_name
        )

        return {
            "success": True,
            "person_name": person_name,
            "faces_registered": len(registered_files),
            "files": registered_files,
            "message": "Face registered successfully."
        }

    # ======================================
    # LOAD REGISTERED FACE DATA
    # ======================================

    def load_training_data(self):

        face_samples = []
        face_labels = []

        label_mapping = {}

        label_id = 0

        for person_folder in sorted(
            self.known_faces_folder.iterdir()
        ):

            if not person_folder.is_dir():
                continue

            person_name = person_folder.name

            image_files = list(
                person_folder.glob("*.jpg")
            )

            if not image_files:
                continue

            label_mapping[str(label_id)] = person_name

            for image_path in image_files:

                face_image = cv2.imread(
                    str(image_path),
                    cv2.IMREAD_GRAYSCALE
                )

                if face_image is None:
                    continue

                face_image = cv2.resize(
                    face_image,
                    (200, 200)
                )

                face_samples.append(
                    face_image
                )

                face_labels.append(
                    label_id
                )

            label_id += 1

        return (
            face_samples,
            face_labels,
            label_mapping
        )

    # ======================================
    # TRAIN FACE MODEL
    # ======================================

    def train_model(self):

        face_samples, face_labels, label_mapping = (
            self.load_training_data()
        )

        if len(face_samples) < 2:

            raise ValueError(
                "At least two registered face images are required."
            )

        if len(set(face_labels)) < 2:

            raise ValueError(
                "Register at least two different people before training."
            )

        labels_array = np.array(
            face_labels,
            dtype=np.int32
        )

        self.recognizer.train(
            face_samples,
            labels_array
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
                label_mapping,
                file,
                indent=4
            )

        self.labels = label_mapping

        logger.info(
            "Face recognition model trained with %s images.",
            len(face_samples)
        )

        return {
            "success": True,
            "training_images": len(face_samples),
            "registered_people": len(label_mapping),
            "message": "Face recognition model trained successfully."
        }

    # ======================================
    # LOAD TRAINED MODEL
    # ======================================

    def load_model(self):

        if not self.model_path.exists():

            logger.info(
                "No trained face recognition model found."
            )

            return False

        if not self.labels_path.exists():

            logger.warning(
                "Face labels file is missing."
            )

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

            logger.info(
                "Face recognition model loaded."
            )

            return True

        except Exception:

            logger.exception(
                "Unable to load face recognition model."
            )

            self.labels = {}

            return False

    # ======================================
    # RECOGNIZE FACE
    # ======================================

    def recognize_face(self, image_path):

        if not self.model_path.exists():

            raise RuntimeError(
                "Face model has not been trained."
            )

        if not self.labels:

            if not self.load_model():

                raise RuntimeError(
                    "Unable to load the face recognition model."
                )

        image = self.validate_image(
            image_path
        )

        gray, faces = self.detect_faces(
            image
        )

        if len(faces) == 0:

            return {
                "success": True,
                "faces_detected": 0,
                "results": [],
                "message": "No faces detected."
            }

        results = []

        for x, y, width, height in faces:

            face = gray[
                y:y + height,
                x:x + width
            ]

            face = cv2.resize(
                face,
                (200, 200)
            )

            predicted_label, confidence = (
                self.recognizer.predict(face)
            )

            person_name = self.labels.get(
                str(predicted_label),
                "Unknown"
            )

            is_match = (
                confidence <= FACE_MATCH_THRESHOLD
                and person_name != "Unknown"
            )

            if not is_match:
                person_name = "Unknown"

            results.append({
                "person_name": person_name,
                "confidence": round(
                    float(confidence),
                    2
                ),
                "matched": bool(is_match),
                "box": {
                    "x": int(x),
                    "y": int(y),
                    "width": int(width),
                    "height": int(height)
                }
            })

        return {
            "success": True,
            "faces_detected": len(faces),
            "results": results,
            "message": "Face recognition completed."
        }


__all__ = [
    "FaceRecognition"
]
