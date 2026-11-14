
import cv2
import json
import numpy as np
import shutil

from pathlib import Path

from config import (
    KNOWN_FACES_FOLDER,
    MODEL_FOLDER
)

from database import Database


class FaceRecognition:

    def __init__(self):

        self.database = Database()

        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades +
            "haarcascade_frontalface_default.xml"
        )

        self.model_path = (
            Path(MODEL_FOLDER) / "face_model.yml"
        )

        self.labels_path = (
            Path(MODEL_FOLDER) / "face_labels.json"
        )

        self.recognizer = (
            cv2.face.LBPHFaceRecognizer_create()
        )

        self.image_size = (200, 200)
        self.max_distance = 70

    def detect_faces(self, image_path):

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError("Could not read image.")

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        return gray, faces

    def register_face(self, image_path, person_name):

        person_name = person_name.strip()

        if not person_name:
            raise ValueError("Person name is required.")

        gray, faces = self.detect_faces(image_path)

        if len(faces) == 0:
            raise ValueError("No face detected.")

        # Select the largest detected face
        x, y, w, h = max(
            faces,
            key=lambda face: face[2] * face[3]
        )

        face = gray[y:y+h, x:x+w]

        face = cv2.resize(
            face,
            self.image_size
        )

        # Create a folder for this person
        person_folder = (
            Path(KNOWN_FACES_FOLDER) / person_name
        )

        person_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        image_count = len(
            list(person_folder.glob("*.jpg"))
        )

        saved_path = (
            person_folder /
            f"face_{image_count + 1}.jpg"
        )

        cv2.imwrite(str(saved_path), face)

        self.database.add_face(
            person_name,
            str(saved_path)
        )

        return str(saved_path)

    def train_model(self):

        records = self.database.get_faces()

        if not records:
            raise ValueError(
                "Register faces before training."
            )

        person_names = sorted({
            record["person_name"]
            for record in records
        })

        name_to_label = {
            name: index
            for index, name in enumerate(person_names)
        }

        face_samples = []
        labels = []

        for record in records:

            image_path = record["image_path"]

            image = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                continue

            image = cv2.resize(
                image,
                self.image_size
            )

            face_samples.append(image)

            labels.append(
                name_to_label[record["person_name"]]
            )

        if not face_samples:
            raise ValueError(
                "No valid registered face images found."
            )

        self.recognizer.train(
            face_samples,
            np.array(labels, dtype=np.int32)
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
                name_to_label,
                file,
                indent=4
            )

        return len(face_samples)

    def recognize_face(self, image_path):

        if not self.model_path.exists():
            raise ValueError(
                "Train the face model first."
            )

        if not self.labels_path.exists():
            raise ValueError(
                "Face label information is missing."
            )

        self.recognizer.read(
            str(self.model_path)
        )

        with open(
            self.labels_path,
            "r",
            encoding="utf-8"
        ) as file:

            name_to_label = json.load(file)

        label_to_name = {
            int(label): name
            for name, label in name_to_label.items()
        }

        gray, faces = self.detect_faces(image_path)

        results = []

        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(
                face,
                self.image_size
            )

            label, distance = self.recognizer.predict(
                face
            )

            if distance <= self.max_distance:
                name = label_to_name.get(
                    label,
                    "Unknown"
                )
            else:
                name = "Unknown"

            results.append({
                "name": name,
                "distance": round(float(distance), 2),
                "box": [int(x), int(y), int(w), int(h)]
            })

        return results
