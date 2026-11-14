
from pathlib import Path

import cv2
from ultralytics import YOLO

from config import (
    MODEL_FOLDER,
    YOLO_MODEL_NAME,
    OBJECT_CONFIDENCE_THRESHOLD
)


class ObjectDetection:

    def __init__(self):
        self.model_folder = Path(MODEL_FOLDER)

        self.model_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        self.model_path = (
            self.model_folder / YOLO_MODEL_NAME
        )

        # Use a local model if it exists.
        # Otherwise, Ultralytics downloads the pretrained model.
        if self.model_path.exists():
            self.model = YOLO(str(self.model_path))
        else:
            self.model = YOLO(YOLO_MODEL_NAME)

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

        image = cv2.imread(str(image_path))

        if image is None:
            raise ValueError(
                "Unable to read the uploaded image."
            )

        return image

    # ======================================
    # DETECT OBJECTS
    # ======================================

    def detect_objects(
        self,
        image_path,
        confidence_threshold=None
    ):

        image = self.validate_image(image_path)

        if confidence_threshold is None:
            confidence_threshold = OBJECT_CONFIDENCE_THRESHOLD

        if not 0 < confidence_threshold <= 1:
            raise ValueError(
                "Confidence threshold must be greater than 0 and at most 1."
            )

        predictions = self.model.predict(
            source=image,
            conf=confidence_threshold,
            verbose=False
        )

        if not predictions:
            return []

        result = predictions[0]

        detected_objects = []

        if result.boxes is None:
            return detected_objects

        for box in result.boxes:

            coordinates = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .tolist()
            )

            confidence = float(
                box.conf[0].item()
            )

            class_id = int(
                box.cls[0].item()
            )

            class_name = self.model.names.get(
                class_id,
                str(class_id)
            )

            detected_objects.append({
                "object": str(class_name),
                "confidence": round(confidence, 4),
                "confidence_percentage": round(
                    confidence * 100,
                    2
                ),
                "box": {
                    "x1": round(float(coordinates[0]), 2),
                    "y1": round(float(coordinates[1]), 2),
                    "x2": round(float(coordinates[2]), 2),
                    "y2": round(float(coordinates[3]), 2)
                }
            })

        return detected_objects

    # ======================================
    # DETECT AND SAVE ANNOTATED IMAGE
    # ======================================

    def detect_and_save(
        self,
        image_path,
        output_path,
        confidence_threshold=None
    ):

        image = self.validate_image(image_path)

        if confidence_threshold is None:
            confidence_threshold = OBJECT_CONFIDENCE_THRESHOLD

        predictions = self.model.predict(
            source=image,
            conf=confidence_threshold,
            verbose=False
        )

        if not predictions:
            raise RuntimeError(
                "YOLO did not return a prediction."
            )

        annotated_image = predictions[0].plot()

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        success = cv2.imwrite(
            str(output_path),
            annotated_image
        )

        if not success:
            raise RuntimeError(
                "Could not save annotated image."
            )

        return {
            "output_path": str(output_path),
            "detections": self.detect_objects(
                image_path,
                confidence_threshold
            )
        }
