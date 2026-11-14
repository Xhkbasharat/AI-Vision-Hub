
import logging

from pathlib import Path

import cv2

from ultralytics import YOLO

from config import (
    MODEL_FOLDER,
    OBJECT_CONFIDENCE_THRESHOLD
)


logger = logging.getLogger("AI-Vision-Hub")


class ObjectDetection:
    """
    Object detection using the YOLO model.

    Supports image validation, object detection,
    confidence filtering, and annotated image output.
    """

    def __init__(self, model_name="yolov8n.pt"):

        self.model_folder = Path(MODEL_FOLDER)

        self.model_folder.mkdir(
            parents=True,
            exist_ok=True
        )

        self.model_name = model_name

        self.model_path = (
            self.model_folder / model_name
        )

        self.confidence_threshold = (
            OBJECT_CONFIDENCE_THRESHOLD
        )

        try:

            if self.model_path.exists():

                self.model = YOLO(
                    str(self.model_path)
                )

            else:

                logger.info(
                    "Loading YOLO model: %s",
                    model_name
                )

                # Ultralytics downloads the model
                # when it is not already available.
                self.model = YOLO(
                    model_name
                )

            logger.info(
                "YOLO object detection initialized."
            )

        except Exception as error:

            logger.exception(
                "Unable to initialize YOLO model."
            )

            raise RuntimeError(
                "Object detection model could not be loaded."
            ) from error

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

        if not image_path.is_file():

            raise ValueError(
                "Image path must point to a file."
            )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise ValueError(
                "Unable to read the supplied image."
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

        image = self.validate_image(
            image_path
        )

        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else self.confidence_threshold
        )

        if not isinstance(threshold, (int, float)):

            raise ValueError(
                "Confidence threshold must be numeric."
            )

        if not 0 <= threshold <= 1:

            raise ValueError(
                "Confidence threshold must be between 0 and 1."
            )

        try:

            predictions = self.model.predict(
                source=image,
                conf=float(threshold),
                verbose=False
            )

            if not predictions:

                return []

            result = predictions[0]

            detections = []

            if result.boxes is None:
                return detections

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

                object_name = result.names.get(
                    class_id,
                    str(class_id)
                )

                x1, y1, x2, y2 = [
                    int(value)
                    for value in coordinates
                ]

                detections.append({
                    "object": str(object_name),
                    "confidence": round(
                        confidence,
                        4
                    ),
                    "percentage": round(
                        confidence * 100,
                        2
                    ),
                    "box": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                })

            logger.info(
                "Detected %s objects.",
                len(detections)
            )

            return detections

        except Exception as error:

            logger.exception(
                "Object detection failed."
            )

            raise RuntimeError(
                "Unable to detect objects in the image."
            ) from error

    # ======================================
    # DETECT AND SAVE ANNOTATED IMAGE
    # ======================================

    def detect_and_save(
        self,
        image_path,
        output_path,
        confidence_threshold=None
    ):

        image = self.validate_image(
            image_path
        )

        detections = self.detect_objects(
            image_path,
            confidence_threshold
        )

        annotated_image = image.copy()

        for detection in detections:

            box = detection["box"]

            x1 = box["x1"]
            y1 = box["y1"]
            x2 = box["x2"]
            y2 = box["y2"]

            label = (
                f'{detection["object"]} '
                f'{detection["percentage"]}%'
            )

            cv2.rectangle(
                annotated_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2
            )

            cv2.putText(
                annotated_image,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        saved = cv2.imwrite(
            str(output_path),
            annotated_image
        )

        if not saved:

            raise RuntimeError(
                "Unable to save annotated image."
            )

        logger.info(
            "Annotated image saved: %s",
            output_path
        )

        return {
            "success": True,
            "output_path": str(output_path),
            "detections": detections,
            "total_objects": len(detections)
        }

    # ======================================
    # UPDATE CONFIDENCE THRESHOLD
    # ======================================

    def set_confidence_threshold(self, threshold):

        if not isinstance(threshold, (int, float)):

            raise ValueError(
                "Threshold must be numeric."
            )

        if not 0 <= threshold <= 1:

            raise ValueError(
                "Threshold must be between 0 and 1."
            )

        self.confidence_threshold = float(
            threshold
        )

        return self.confidence_threshold


__all__ = [
    "ObjectDetection"
]
