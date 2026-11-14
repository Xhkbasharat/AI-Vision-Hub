
from pathlib import Path

from ultralytics import YOLO

from config import (
    YOLO_MODEL_NAME,
    OBJECT_CONFIDENCE_THRESHOLD,
    MODEL_FOLDER
)


class ObjectDetection:

    def __init__(self):

        model_path = Path(MODEL_FOLDER) / YOLO_MODEL_NAME

        # Load local weights if available.
        # Otherwise, Ultralytics downloads the model.
        if model_path.exists():
            self.model = YOLO(str(model_path))
        else:
            self.model = YOLO(YOLO_MODEL_NAME)

    def detect_objects(self, image_path):

        image_path = Path(image_path)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        results = self.model.predict(
            source=str(image_path),
            conf=OBJECT_CONFIDENCE_THRESHOLD,
            verbose=False
        )

        detections = []

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0].item())

                confidence = float(
                    box.conf[0].item()
                )

                coordinates = (
                    box.xyxy[0].cpu().tolist()
                )

                object_name = self.model.names[
                    class_id
                ]

                detections.append({
                    "object": object_name,
                    "confidence": round(
                        confidence, 4
                    ),
                    "confidence_percentage": round(
                        confidence * 100, 2
                    ),
                    "box": [
                        round(value, 2)
                        for value in coordinates
                    ]
                })

        return detections

    def detect_and_save(self, image_path, output_path):

        image_path = Path(image_path)
        output_path = Path(output_path)

        detections = self.detect_objects(
            image_path
        )

        results = self.model.predict(
            source=str(image_path),
            conf=OBJECT_CONFIDENCE_THRESHOLD,
            verbose=False
        )

        if not results:
            raise RuntimeError(
                "YOLO did not return a result."
            )

        annotated_image = results[0].plot()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        from PIL import Image

        Image.fromarray(
            annotated_image[:, :, ::-1]
        ).save(output_path)

        return {
            "detections": detections,
            "output_image": str(output_path)
        }
