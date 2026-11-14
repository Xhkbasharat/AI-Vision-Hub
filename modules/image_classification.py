
from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.utils import load_img, img_to_array


class ImageClassification:

    def __init__(self):
        self.image_size = (224, 224)

        # Load pretrained ImageNet MobileNetV2.
        self.model = MobileNetV2(
            weights="imagenet",
            include_top=True,
            input_shape=(224, 224, 3)
        )

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
                "The supplied path is not a file."
            )

        return image_path

    # ======================================
    # PREPROCESS IMAGE
    # ======================================

    def preprocess_image(self, image_path):

        image_path = self.validate_image(image_path)

        try:
            image = load_img(
                str(image_path),
                target_size=self.image_size,
                color_mode="rgb"
            )

            image_array = img_to_array(image)

        except Exception as error:
            raise ValueError(
                "Unable to process this image."
            ) from error

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        image_array = preprocess_input(
            image_array
        )

        return image_array

    # ======================================
    # CLASSIFY IMAGE
    # ======================================

    def classify_image(
        self,
        image_path,
        top_k=5
    ):

        if not isinstance(top_k, int) or not 1 <= top_k <= 10:
            raise ValueError(
                "top_k must be an integer between 1 and 10."
            )

        processed_image = self.preprocess_image(
            image_path
        )

        predictions = self.model.predict(
            processed_image,
            verbose=0
        )

        decoded = decode_predictions(
            predictions,
            top=top_k
        )[0]

        results = []

        for class_id, class_name, score in decoded:

            results.append({
                "class_id": class_id,
                "label": class_name.replace("_", " "),
                "confidence": round(float(score), 4),
                "confidence_percentage": round(
                    float(score) * 100,
                    2
                )
            })

        return {
            "model": "MobileNetV2",
            "dataset": "ImageNet",
            "predictions": results,
            "best_prediction": results[0],
            "warning": (
                "Predictions are model estimates, "
                "not guaranteed identifications."
            )
        }

    # ======================================
    # GET BEST PREDICTION
    # ======================================

    def get_best_prediction(self, image_path):

        result = self.classify_image(
            image_path,
            top_k=1
        )

        return result["best_prediction"]
