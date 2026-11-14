
import logging

from pathlib import Path

import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions
)

from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array
)


logger = logging.getLogger("AI-Vision-Hub")


class ImageClassification:
    """
    Image classification using pretrained MobileNetV2.

    Uses ImageNet classes to identify objects and
    visual categories in uploaded images.
    """

    def __init__(self):

        self.model_name = "MobileNetV2"

        self.input_size = (224, 224)

        self.dataset_name = "ImageNet"

        self.model = None

        self._load_model()

    # ======================================
    # LOAD MODEL
    # ======================================

    def _load_model(self):

        try:

            logger.info(
                "Loading MobileNetV2 model."
            )

            self.model = MobileNetV2(
                weights="imagenet",
                include_top=True,
                input_shape=(224, 224, 3)
            )

            logger.info(
                "MobileNetV2 loaded successfully."
            )

        except Exception as error:

            logger.exception(
                "Unable to load MobileNetV2."
            )

            raise RuntimeError(
                "Image classification model could not be loaded."
            ) from error

    # ======================================
    # VALIDATE IMAGE
    # ======================================

    @staticmethod
    def validate_image(image_path):

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                "Image file does not exist."
            )

        if not image_path.is_file():

            raise ValueError(
                "Image path must point to a file."
            )

        try:

            image = load_img(
                image_path,
                target_size=(224, 224)
            )

            return image

        except Exception as error:

            raise ValueError(
                "Unable to read or process the supplied image."
            ) from error

    # ======================================
    # PREPROCESS IMAGE
    # ======================================

    def preprocess_image(self, image_path):

        image = self.validate_image(
            image_path
        )

        image_array = img_to_array(
            image
        )

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

        if not isinstance(top_k, int):

            raise ValueError(
                "top_k must be an integer."
            )

        if not 1 <= top_k <= 1000:

            raise ValueError(
                "top_k must be between 1 and 1000."
            )

        try:

            processed_image = self.preprocess_image(
                image_path
            )

            predictions = self.model.predict(
                processed_image,
                verbose=0
            )

            decoded_predictions = decode_predictions(
                predictions,
                top=top_k
            )[0]

            results = []

            for class_id, class_name, confidence in decoded_predictions:

                results.append({
                    "class_id": str(class_id),
                    "class_name": str(class_name),
                    "confidence": round(
                        float(confidence),
                        4
                    ),
                    "percentage": round(
                        float(confidence) * 100,
                        2
                    )
                })

            if not results:

                raise RuntimeError(
                    "The model did not return any predictions."
                )

            best_prediction = results[0]

            logger.info(
                "Image classification completed: %s",
                best_prediction["class_name"]
            )

            return {
                "success": True,
                "model": self.model_name,
                "dataset": self.dataset_name,
                "predictions": results,
                "best_prediction": best_prediction,
                "warning": (
                    "Predictions are model estimates, "
                    "not guaranteed identifications."
                )
            }

        except (ValueError, FileNotFoundError):

            raise

        except Exception as error:

            logger.exception(
                "Image classification failed."
            )

            raise RuntimeError(
                "Unable to classify the supplied image."
            ) from error

    # ======================================
    # GET BEST PREDICTION
    # ======================================

    def get_best_prediction(self, image_path):

        result = self.classify_image(
            image_path,
            top_k=1
        )

        return result["best_prediction"]


__all__ = [
    "ImageClassification"
]
