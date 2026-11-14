
import tensorflow as tf

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import (
    preprocess_input,
    decode_predictions
)
from tensorflow.keras.preprocessing import image

import numpy as np


class ImageClassification:

    def __init__(self):

        print("Loading MobileNetV2 model...")

        self.model = MobileNetV2(
            weights="imagenet",
            include_top=True
        )

        print("Image classification model loaded!")

    def classify_image(self, image_path, top_k=5):

        # Load and resize image
        img = image.load_img(
            image_path,
            target_size=(224, 224)
        )

        # Convert image to numerical array
        img_array = image.img_to_array(img)

        # Add batch dimension
        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # Preprocess image
        img_array = preprocess_input(img_array)

        # Predict image categories
        predictions = self.model.predict(
            img_array,
            verbose=0
        )

        # Decode predictions
        decoded = decode_predictions(
            predictions,
            top=top_k
        )[0]

        results = []

        for _, label, score in decoded:

            results.append({
                "label": label.replace("_", " "),
                "score": round(float(score), 4),
                "percentage": round(
                    float(score) * 100, 2
                )
            })

        return results

    def get_best_prediction(self, image_path):

        results = self.classify_image(
            image_path,
            top_k=1
        )

        if not results:
            return None

        return results[0]
