
import logging
import re

from pathlib import Path

import joblib

from config import MODEL_FOLDER


logger = logging.getLogger("AI-Vision-Hub")


class FakeNewsDetector:
    """
    Fake news classification using a trained
    Scikit-learn machine-learning pipeline.
    """

    def __init__(self):

        self.model_path = (
            Path(MODEL_FOLDER)
            / "fake_news_model.joblib"
        )

        self.model = None

        self._load_model()

    # ======================================
    # LOAD TRAINED MODEL
    # ======================================

    def _load_model(self):

        if not self.model_path.exists():

            raise FileNotFoundError(
                "Fake news model not found. "
                "Run train_fake_news.py first."
            )

        try:

            logger.info(
                "Loading fake news model."
            )

            self.model = joblib.load(
                self.model_path
            )

            if not hasattr(self.model, "predict"):

                raise ValueError(
                    "Loaded model does not support prediction."
                )

            if not hasattr(self.model, "predict_proba"):

                raise ValueError(
                    "Loaded model does not support probability predictions."
                )

            logger.info(
                "Fake news model loaded successfully."
            )

        except Exception as error:

            logger.exception(
                "Unable to load fake news model."
            )

            raise RuntimeError(
                "Fake news detection model could not be loaded."
            ) from error

    # ======================================
    # VALIDATE INPUT
    # ======================================

    @staticmethod
    def validate_text(text):

        if not isinstance(text, str):

            raise ValueError(
                "News content must be text."
            )

        text = text.strip()

        if not text:

            raise ValueError(
                "News content cannot be empty."
            )

        if len(text) > 10000:

            raise ValueError(
                "News content cannot exceed 10000 characters."
            )

        return text

    # ======================================
    # CLEAN TEXT
    # ======================================

    @staticmethod
    def clean_text(text):

        text = text.lower()

        text = re.sub(
            r"https?://\S+|www\.\S+",
            "",
            text
        )

        text = re.sub(
            r"[^a-z\s]",
            " ",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    # ======================================
    # PREDICT NEWS CATEGORY
    # ======================================

    def predict(self, text):

        text = self.validate_text(text)

        cleaned_text = self.clean_text(
            text
        )

        if not cleaned_text:

            raise ValueError(
                "News content contains no usable text."
            )

        try:

            prediction = self.model.predict(
                [cleaned_text]
            )[0]

            probabilities = self.model.predict_proba(
                [cleaned_text]
            )[0]

            classes = list(
                self.model.classes_
            )

            scores = {}

            for index, label in enumerate(classes):

                scores[str(label)] = round(
                    float(probabilities[index]),
                    4
                )

            confidence = float(
                max(probabilities)
            )

            label = str(
                prediction
            ).upper()

            if label == "FAKE":

                verdict = "Fake"

            elif label == "REAL":

                verdict = "Real"

            else:

                verdict = label.title()

            logger.info(
                "Fake news prediction completed: %s",
                verdict
            )

            return {
                "success": True,
                "prediction": verdict,
                "label": label,
                "confidence": round(
                    confidence,
                    4
                ),
                "confidence_percentage": round(
                    confidence * 100,
                    2
                ),
                "scores": scores,
                "model": "TF-IDF + Logistic Regression",
                "warning": (
                    "This is an automated text classification "
                    "estimate, not independent fact-checking. "
                    "Verify claims using reliable sources."
                )
            }

        except Exception as error:

            logger.exception(
                "Fake news prediction failed."
            )

            raise RuntimeError(
                "Unable to analyze the supplied news content."
            ) from error


__all__ = [
    "FakeNewsDetector"
]
