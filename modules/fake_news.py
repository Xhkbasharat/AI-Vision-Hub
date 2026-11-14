
import re
from pathlib import Path

import joblib

from config import MODEL_FOLDER


class FakeNewsDetector:

    def __init__(self):

        self.model_path = (
            Path(MODEL_FOLDER) / "fake_news_model.joblib"
        )

        self.model = None

    # ======================================
    # CLEAN TEXT
    # ======================================

    @staticmethod
    def clean_text(text):

        if not isinstance(text, str):
            raise ValueError(
                "News text must be a string."
            )

        text = text.lower()

        text = re.sub(
            r"http\S+|www\S+",
            " ",
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
    # VALIDATE INPUT
    # ======================================

    @staticmethod
    def validate_text(text):

        if not isinstance(text, str):
            raise ValueError(
                "Input must be text."
            )

        text = text.strip()

        if not text:
            raise ValueError(
                "Enter a news article or headline."
            )

        if len(text) > 10000:
            raise ValueError(
                "Text must be 10000 characters or fewer."
            )

        return text

    # ======================================
    # LOAD TRAINED MODEL
    # ======================================

    def load_model(self):

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Fake news model is not trained. "
                "Run python train_fake_news.py first."
            )

        try:
            self.model = joblib.load(
                self.model_path
            )

        except Exception as error:
            raise RuntimeError(
                "Could not load fake news model."
            ) from error

        if not hasattr(self.model, "predict"):
            raise RuntimeError(
                "The saved file is not a valid classifier."
            )

        if not hasattr(self.model, "predict_proba"):
            raise RuntimeError(
                "The classifier must support predict_proba."
            )

        return self.model

    # ======================================
    # PREDICT NEWS LABEL
    # ======================================

    def predict(self, text):

        text = self.validate_text(text)

        cleaned_text = self.clean_text(text)

        if not cleaned_text:
            raise ValueError(
                "No usable text remains after cleaning."
            )

        if self.model is None:
            self.load_model()

        prediction = self.model.predict(
            [cleaned_text]
        )[0]

        probabilities = self.model.predict_proba(
            [cleaned_text]
        )[0]

        scores = {
            str(label): round(float(score), 4)
            for label, score in zip(
                self.model.classes_,
                probabilities
            )
        }

        return {
            "prediction": str(prediction),
            "scores": scores,
            "warning": (
                "This model identifies patterns learned "
                "from its training dataset. It does not "
                "independently verify news claims."
            )
        }
