
from transformers import pipeline

from config import SENTIMENT_MODEL_NAME


class SentimentAnalysis:

    def __init__(self):
        self.model_name = SENTIMENT_MODEL_NAME

        # Load the pretrained sentiment model.
        self.classifier = pipeline(
            task="sentiment-analysis",
            model=self.model_name
        )

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
                "Text cannot be empty."
            )

        if len(text) > 5000:
            raise ValueError(
                "Text must be 5000 characters or fewer."
            )

        return text

    # ======================================
    # ANALYZE SENTIMENT
    # ======================================

    def analyze_sentiment(self, text):

        text = self.validate_text(text)

        prediction = self.classifier(
            text,
            truncation=True,
            max_length=512
        )[0]

        label = prediction["label"].upper()
        score = float(prediction["score"])

        if label in ("POSITIVE", "LABEL_1"):
            sentiment = "positive"

        elif label in ("NEGATIVE", "LABEL_0"):
            sentiment = "negative"

        else:
            sentiment = "unknown"

        return {
            "sentiment": sentiment,
            "label": label,
            "confidence": round(score, 4),
            "confidence_percentage": round(
                score * 100,
                2
            ),
            "model": self.model_name,
            "warning": (
                "This model is designed for English text. "
                "It may not reliably interpret sarcasm, "
                "context, or other languages."
            )
        }

    # ======================================
    # ANALYZE MULTIPLE TEXTS
    # ======================================

    def analyze_multiple(self, texts):

        if not isinstance(texts, list):
            raise ValueError(
                "Input must be a list of texts."
            )

        if not texts:
            raise ValueError(
                "Provide at least one text."
            )

        if len(texts) > 50:
            raise ValueError(
                "A maximum of 50 texts is allowed."
            )

        results = []

        for text in texts:
            results.append(
                self.analyze_sentiment(text)
            )

        return results
