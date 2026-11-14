
import logging

from transformers import pipeline

from config import SENTIMENT_MODEL_NAME


logger = logging.getLogger("AI-Vision-Hub")


class SentimentAnalyzer:
    """
    Sentiment analysis using a pretrained
    Hugging Face Transformer model.
    """

    def __init__(self):

        self.model_name = SENTIMENT_MODEL_NAME

        self.analyzer = None

        self._load_model()

    # ======================================
    # LOAD MODEL
    # ======================================

    def _load_model(self):

        try:

            logger.info(
                "Loading sentiment model: %s",
                self.model_name
            )

            self.analyzer = pipeline(
                task="sentiment-analysis",
                model=self.model_name
            )

            logger.info(
                "Sentiment model loaded successfully."
            )

        except Exception as error:

            logger.exception(
                "Unable to load sentiment model."
            )

            raise RuntimeError(
                "Sentiment analysis model could not be loaded."
            ) from error

    # ======================================
    # VALIDATE TEXT
    # ======================================

    @staticmethod
    def validate_text(text):

        if not isinstance(text, str):

            raise ValueError(
                "Input must be a text string."
            )

        text = text.strip()

        if not text:

            raise ValueError(
                "Text cannot be empty."
            )

        if len(text) > 5000:

            raise ValueError(
                "Text cannot exceed 5000 characters."
            )

        return text

    # ======================================
    # ANALYZE SENTIMENT
    # ======================================

    def analyze_sentiment(self, text):

        text = self.validate_text(text)

        try:

            result = self.analyzer(
                text,
                truncation=True,
                max_length=512
            )

            if not result:

                raise RuntimeError(
                    "The sentiment model returned no results."
                )

            prediction = result[0]

            raw_label = str(
                prediction["label"]
            ).upper()

            confidence = float(
                prediction["score"]
            )

            # Normalize common binary sentiment labels.
            if raw_label in [
                "POSITIVE",
                "LABEL_1"
            ]:

                sentiment = "Positive"

            elif raw_label in [
                "NEGATIVE",
                "LABEL_0"
            ]:

                sentiment = "Negative"

            else:

                sentiment = raw_label.title()

            confidence = max(
                0.0,
                min(confidence, 1.0)
            )

            logger.info(
                "Sentiment analysis completed: %s",
                sentiment
            )

            return {
                "success": True,
                "sentiment": sentiment,
                "label": raw_label,
                "confidence": round(
                    confidence,
                    4
                ),
                "confidence_percentage": round(
                    confidence * 100,
                    2
                ),
                "model": self.model_name,
                "warning": (
                    "Sentiment predictions are estimates. "
                    "Sarcasm, context, and mixed emotions may "
                    "not be interpreted correctly."
                )
            }

        except Exception as error:

            logger.exception(
                "Sentiment analysis failed."
            )

            raise RuntimeError(
                "Unable to analyze the supplied text."
            ) from error

    # ======================================
    # ANALYZE MULTIPLE TEXTS
    # ======================================

    def analyze_multiple(self, texts):

        if not isinstance(texts, list):

            raise ValueError(
                "Input must be a list of text strings."
            )

        if not texts:

            raise ValueError(
                "At least one text entry is required."
            )

        if len(texts) > 100:

            raise ValueError(
                "A maximum of 100 text entries is allowed."
            )

        results = []

        for text in texts:

            results.append(
                self.analyze_sentiment(text)
            )

        return {
            "success": True,
            "total": len(results),
            "results": results
        }


__all__ = [
    "SentimentAnalyzer"
]
