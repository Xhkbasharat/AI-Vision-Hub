
from transformers import pipeline

from config import SENTIMENT_MODEL_NAME


class SentimentAnalysis:

    def __init__(self):

        print("Loading sentiment analysis model...")

        self.classifier = pipeline(
            task="sentiment-analysis",
            model=SENTIMENT_MODEL_NAME
        )

        print("Sentiment model loaded successfully!")

    def analyze_sentiment(self, text):

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "Please enter some text to analyze."
            )

        # Limit input length for practical processing
        text = text.strip()[:2000]

        result = self.classifier(
            text,
            truncation=True,
            max_length=512
        )[0]

        label = result["label"].upper()

        sentiment = (
            "Positive" if label == "POSITIVE"
            else "Negative"
        )

        return {
            "text": text,
            "sentiment": sentiment,
            "label": label,
            "confidence": round(
                float(result["score"]), 4
            ),
            "confidence_percentage": round(
                float(result["score"]) * 100, 2
            )
        }

    def analyze_multiple(self, texts):

        if not isinstance(texts, list):
            raise ValueError(
                "Input must be a list of text entries."
            )

        if not texts:
            return []

        clean_texts = [
            text.strip()[:2000]
            for text in texts
            if isinstance(text, str) and text.strip()
        ]

        if not clean_texts:
            raise ValueError(
                "No valid text entries provided."
            )

        predictions = self.classifier(
            clean_texts,
            truncation=True,
            max_length=512
        )

        results = []

        for text, prediction in zip(
            clean_texts, predictions
        ):

            label = prediction["label"].upper()

            results.append({
                "text": text,
                "sentiment": (
                    "Positive" if label == "POSITIVE"
                    else "Negative"
                ),
                "confidence": round(
                    float(prediction["score"]), 4
                )
            })

        return results
