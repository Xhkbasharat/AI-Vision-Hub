
import re
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from config import MODEL_FOLDER


class FakeNewsDetector:

    def __init__(self):

        self.model_path = (
            Path(MODEL_FOLDER) / "fake_news_model.pkl"
        )

        self.model = None
        self.accuracy = None

    def clean_text(self, text):

        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"[^a-z\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def train(self, texts, labels):

        if len(texts) != len(labels):
            raise ValueError(
                "Texts and labels must have equal lengths."
            )

        if len(texts) < 10:
            raise ValueError(
                "Provide at least 10 training examples."
            )

        clean_texts = [
            self.clean_text(text)
            for text in texts
        ]

        if len(set(labels)) != 2:
            raise ValueError(
                "Training requires exactly two classes."
            )

        X_train, X_test, y_train, y_test = (
            train_test_split(
                clean_texts,
                labels,
                test_size=0.2,
                random_state=42,
                stratify=labels
            )
        )

        self.model = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=10000,
                    ngram_range=(1, 2)
                )
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced"
                )
            )
        ])

        self.model.fit(X_train, y_train)

        predictions = self.model.predict(X_test)

        self.accuracy = round(
            accuracy_score(y_test, predictions),
            4
        )

        self.model_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(self.model_path, "wb") as file:
            pickle.dump(self.model, file)

        return self.accuracy

    def load_model(self):

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Train the model before making predictions."
            )

        with open(self.model_path, "rb") as file:
            self.model = pickle.load(file)

    def predict(self, text):

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "Please enter a news article or headline."
            )

        if self.model is None:
            self.load_model()

        cleaned = self.clean_text(text)

        prediction = self.model.predict([cleaned])[0]

        probabilities = self.model.predict_proba(
            [cleaned]
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
                "This is a text-pattern classification result, "
                "not independent verification of the news claim."
            )
        }
