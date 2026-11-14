
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

import joblib

from config import MODEL_FOLDER


# Dataset location
DATASET_PATH = (
    Path("data") / "fake_news_dataset.csv"
)

# Model output location
MODEL_PATH = (
    Path(MODEL_FOLDER) / "fake_news_model.joblib"
)


def train_fake_news_model():

    print("Loading fake news dataset...")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found: " + str(DATASET_PATH)
        )

    data = pd.read_csv(DATASET_PATH)

    required_columns = {"text", "label"}

    if not required_columns.issubset(data.columns):
        raise ValueError(
            "Dataset must contain text and label columns."
        )

    data = data.dropna(
        subset=["text", "label"]
    )

    data["text"] = data["text"].astype(str).str.strip()

    data["label"] = (
        data["label"].astype(str).str.upper().str.strip()
    )

    data = data[
        data["label"].isin(["REAL", "FAKE"])
    ]

    if data["label"].nunique() != 2:
        raise ValueError(
            "Dataset must contain both REAL and FAKE labels."
        )

    if data["label"].value_counts().min() < 2:
        raise ValueError(
            "Each class needs at least two examples."
        )

    X = data["text"]
    y = data["label"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

    print("Training TF-IDF and Logistic Regression...")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                max_features=20000,
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

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\nModel evaluation")
    print("----------------")

    print("Accuracy:", round(accuracy, 4))

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, MODEL_PATH)

    print("Model saved:", MODEL_PATH)


if __name__ == "__main__":
    train_fake_news_model()
