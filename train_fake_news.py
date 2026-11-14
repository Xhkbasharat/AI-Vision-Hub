
import re
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import DATA_DIR, MODEL_FOLDER


# ==========================================
# FILE PATHS
# ==========================================

DATASET_PATH = Path(DATA_DIR) / "fake_news_dataset.csv"

MODEL_PATH = (
    Path(MODEL_FOLDER) / "fake_news_model.joblib"
)


# ==========================================
# TEXT CLEANING
# ==========================================

def clean_text(text):

    text = str(text).lower()

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


# ==========================================
# LOAD DATASET
# ==========================================

def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    dataset = pd.read_csv(DATASET_PATH)

    required_columns = {
        "text",
        "label"
    }

    if not required_columns.issubset(dataset.columns):
        raise ValueError(
            "Dataset must contain text and label columns."
        )

    dataset = dataset[
        ["text", "label"]
    ].dropna()

    dataset["text"] = dataset["text"].apply(clean_text)

    dataset["label"] = (
        dataset["label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    dataset = dataset[
        dataset["label"].isin(["REAL", "FAKE"])
    ]

    dataset = dataset[
        dataset["text"].str.len() > 0
    ]

    dataset = dataset.drop_duplicates(
        subset=["text", "label"]
    )

    if dataset.empty:
        raise ValueError(
            "No valid labelled records were found."
        )

    class_counts = dataset["label"].value_counts()

    if not {"REAL", "FAKE"}.issubset(
        set(class_counts.index)
    ):
        raise ValueError(
            "Dataset must contain both REAL and FAKE labels."
        )

    if class_counts.min() < 5:
        raise ValueError(
            "At least five examples per class are required."
        )

    return dataset


# ==========================================
# TRAIN MODEL
# ==========================================

def train_model():

    print("Loading fake-news dataset...")

    dataset = load_dataset()

    X = dataset["text"]
    y = dataset["label"]

    print("\nDataset summary:")
    print(dataset["label"].value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=20000,
                ngram_range=(1, 2),
                stop_words="english"
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train
    )

    # ======================================
    # EVALUATION
    # ======================================

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\nModel evaluation")
    print("================")

    print(
        f"Test accuracy: {accuracy:.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=["FAKE", "REAL"],
            zero_division=0
        )
    )

    print("\nConfusion matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=["FAKE", "REAL"]
        )
    )

    # ======================================
    # SAVE MODEL
    # ======================================

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\nModel saved successfully:")
    print(MODEL_PATH)


# ==========================================
# PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":

    try:
        train_model()

    except Exception as error:
        print("\nTraining failed:")
        print(error)
