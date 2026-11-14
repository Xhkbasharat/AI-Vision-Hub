
import logging
import re
import sys

from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from config import BASE_DIR, MODEL_FOLDER


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("AI-Vision-Hub")


# ======================================
# PATHS
# ======================================

DATASET_PATH = (
    Path(BASE_DIR)
    / "data"
    / "fake_news_dataset.csv"
)

OUTPUT_PATH = (
    Path(MODEL_FOLDER)
    / "fake_news_model.joblib"
)


# ======================================
# CLEAN TEXT
# ======================================

def clean_text(text):

    text = str(text).lower()

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
# LOAD DATASET
# ======================================

def load_dataset():

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    logger.info(
        "Loading dataset: %s",
        DATASET_PATH
    )

    dataframe = pd.read_csv(
        DATASET_PATH
    )

    required_columns = [
        "text",
        "label"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    dataframe = dataframe[
        required_columns
    ].copy()

    dataframe = dataframe.dropna()

    dataframe["text"] = (
        dataframe["text"]
        .astype(str)
        .str.strip()
    )

    dataframe["label"] = (
        dataframe["label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    dataframe["text"] = dataframe[
        "text"
    ].apply(clean_text)

    dataframe = dataframe[
        dataframe["label"].isin([
            "REAL",
            "FAKE"
        ])
    ]

    dataframe = dataframe[
        dataframe["text"] != ""
    ]

    # Remove ambiguous duplicate articles.
    dataframe["text_key"] = (
        dataframe["text"]
    )

    label_counts = dataframe.groupby(
        "text_key"
    )["label"].nunique()

    ambiguous_texts = label_counts[
        label_counts > 1
    ].index

    dataframe = dataframe[
        ~dataframe["text_key"].isin(
            ambiguous_texts
        )
    ]

    dataframe = dataframe.drop_duplicates(
        subset=["text_key"]
    )

    dataframe = dataframe.drop(
        columns=["text_key"]
    )

    class_counts = dataframe["label"].value_counts()

    logger.info(
        "Dataset class distribution:\n%s",
        class_counts
    )

    if len(class_counts) != 2:

        raise ValueError(
            "Dataset must contain both REAL and FAKE labels."
        )

    if class_counts.min() < 10:

        raise ValueError(
            "At least 10 examples per class are required. "
            "More data is recommended for reliable evaluation."
        )

    logger.info(
        "Dataset loaded successfully: %s records.",
        len(dataframe)
    )

    return dataframe


# ======================================
# TRAIN MODEL
# ======================================

def train_model():

    dataframe = load_dataset()

    X = dataframe["text"]

    y = dataframe["label"]

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

    logger.info(
        "Training records: %s",
        len(X_train)
    )

    logger.info(
        "Testing records: %s",
        len(X_test)
    )

    # Build classification pipeline.
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                max_features=20000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True
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

    logger.info(
        "Starting model training."
    )

    model.fit(
        X_train,
        y_train
    )

    logger.info(
        "Model training completed."
    )

    # Evaluate model.
    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n========== MODEL EVALUATION ==========")

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=["FAKE", "REAL"]
        )
    )

    print("======================================\n")

    # Save trained model.
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        OUTPUT_PATH
    )

    logger.info(
        "Trained model saved: %s",
        OUTPUT_PATH
    )

    return {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test),
        "model_path": str(OUTPUT_PATH)
    }


# ======================================
# MAIN
# ======================================

if __name__ == "__main__":

    try:

        result = train_model()

        print("\nTraining completed successfully.")

        print(
            f"Model accuracy: {result['accuracy']:.2%}"
        )

        print(
            f"Model saved at: {result['model_path']}"
        )

    except Exception as error:

        logger.exception(
            "Fake news model training failed."
        )

        print(
            f"\nTraining failed: {error}"
        )

        sys.exit(1)
