
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier


class MLPrediction:

    def __init__(self):

        self.random_forest = None
        self.xgboost_model = None

        self.feature_names = []

        self.accuracy = {}

        self.train_models()

    def train_models(self):

        # Load a sample classification dataset
        dataset = load_breast_cancer()

        X = dataset.data
        y = dataset.target

        self.feature_names = list(
            dataset.feature_names
        )

        # Split training and testing data
        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
        )

        # Random Forest model
        self.random_forest = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        self.random_forest.fit(
            X_train,
            y_train
        )

        rf_predictions = (
            self.random_forest.predict(X_test)
        )

        self.accuracy["random_forest"] = round(
            accuracy_score(
                y_test,
                rf_predictions
            ),
            4
        )

        # XGBoost model
        self.xgboost_model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=42,
            n_jobs=1
        )

        self.xgboost_model.fit(
            X_train,
            y_train
        )

        xgb_predictions = (
            self.xgboost_model.predict(X_test)
        )

        self.accuracy["xgboost"] = round(
            accuracy_score(
                y_test,
                xgb_predictions
            ),
            4
        )

    def predict(self, values, model_name="random_forest"):

        # Validate input
        if len(values) != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} "
                f"features, received {len(values)}."
            )

        try:
            input_data = np.asarray(
                values,
                dtype=float
            ).reshape(1, -1)
        except (TypeError, ValueError):
            raise ValueError(
                "All input features must be numerical."
            )

        if not np.isfinite(input_data).all():
            raise ValueError(
                "Input contains invalid numerical values."
            )

        # Select model
        if model_name == "random_forest":
            model = self.random_forest

        elif model_name == "xgboost":
            model = self.xgboost_model

        else:
            raise ValueError(
                "Choose random_forest or xgboost."
            )

        # Generate prediction
        prediction = int(
            model.predict(input_data)[0]
        )

        probabilities = model.predict_proba(
            input_data
        )[0]

        return {
            "prediction": prediction,
            "class_name": (
                "malignant" if prediction == 0
                else "benign"
            ),
            "scores": {
                "malignant": round(
                    float(probabilities[0]), 4
                ),
                "benign": round(
                    float(probabilities[1]), 4
                )
            },
            "model": model_name
        }

    def get_model_accuracy(self):
        return self.accuracy.copy()
