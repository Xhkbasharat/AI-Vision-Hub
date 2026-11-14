
import logging

import numpy as np

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier


logger = logging.getLogger("AI-Vision-Hub")


class MLPrediction:
    """
    Machine-learning prediction module.

    Supports:
    - Random Forest
    - XGBoost

    Demonstrates multiclass classification using
    the Iris flower dataset.
    """

    def __init__(self):

        self.models = {}

        self.accuracies = {}

        self.feature_names = []

        self.target_names = []

        self._load_dataset()

        self._train_models()

    # ======================================
    # LOAD DATASET
    # ======================================

    def _load_dataset(self):

        dataset = load_iris()

        self.X = dataset.data

        self.y = dataset.target

        self.feature_names = list(
            dataset.feature_names
        )

        self.target_names = list(
            dataset.target_names
        )

        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(
                self.X,
                self.y,
                test_size=0.2,
                random_state=42,
                stratify=self.y
            )
        )

        logger.info(
            "Iris dataset loaded successfully."
        )

    # ======================================
    # TRAIN MODELS
    # ======================================

    def _train_models(self):

        try:

            # Random Forest.
            random_forest = RandomForestClassifier(
                n_estimators=150,
                max_depth=5,
                random_state=42
            )

            random_forest.fit(
                self.X_train,
                self.y_train
            )

            rf_predictions = random_forest.predict(
                self.X_test
            )

            rf_accuracy = accuracy_score(
                self.y_test,
                rf_predictions
            )

            self.models["random_forest"] = (
                random_forest
            )

            self.accuracies["random_forest"] = (
                float(rf_accuracy)
            )

            # XGBoost.
            xgboost = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=1
            )

            xgboost.fit(
                self.X_train,
                self.y_train
            )

            xgb_predictions = xgboost.predict(
                self.X_test
            )

            xgb_accuracy = accuracy_score(
                self.y_test,
                xgb_predictions
            )

            self.models["xgboost"] = xgboost

            self.accuracies["xgboost"] = (
                float(xgb_accuracy)
            )

            logger.info(
                "Machine-learning models trained successfully."
            )

        except Exception as error:

            logger.exception(
                "Machine-learning model training failed."
            )

            raise RuntimeError(
                "Unable to train machine-learning models."
            ) from error

    # ======================================
    # VALIDATE INPUT
    # ======================================

    def validate_input(self, values):

        if not isinstance(
            values,
            (list, tuple, np.ndarray)
        ):

            raise ValueError(
                "Input values must be a list of numbers."
            )

        if len(values) != 4:

            raise ValueError(
                "Exactly four feature values are required."
            )

        try:

            values = np.asarray(
                values,
                dtype=float
            )

        except (TypeError, ValueError) as error:

            raise ValueError(
                "All feature values must be numeric."
            ) from error

        if not np.all(np.isfinite(values)):

            raise ValueError(
                "Feature values must be finite numbers."
            )

        if np.any(values <= 0):

            raise ValueError(
                "Feature values must be greater than zero."
            )

        return values.reshape(1, -1)

    # ======================================
    # MAKE PREDICTION
    # ======================================

    def predict(
        self,
        values,
        model_name="random_forest"
    ):

        if model_name not in self.models:

            raise ValueError(
                "Unsupported model. Choose random_forest or xgboost."
            )

        input_data = self.validate_input(
            values
        )

        model = self.models[model_name]

        try:

            prediction = int(
                model.predict(input_data)[0]
            )

            probabilities = model.predict_proba(
                input_data
            )[0]

            confidence = float(
                np.max(probabilities)
            )

            scores = {}

            for index, class_name in enumerate(
                self.target_names
            ):

                scores[class_name] = round(
                    float(probabilities[index]),
                    4
                )

            predicted_class = (
                self.target_names[prediction]
            )

            logger.info(
                "Prediction completed using %s.",
                model_name
            )

            return {
                "success": True,
                "prediction": predicted_class,
                "model": model_name,
                "confidence": round(
                    confidence,
                    4
                ),
                "scores": scores,
                "feature_names": self.feature_names,
                "model_accuracy": round(
                    self.accuracies[model_name],
                    4
                ),
                "warning": (
                    "Demonstration prediction using the Iris dataset. "
                    "Results are not suitable for high-stakes decisions."
                )
            }

        except Exception as error:

            logger.exception(
                "Machine-learning prediction failed."
            )

            raise RuntimeError(
                "Unable to generate the prediction."
            ) from error

    # ======================================
    # GET MODEL INFORMATION
    # ======================================

    def get_model_info(self):

        return {
            "available_models": list(
                self.models.keys()
            ),
            "accuracies": self.accuracies,
            "features": self.feature_names,
            "classes": self.target_names,
            "dataset": "Iris"
        }


__all__ = [
    "MLPrediction"
]
