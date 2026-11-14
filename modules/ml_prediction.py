
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


class MLPrediction:
    def __init__(self):
        dataset = load_iris()

        self.feature_names = list(dataset.feature_names)
        self.target_names = list(dataset.target_names)

        X = dataset.data
        y = dataset.target

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        self.models = {
            "random_forest": RandomForestClassifier(
                n_estimators=150,
                random_state=42
            ),
            "xgboost": XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                random_state=42
            )
        }

        self.accuracies = {}

        for name, model in self.models.items():
            model.fit(X_train, y_train)
            self.accuracies[name] = round(
                model.score(X_test, y_test),
                4
            )

    def predict(self, values, model_name="random_forest"):
        if model_name not in self.models:
            raise ValueError(
                "Choose random_forest or xgboost."
            )

        if not isinstance(values, (list, tuple)) or len(values) != 4:
            raise ValueError(
                "Enter exactly four flower measurements."
            )

        try:
            measurements = [float(value) for value in values]
        except (TypeError, ValueError):
            raise ValueError(
                "All four measurements must be numbers."
            )

        if any(value <= 0 for value in measurements):
            raise ValueError(
                "Measurements must be greater than zero."
            )

        model = self.models[model_name]
        prediction = int(model.predict([measurements])[0])
        probabilities = model.predict_proba([measurements])[0]

        return {
            "prediction": self.target_names[prediction],
            "model": model_name,
            "confidence": round(
                float(max(probabilities)), 4
            ),
            "scores": {
                self.target_names[index]: round(float(score), 4)
                for index, score in enumerate(probabilities)
            },
            "feature_names": self.feature_names,
            "model_accuracy": self.accuracies[model_name],
            "warning": (
                "Educational demonstration using the Iris dataset. "
                "Predictions are not suitable for scientific or "
                "commercial decisions."
            )
        }
