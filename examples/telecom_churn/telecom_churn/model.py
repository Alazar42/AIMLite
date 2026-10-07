"""Customer Churn Prediction (Scratch Model Paradigm): model.py

Classifier model for customer churn prediction.
Supports scikit-learn (RandomForestClassifier) with a genuine pure-Python LogisticRegression fallback.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from aimlite import Model


DEFAULT_FEATURE_COLUMNS: List[str] = [
    "AccountWeeks",
    "ContractRenewal",
    "DataPlan",
    "DataUsage",
    "CustServCalls",
    "DayMins",
    "DayCalls",
    "MonthlyCharge",
    "OverageFee",
    "RoamMins",
]


class PureLogisticClassifier:
    """Zero-dependency pure-Python Logistic Regression classifier.

    Standardizes inputs via z-score normalization and optimizes weights
    via batch gradient descent with sigmoid activation.
    Provides mathematically genuine probabilities and predictions.
    """

    def __init__(self, lr: float = 0.5, epochs: int = 300) -> None:
        self.lr = lr
        self.epochs = epochs
        self.weights: List[float] = []
        self.bias: float = 0.0
        self.mean: List[float] = []
        self.std: List[float] = []

    def fit(self, X: List[List[float]], y: List[int]) -> PureLogisticClassifier:
        n_samples = len(X)
        if n_samples == 0:
            return self
        n_features = len(X[0])

        # Compute feature means and standard deviations
        self.mean = [sum(X[i][j] for i in range(n_samples)) / n_samples for j in range(n_features)]
        self.std = [
            math.sqrt(sum((X[i][j] - self.mean[j]) ** 2 for i in range(n_samples)) / n_samples) or 1.0
            for j in range(n_features)
        ]

        # Standardize training matrix
        X_scaled = [
            [(X[i][j] - self.mean[j]) / self.std[j] for j in range(n_features)]
            for i in range(n_samples)
        ]
        self.weights = [0.0] * n_features
        self.bias = 0.0

        # Gradient descent optimization
        for _ in range(self.epochs):
            dw = [0.0] * n_features
            db = 0.0
            for i in range(n_samples):
                z = sum(w * x for w, x in zip(self.weights, X_scaled[i])) + self.bias
                z = max(min(z, 20.0), -20.0)
                pred = 1.0 / (1.0 + math.exp(-z))
                error = pred - y[i]
                for j in range(n_features):
                    dw[j] += error * X_scaled[i][j]
                db += error
            for j in range(n_features):
                self.weights[j] -= (self.lr / n_samples) * dw[j]
            self.bias -= (self.lr / n_samples) * db
        return self

    def predict_proba(self, X: List[List[float]]) -> List[float]:
        probs = []
        n_features = len(self.weights) if self.weights else 0
        for row in X:
            if not self.weights or len(row) < n_features:
                probs.append(0.5)
                continue
            scaled = [(row[j] - self.mean[j]) / self.std[j] for j in range(n_features)]
            z = sum(w * x for w, x in zip(self.weights, scaled)) + self.bias
            z = max(min(z, 20.0), -20.0)
            probs.append(round(1.0 / (1.0 + math.exp(-z)), 4))
        return probs

    def predict(self, X: List[List[float]]) -> List[int]:
        probs = self.predict_proba(X)
        return [1 if p >= 0.5 else 0 for p in probs]


class ChurnClassifier(Model):
    """Customer Churn classifier model predicting whether a customer will churn (0 or 1)."""

    def __init__(
        self,
        name: str = "churn_classifier",
        config: Optional[Dict[str, Any]] = None,
        n_estimators: int = 100,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(name=name, config=config, **kwargs)
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimator: Optional[Any] = None
        self.pure_clf: Optional[PureLogisticClassifier] = None

        try:
            from telecom_churn.data import FEATURE_COLUMNS
            self.feature_names = list(FEATURE_COLUMNS)
        except ImportError:
            try:
                from data import FEATURE_COLUMNS
                self.feature_names = list(FEATURE_COLUMNS)
            except ImportError:
                self.feature_names = list(DEFAULT_FEATURE_COLUMNS)

        self._init_estimator()

    def _init_estimator(self) -> None:
        """Initializes underlying classifier engine."""
        try:
            from sklearn.ensemble import RandomForestClassifier

            self.estimator = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                class_weight="balanced",
            )
        except ImportError:
            self.estimator = None

    def fit(self, X: List[List[float]], y: List[int]) -> ChurnClassifier:
        """Fits classifier on numeric feature matrix X and label vector y."""
        if not X:
            return self

        if self.estimator is not None:
            self.estimator.fit(X, y)
        else:
            self.pure_clf = PureLogisticClassifier()
            self.pure_clf.fit(X, y)
        return self

    def predict(self, inputs: Any, **kwargs: Any) -> List[int]:
        """Predicts binary churn class (0 for retained, 1 for churn)."""
        X = self._normalize_inputs(inputs)
        if not X:
            return []

        if self.estimator is not None and hasattr(self.estimator, "predict"):
            preds = self.estimator.predict(X)
            return [int(p) for p in preds]

        if self.pure_clf is not None:
            return self.pure_clf.predict(X)

        return [0 for _ in X]

    def predict_proba(self, inputs: Any) -> List[float]:
        """Computes probability of churn (risk score between 0.0 and 1.0)."""
        X = self._normalize_inputs(inputs)
        if not X:
            return []

        if self.estimator is not None and hasattr(self.estimator, "predict_proba"):
            probs = self.estimator.predict_proba(X)
            classes = list(getattr(self.estimator, "classes_", [0, 1]))
            if 1 in classes:
                idx_1 = classes.index(1)
                return [float(p[idx_1]) for p in probs]
            return [float(p[-1]) for p in probs]

        if self.pure_clf is not None:
            return self.pure_clf.predict_proba(X)

        return [0.5 for _ in X]

    def save(self, destination: Union[str, Path], **kwargs: Any) -> None:
        """Serializes model weights and parameters to disk."""
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        pure_data = None
        if self.pure_clf is not None:
            pure_data = {
                "weights": self.pure_clf.weights,
                "bias": self.pure_clf.bias,
                "mean": self.pure_clf.mean,
                "std": self.pure_clf.std,
            }

        payload = {
            "name": self.name,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "estimator": self.estimator,
            "pure_data": pure_data,
            "feature_names": getattr(self, "feature_names", []),
        }
        with open(dest_path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, source: Union[str, Path], **kwargs: Any) -> None:
        """Restores model weights and parameters from disk."""
        src_path = Path(source)
        if not src_path.is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {source}")

        with open(src_path, "rb") as f:
            payload = pickle.load(f)

        self.name = payload.get("name", self.name)
        self.n_estimators = payload.get("n_estimators", self.n_estimators)
        self.random_state = payload.get("random_state", self.random_state)
        self.estimator = payload.get("estimator", None)
        
        pure_data = payload.get("pure_data")
        if pure_data:
            self.pure_clf = PureLogisticClassifier()
            self.pure_clf.weights = pure_data.get("weights", [])
            self.pure_clf.bias = pure_data.get("bias", 0.0)
            self.pure_clf.mean = pure_data.get("mean", [])
            self.pure_clf.std = pure_data.get("std", [])
        else:
            self.pure_clf = payload.get("pure_clf", None)

        if "feature_names" in payload:
            self.feature_names = payload["feature_names"]

    def _normalize_inputs(self, inputs: Any) -> List[List[float]]:
        """Converts diverse input formats into 2D float feature matrix."""
        feature_cols = getattr(self, "feature_names", None) or DEFAULT_FEATURE_COLUMNS

        if isinstance(inputs, dict):
            # If payload is wrapped in {"features": [...]}
            if "features" in inputs:
                return self._normalize_inputs(inputs["features"])

            # Map dictionary keys with case and punctuation insensitivity
            norm_map = {k.lower().replace("_", "").replace("-", ""): v for k, v in inputs.items()}
            row = []
            matched = False
            for col in feature_cols:
                col_key = col.lower().replace("_", "").replace("-", "")
                if col in inputs:
                    try:
                        row.append(float(inputs[col]))
                        matched = True
                    except (ValueError, TypeError):
                        row.append(0.0)
                elif col_key in norm_map:
                    try:
                        row.append(float(norm_map[col_key]))
                        matched = True
                    except (ValueError, TypeError):
                        row.append(0.0)
                else:
                    row.append(0.0)

            # If no column names matched at all, but dictionary has numeric values, use them directly
            if not matched:
                numeric_vals = []
                for v in inputs.values():
                    try:
                        numeric_vals.append(float(v))
                    except (ValueError, TypeError):
                        pass
                if numeric_vals:
                    # Pad or slice to match expected feature count
                    while len(numeric_vals) < len(feature_cols):
                        numeric_vals.append(0.0)
                    return [numeric_vals[:len(feature_cols)]]

            return [row]

        if isinstance(inputs, list):
            if not inputs:
                return []
            if isinstance(inputs[0], dict):
                return [self._normalize_inputs(item)[0] for item in inputs]
            if isinstance(inputs[0], (int, float)):
                return [[float(x) for x in inputs]]
            if isinstance(inputs[0], list):
                return [[float(x) for x in row] for row in inputs]

        return []
