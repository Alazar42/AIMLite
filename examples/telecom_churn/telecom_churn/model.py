"""Customer Churn Prediction (Scratch Model Paradigm): model.py

Classifier model for customer churn prediction.
Supports scikit-learn (RandomForestClassifier) with pure-Python fallback.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from aimlite import Model


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
            self._baseline_threshold = sum(y) / len(y) if y else 0.5
        return self

    def predict(self, inputs: Any, **kwargs: Any) -> List[int]:
        """Predicts binary churn class (0 for retained, 1 for churn)."""
        X = self._normalize_inputs(inputs)
        if not X:
            return []

        if self.estimator is not None and hasattr(self.estimator, "predict"):
            preds = self.estimator.predict(X)
            return [int(p) for p in preds]

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

        return [0.15 for _ in X]

    def save(self, destination: Union[str, Path], **kwargs: Any) -> None:
        """Serializes model weights and parameters to disk."""
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "name": self.name,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "estimator": self.estimator,
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

    def _normalize_inputs(self, inputs: Any) -> List[List[float]]:
        """Converts diverse input formats into 2D float feature matrix."""
        from data import FEATURE_COLUMNS

        if isinstance(inputs, dict):
            return [[float(inputs.get(col, 0.0)) for col in FEATURE_COLUMNS]]
        if isinstance(inputs, list):
            if not inputs:
                return []
            if isinstance(inputs[0], dict):
                return [[float(item.get(col, 0.0)) for col in FEATURE_COLUMNS] for item in inputs]
            if isinstance(inputs[0], (int, float)):
                return [[float(x) for x in inputs]]
            if isinstance(inputs[0], list):
                return [[float(x) for x in row] for row in inputs]
        return []
