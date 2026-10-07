"""Customer Churn Prediction (Scratch Model Paradigm): model.py

Classifier model for customer churn prediction using scikit-learn.
Bring Your Own Framework: AIMLite executes user-defined models seamlessly.
Install dependencies into your project with:
    aimlite install scikit-learn pandas
"""

from __future__ import annotations

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
        self.is_fitted: bool = False

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
        """Initializes underlying scikit-learn classifier engine."""
        try:
            from sklearn.ensemble import RandomForestClassifier

            self.estimator = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                class_weight="balanced",
            )
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required to initialize ChurnClassifier. "
                "Install it into your project using: aimlite install scikit-learn"
            ) from e

    def fit(self, X: List[List[float]], y: List[int]) -> ChurnClassifier:
        """Fits classifier on numeric feature matrix X and label vector y."""
        if not X:
            return self

        if self.estimator is None:
            self._init_estimator()
        self.estimator.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, inputs: Any, **kwargs: Any) -> List[int]:
        """Predicts binary churn class (0 for retained, 1 for churn)."""
        X = self._normalize_inputs(inputs)
        if not X:
            return []

        if self.estimator is None:
            raise RuntimeError("ChurnClassifier is not trained or loaded. Call fit() or load() first.")

        preds = self.estimator.predict(X)
        return [int(p) for p in preds]

    def predict_proba(self, inputs: Any) -> List[float]:
        """Computes probability of churn (risk score between 0.0 and 1.0)."""
        X = self._normalize_inputs(inputs)
        if not X:
            return []

        if self.estimator is None:
            raise RuntimeError("ChurnClassifier is not trained or loaded. Call fit() or load() first.")

        probs = self.estimator.predict_proba(X)
        classes = list(getattr(self.estimator, "classes_", [0, 1]))
        if 1 in classes:
            idx_1 = classes.index(1)
            return [float(p[idx_1]) for p in probs]
        return [float(p[-1]) for p in probs]

    def save(self, destination: Union[str, Path], **kwargs: Any) -> None:
        """Serializes model weights and parameters to disk."""
        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "name": self.name,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
            "estimator": self.estimator,
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
        self.is_fitted = self.estimator is not None

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
