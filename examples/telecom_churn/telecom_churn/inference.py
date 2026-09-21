"""Customer Churn Prediction (Scratch Model Paradigm): inference.py

Production inference handler for customer churn risk scoring and intervention routing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from aimlite import BaseInference, Model


class ChurnInference(BaseInference):
    """Production inference endpoint returning customer churn probability and retention decisions."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.checkpoint_path = Path("artifacts") / "churn_classifier.pkl"

    def run(self, model: Model, raw_input: Any, **kwargs: Any) -> Dict[str, Any]:
        """Runs churn prediction on incoming customer attributes."""
        # Ensure model weights are loaded if checkpoint exists
        if self.checkpoint_path.is_file():
            model.load(self.checkpoint_path)

        # Compute predictions and probability
        preds = model.predict(raw_input)
        churn_pred = preds[0] if preds else 0

        risk_score = 0.5
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(raw_input)
            if probs:
                risk_score = round(probs[0], 4)

        # Retention decision threshold
        decision = "Intervene (High Churn Risk)" if risk_score >= 0.5 else "Retain (Low Risk)"

        return {
            "churn_prediction": int(churn_pred),
            "churn_risk": risk_score,
            "decision": decision,
            "status": "success",
        }

    def get_routes(self) -> Dict[str, Any]:
        """Declares HTTP route mappings for AIMLite server."""
        return {
            "POST /predict": self.run,
            "GET /health": self.health,
        }
