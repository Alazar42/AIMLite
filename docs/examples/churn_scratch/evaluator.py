"""Customer Churn Prediction (Scratch Model Paradigm): evaluator.py

Computes precision, recall, F1-score, and classification accuracy
on evaluation partitions.
"""

from __future__ import annotations

from typing import Any, Dict

from aimlite import BaseEvaluator, Dataset, Model


class ChurnEvaluator(BaseEvaluator):
    """Evaluator assessing customer churn classification metrics."""

    def evaluate(self, model: Model, dataset: Dataset, **kwargs: Any) -> Dict[str, float]:
        """Calculates accuracy, precision, recall, and F1 metrics."""
        if not hasattr(dataset, "to_arrays"):
            return {"accuracy": 0.0}

        X, y = dataset.to_arrays()
        if not X:
            return {"accuracy": 0.0}

        preds = model.predict(X)

        tp = sum(1 for p, actual in zip(preds, y) if p == 1 and actual == 1)
        fp = sum(1 for p, actual in zip(preds, y) if p == 1 and actual == 0)
        fn = sum(1 for p, actual in zip(preds, y) if p == 0 and actual == 1)
        tn = sum(1 for p, actual in zip(preds, y) if p == 0 and actual == 0)

        total = len(y)
        accuracy = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "sample_count": float(total),
        }
