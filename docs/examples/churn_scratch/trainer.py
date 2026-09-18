"""Customer Churn Prediction (Scratch Model Paradigm): trainer.py

Orchestrates data preparation, train/val splitting, classifier fitting,
and model checkpoint persistence into the artifacts directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from aimlite import BaseTrainer, Dataset, Model


class ChurnTrainer(BaseTrainer):
    """Trainer pipeline for fitting ChurnClassifier on TelecomChurnDataset."""

    def fit(self, model: Model, dataset: Dataset, **kwargs: Any) -> Dict[str, Any]:
        """Trains the churn classifier model and saves artifacts."""
        if not hasattr(dataset, "to_arrays"):
            return {"status": "failed", "error": "Dataset missing to_arrays method"}

        X, y = dataset.to_arrays()
        if not X:
            return {"status": "failed", "error": "No training samples found in dataset"}

        # 80/20 train/validation split
        split_idx = int(len(X) * 0.8)
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_val, y_val = X[split_idx:], y[split_idx:]

        if hasattr(model, "fit"):
            model.fit(X_train, y_train)

        # Compute train & val performance
        train_preds = model.predict(X_train)
        val_preds = model.predict(X_val) if X_val else []

        train_acc = (
            sum(1 for p, actual in zip(train_preds, y_train) if p == actual) / len(y_train)
            if y_train
            else 0.0
        )
        val_acc = (
            sum(1 for p, actual in zip(val_preds, y_val) if p == actual) / len(y_val)
            if y_val
            else 0.0
        )

        # Save checkpoint to project artifacts directory
        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = artifacts_dir / "churn_classifier.pkl"
        model.save(checkpoint_path)

        return {
            "status": "completed",
            "samples_total": len(X),
            "samples_train": len(X_train),
            "samples_val": len(X_val),
            "train_accuracy": round(train_acc, 4),
            "val_accuracy": round(val_acc, 4),
            "checkpoint": str(checkpoint_path),
        }
