"""ModelKit App: trainer.py

Define your model training or fine-tuning workflow here.
Inherit from:
- BaseTrainer (standard training from scratch)
- AdapterTrainer (parameter-efficient fine-tuning / LoRA)
"""

from typing import Any, Dict
from modelkit import BaseTrainer, Dataset, Model


class AppTrainer(BaseTrainer):
    """Application training orchestrator."""

    def fit(self, model: Model, dataset: Dataset, **kwargs: Any) -> Dict[str, Any]:
        """Runs the optimization step and saves checkpoints."""
        return {"status": "completed"}
