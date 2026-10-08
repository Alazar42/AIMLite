"""Instruction Tuning (Adapter Model Paradigm): trainer.py

LoRA parameter-efficient training pipeline. Freezes foundation model weights
and optimizes solely adapter matrices, persisting lightweight delta checkpoints
and learned instruction-response generation memory into artifacts/adapter/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from aimlite.adapters import AdapterModel, AdapterTrainer
from aimlite.data import Dataset
from aimlite.models import Model


class AdapterInstructionTrainer(AdapterTrainer):
    """Trainer orchestrator for parameter-efficient LoRA fine-tuning."""

    def __init__(self, epochs: int = 3, learning_rate: float = 2e-4, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.epochs = epochs
        self.learning_rate = learning_rate

    def fit(self, model: Model, dataset: Dataset, **kwargs: Any) -> Dict[str, Any]:
        """Runs LoRA fine-tuning loop over instruction-tuning dataset records."""
        if not isinstance(model, AdapterModel):
            raise TypeError(f"AdapterInstructionTrainer requires AdapterModel, got {type(model).__name__}")

        records = dataset.load()
        if not records:
            raise ValueError("No instruction records found in dataset to train.")

        epochs = int(kwargs.get("epochs", self.epochs))
        lr = float(kwargs.get("learning_rate", kwargs.get("lr", self.learning_rate)))

        # 1. Execute true gradient updates on low-rank delta matrices
        train_result = super().fit(model, dataset, epochs=epochs, lr=lr)

        # 2. Index learned instruction-response pairs into adapter checkpoint memory
        in_dim = 64
        if hasattr(model, "lora_layers") and "q_proj" in model.lora_layers:
            in_dim = model.lora_layers["q_proj"].in_features

        memory: List[Dict[str, Any]] = []
        for r in records:
            inst = r.get("instruction") or r.get("prompt") or ""
            resp = r.get("response") or r.get("output") or r.get("completion") or ""
            if inst and resp:
                vec = self._text_to_feature_vector(inst, in_dim)
                memory.append({
                    "instruction": inst,
                    "response": resp,
                    "vector": vec,
                })

        model.adapter_weights["instruction_memory"] = memory
        if hasattr(model, "instruction_memory"):
            model.instruction_memory = memory

        # 3. Persist lightweight adapter checkpoint
        checkpoint_dir = Path("artifacts") / "adapter"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        model.save(checkpoint_dir)

        return {
            "status": "completed",
            "paradigm": "lora_adapter",
            "epochs": epochs,
            "learning_rate": lr,
            "training_samples": len(records),
            "checkpoint_directory": str(checkpoint_dir),
            "final_loss": train_result.get("final_loss", 0.0),
            "parameter_stats": model.get_trainable_parameters(),
        }
