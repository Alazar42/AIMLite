"""Instruction Tuning (Adapter Model Paradigm): inference.py

Production inference endpoint for serving fine-tuned LoRA adapter weights
on top of frozen foundation models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from aimlite import BaseInference, Model


class AdapterInference(BaseInference):
    """Inference handler serving LoRA-adapted language instructions."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.adapter_dir = Path("artifacts") / "adapter"

    def run(self, model: Model, raw_input: Any, **kwargs: Any) -> Dict[str, Any]:
        """Runs instruction generation through the adapter-tuned model."""
        if self.adapter_dir.is_dir():
            model.load(self.adapter_dir)

        result = model.predict(raw_input)
        return {
            **result,
            "status": "success",
        }

    def get_routes(self) -> Dict[str, Any]:
        """Declares HTTP route mappings for AIMLite server."""
        return {
            "POST /predict": self.run,
            "GET /health": self.health,
        }
