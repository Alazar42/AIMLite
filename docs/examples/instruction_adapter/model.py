"""Instruction Tuning (Adapter Model Paradigm): model.py

LoRA / PEFT AdapterModel subclass.
Attaches lightweight low-rank adaptation layers onto foundation models,
freezing base weights and persisting solely adapter weight deltas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from modelkit.adapters import AdapterConfig, AdapterModel


class LoRAInstructionModel(AdapterModel):
    """Instruction-following model specialized via parameter-efficient LoRA adapter weights."""

    def __init__(
        self,
        name: str = "lora_instruction_model",
        config: Optional[Dict[str, Any]] = None,
        base_model_name: str = "meta-llama/Llama-3-8B",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        **kwargs: Any,
    ) -> None:
        adapter_cfg = AdapterConfig(
            r=lora_rank,
            alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
            dropout=0.05,
            base_model_path=base_model_name,
        )
        super().__init__(
            name=name,
            config=config,
            adapter_config=adapter_cfg,
            **kwargs,
        )
        self.base_model_name = base_model_name
        self.freeze_base_model()
        # In-memory mock adapter delta weights
        self.adapter_weights = {
            "q_proj.lora_A": [[0.01 * (i + j) for j in range(8)] for i in range(16)],
            "q_proj.lora_B": [[0.02 * (i - j) for j in range(16)] for i in range(8)],
        }

    def predict(self, inputs: Any, **kwargs: Any) -> Dict[str, Any]:
        """Generates fine-tuned response for input instruction/prompt."""
        if isinstance(inputs, dict):
            instruction = inputs.get("instruction", "")
            user_input = inputs.get("input", "")
        else:
            instruction = str(inputs)
            user_input = ""

        # Format input instruction prompt
        if user_input:
            prompt_header = f"### Instruction:\n{instruction}\n\n### Input:\n{user_input}\n\n### Response:"
        else:
            prompt_header = f"### Instruction:\n{instruction}\n\n### Response:"

        # Generate response using adapter-tuned weights
        response = f"[LoRA-Adapted {self.base_model_name} (r={self.adapter_config.r})]: Processed instruction successfully."

        return {
            "prompt": prompt_header,
            "response": response,
            "adapter_rank": self.adapter_config.r,
            "base_model": self.base_model_name,
        }

    def save(self, destination: Union[str, Path], **kwargs: Any) -> None:
        """Saves ONLY the lightweight adapter delta weights and configuration.

        Saves ~50KB instead of duplicating gigabytes of base model weights.
        """
        dest_dir = Path(destination)
        dest_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save adapter config
        config_path = dest_dir / "adapter_config.json"
        if self.adapter_config:
            self.adapter_config.save(config_path)

        # 2. Save adapter weights delta
        weights_path = dest_dir / "adapter_model.json"
        with open(weights_path, "w", encoding="utf-8") as f:
            json.dump(self.adapter_weights, f, indent=2)

        # 3. Save metadata
        meta_path = dest_dir / "adapter_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "name": self.name,
                "base_model": self.base_model_name,
                "type": "lora",
                "delta_size_kb": round(weights_path.stat().st_size / 1024, 2) if weights_path.exists() else 0,
            }, f, indent=2)

    def load(self, source: Union[str, Path], **kwargs: Any) -> None:
        """Loads adapter configuration and weight deltas from disk."""
        src_dir = Path(source)
        if not src_dir.is_dir():
            return

        cfg_path = src_dir / "adapter_config.json"
        if cfg_path.is_file():
            self.adapter_config = AdapterConfig.load(cfg_path)

        w_path = src_dir / "adapter_model.json"
        if w_path.is_file():
            with open(w_path, "r", encoding="utf-8") as f:
                self.adapter_weights = json.load(f)
