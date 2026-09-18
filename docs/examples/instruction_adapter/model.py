"""Instruction Tuning (Adapter Model Paradigm): model.py

LoRA / PEFT AdapterModel subclass.
Attaches lightweight low-rank adaptation layers onto foundation models,
freezing base weights, calculating trainable parameter savings, and
persisting solely adapter weight deltas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from aimlite.adapters import AdapterConfig, AdapterModel, LoRALayer


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
            base_model_name=base_model_name,
            **kwargs,
        )
        self.base_model_name = base_model_name
        self.freeze_base_model()

        # Initialize real mathematical LoRA layers
        self.lora_layers = {
            "q_proj": LoRALayer(in_features=64, out_features=64, r=lora_rank, alpha=lora_alpha),
            "v_proj": LoRALayer(in_features=64, out_features=64, r=lora_rank, alpha=lora_alpha),
        }

        # Multi-adapter setup: register a specialized coding adapter
        coding_cfg = AdapterConfig(
            r=16,
            alpha=32.0,
            target_modules=["q_proj", "v_proj", "k_proj"],
            base_model_path=base_model_name,
        )
        self.add_adapter("code_specialist", coding_cfg)

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

        active_adapter_info = self.adapter_manager.get_active_adapter()
        active_name = active_adapter_info[0] if active_adapter_info else "base_frozen"
        active_r = active_adapter_info[1].r if active_adapter_info else 0

        # Simulate low-rank transformation representation
        sample_vec = [float((i + len(instruction)) % 10) / 10.0 for i in range(64)]
        q_out = self.lora_layers["q_proj"].forward(sample_vec)

        response = (
            f"[LoRA-Adapted {self.base_model_name} (adapter='{active_name}', r={active_r})]: "
            f"Processed instruction successfully."
        )

        return {
            "prompt": prompt_header,
            "response": response,
            "adapter_name": active_name,
            "adapter_rank": active_r,
            "base_model": self.base_model_name,
            "latent_sample": [round(v, 4) for v in q_out[:4]],
        }

    def save(self, destination: Union[str, Path], **kwargs: Any) -> None:
        """Saves ONLY the lightweight adapter delta weights and configuration.

        Saves ~50KB instead of duplicating gigabytes of base model weights.
        """
        super().save(destination, **kwargs)

        dest_dir = Path(destination)
        # Also maintain adapter_model.json for legacy backwards-compatibility in examples
        weights_path = dest_dir / "adapter_model.json"
        with open(weights_path, "w", encoding="utf-8") as f:
            layer_repr = {
                k: {
                    "lora_A_shape": [len(v.lora_A), len(v.lora_A[0])],
                    "lora_B_shape": [len(v.lora_B), len(v.lora_B[0])],
                    "scaling": v.scaling,
                    "merged": v.merged,
                }
                for k, v in self.lora_layers.items()
            }
            json.dump(layer_repr, f, indent=2)

    def load(self, source: Union[str, Path], **kwargs: Any) -> None:
        """Loads adapter configuration and weight deltas from disk."""
        super().load(source, **kwargs)
