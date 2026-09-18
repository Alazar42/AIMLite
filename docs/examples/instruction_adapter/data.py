"""Instruction Tuning (Adapter Model Paradigm): data.py

Dataset loader for instruction fine-tuning prompt-response datasets.
Ingests JSON or JSONL records containing prompt/instruction/output pairs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from aimlite import Dataset

SAMPLE_INSTRUCTIONS = [
    {
        "instruction": "Summarize the customer feedback in one sentence.",
        "input": "The onboarding was effortless and the support team responded in 5 minutes.",
        "output": "Customer experienced rapid onboarding and responsive 5-minute support.",
    },
    {
        "instruction": "Extract action items from the meeting notes.",
        "input": "Bob will deploy the model checkpoint by Friday. Alice will update docs.",
        "output": "- Bob: Deploy checkpoint by Friday\n- Alice: Update documentation",
    },
    {
        "instruction": "Convert the SQL query intent into structured parameters.",
        "input": "Find all users who churned last month with active contracts.",
        "output": "SELECT * FROM users WHERE churned = 1 AND contract_active = 1 AND date >= DATE_SUB(NOW(), INTERVAL 1 MONTH);",
    },
]


class InstructionDataset(Dataset):
    """Instruction tuning dataset loader formatting prompt-response pairs for LoRA adaptation."""

    def __init__(
        self,
        source: Optional[str | Path] = None,
        data_path: Optional[str | Path] = None,
        **kwargs: Any,
    ) -> None:
        src = source or data_path
        super().__init__(source=src, **kwargs)

    def load(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Loads instruction tuning pairs from data directory or sample instructions."""
        resolved = self._resolve_file_path(self.source)
        if resolved and resolved.is_file():
            try:
                content = resolved.read_text(encoding="utf-8")
                if resolved.suffix in (".jsonl", ".txt"):
                    records = [json.loads(line) for line in content.splitlines() if line.strip()]
                else:
                    records = json.loads(content)
                if isinstance(records, list):
                    return records
            except Exception:
                pass

        # Check for any .json files in data/
        data_dir = Path("data")
        for jf in data_dir.glob("*.json"):
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                if isinstance(data, list) and data and "instruction" in data[0]:
                    return data
            except Exception:
                continue

        return SAMPLE_INSTRUCTIONS

    def format_prompts(self) -> List[Dict[str, str]]:
        """Formats records into standardized prompt/completion strings."""
        records = self.load()
        formatted: List[Dict[str, str]] = []
        for r in records:
            inst = r.get("instruction", "")
            inp = r.get("input", "")
            out = r.get("output", "")
            prompt = f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n" if inp else f"### Instruction:\n{inst}\n\n### Response:\n"
            formatted.append({"prompt": prompt, "completion": out})
        return formatted
