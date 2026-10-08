"""Instruction Tuning (Adapter Model Paradigm): data.py

Dataset loader for instruction fine-tuning prompt-response datasets.
Ingests JSON or JSONL records containing prompt/instruction/output pairs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from aimlite import Dataset


class InstructionDataset(Dataset):
    """Instruction tuning dataset loader formatting prompt-response pairs for LoRA adaptation."""

    filename: str = "instructions.jsonl"

    def __init__(
        self,
        source: Optional[str | Path] = None,
        data_path: Optional[str | Path] = None,
        **kwargs: Any,
    ) -> None:
        src = source or data_path or self.filename
        super().__init__(source=src, **kwargs)

    def load(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Loads instruction tuning pairs from data directory.

        Raises FileNotFoundError if no instruction dataset is provided in data/.
        """
        resolved = self._resolve_file_path(self.source)
        if resolved and resolved.is_file():
            content = resolved.read_text(encoding="utf-8")
            if resolved.suffix in (".jsonl", ".txt"):
                records = [json.loads(line) for line in content.splitlines() if line.strip()]
            else:
                records = json.loads(content)
            if isinstance(records, list) and records:
                return records

        # Search data/ directory for any .jsonl or .json instruction files
        data_dir = self._get_data_dir() if hasattr(self, "_get_data_dir") else Path("data")
        for pattern in ("*.jsonl", "*.json"):
            for jf in data_dir.glob(pattern):
                try:
                    content = jf.read_text(encoding="utf-8")
                    if jf.suffix == ".jsonl":
                        records = [json.loads(line) for line in content.splitlines() if line.strip()]
                    else:
                        records = json.loads(content)
                    if isinstance(records, list) and records:
                        return records
                except Exception:
                    continue

        raise FileNotFoundError(
            "No instruction dataset found in data/. "
            "Please add an instructions.jsonl file with {'instruction': ..., 'response': ...} pairs to data/."
        )

    def format_prompts(self) -> List[Dict[str, str]]:
        """Formats records into standardized prompt/completion strings."""
        records = self.load()
        formatted: List[Dict[str, str]] = []
        for r in records:
            inst = r.get("instruction") or r.get("prompt") or ""
            inp = r.get("input", "")
            out = r.get("response") or r.get("output") or r.get("completion") or ""
            prompt = (
                f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:"
                if inp
                else f"### Instruction:\n{inst}\n\n### Response:"
            )
            formatted.append({"prompt": prompt, "completion": out})
        return formatted
