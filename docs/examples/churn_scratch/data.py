"""Customer Churn Prediction (Scratch Model Paradigm): data.py

Dataset loader for the Kaggle Telecom Churn dataset:
https://www.kaggle.com/datasets/barun2104/telecom-churn

Expected CSV columns:
  Churn (0/1), AccountWeeks, ContractRenewal, DataPlan, DataUsage,
  CustServCalls, DayMins, DayCalls, MonthlyCharge, OverageFee, RoamMins
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modelkit import Dataset

FEATURE_COLUMNS = [
    "AccountWeeks",
    "ContractRenewal",
    "DataPlan",
    "DataUsage",
    "CustServCalls",
    "DayMins",
    "DayCalls",
    "MonthlyCharge",
    "OverageFee",
    "RoamMins",
]
TARGET_COLUMN = "Churn"


class TelecomChurnDataset(Dataset):
    """Customer Churn dataset loader for Kaggle Telecom Churn CSV data."""

    def __init__(
        self,
        source: Optional[str | Path] = None,
        data_path: Optional[str | Path] = None,
        **kwargs: Any,
    ) -> None:
        src = source or data_path
        super().__init__(source=src, **kwargs)
        self.feature_names: List[str] = FEATURE_COLUMNS
        self.target_name: str = TARGET_COLUMN

    def load(self, **kwargs: Any) -> List[Dict[str, Any]]:
        """Parses the CSV and returns clean structured records.

        Supports pandas if installed, with a zero-dependency csv fallback.
        """
        resolved = self._resolve_file_path(self.source)
        if not resolved or not resolved.is_file():
            return []

        # Try pandas first for high-performance reading
        try:
            import pandas as pd

            df = pd.read_csv(resolved)
            # Normalize column names (strip whitespace)
            df.columns = [c.strip() for c in df.columns]
            return df.to_dict(orient="records")
        except ImportError:
            pass

        # Zero-dependency csv reader fallback
        records: List[Dict[str, Any]] = []
        with open(resolved, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                record = {}
                for k, v in row.items():
                    k_clean = k.strip()
                    try:
                        record[k_clean] = float(v.strip())
                    except ValueError:
                        record[k_clean] = v.strip()
                records.append(record)
        return records

    def to_arrays(self) -> Tuple[List[List[float]], List[int]]:
        """Converts loaded records into numeric feature matrix X and target labels y."""
        records = self.load()
        X: List[List[float]] = []
        y: List[int] = []

        for row in records:
            if self.target_name not in row:
                continue
            feats = [float(row.get(col, 0.0)) for col in self.feature_names]
            label = int(float(row[self.target_name]))
            X.append(feats)
            y.append(label)

        return X, y
