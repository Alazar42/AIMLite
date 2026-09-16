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
    """Customer Churn dataset loader for Kaggle Telecom Churn CSV data.

    Expected CSV file: data/telecom_churn.csv
    Kaggle: https://www.kaggle.com/datasets/barun2104/telecom-churn
    """

    # Explicitly set the data file inside the project's data/ directory
    filename: str = "telecom_churn.csv"

    def __init__(
        self,
        filename: str = "telecom_churn.csv",
        source: Optional[str | Path] = None,
        data_path: Optional[str | Path] = None,
        **kwargs: Any,
    ) -> None:
        src = source or data_path
        super().__init__(filename=filename, source=src, **kwargs)
        self.feature_names: List[str] = FEATURE_COLUMNS
        self.target_name: str = TARGET_COLUMN

    def load(self, source: Optional[str | Path] = None, **kwargs: Any) -> List[Dict[str, Any]]:
        """Parses the CSV and returns clean structured records.

        Supports pandas if installed, with a zero-dependency csv fallback.
        Populates self._data and self.columns for ModelKit validation and training.
        """
        target = source or self.source or self.filename
        resolved = self._resolve_file_path(target)
        if not resolved or not resolved.is_file():
            data_dir = self._get_data_dir()
            candidate = data_dir / self.filename
            if candidate.is_file():
                resolved = candidate
            else:
                self._data = []
                return []

        self.resolved_path = resolved

        # Try pandas first for high-performance reading
        try:
            import pandas as pd

            df = pd.read_csv(resolved)
            # Normalize column names (strip whitespace)
            df.columns = [c.strip() for c in df.columns]
            self.columns = list(df.columns)
            self._data = df.to_dict(orient="records")
            return self._data
        except ImportError:
            pass

        # Zero-dependency csv reader fallback
        records: List[Dict[str, Any]] = []
        with open(resolved, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.columns = [c.strip() for c in (reader.fieldnames or [])]
            for row in reader:
                record = {}
                for k, v in row.items():
                    k_clean = k.strip()
                    try:
                        record[k_clean] = float(v.strip())
                    except ValueError:
                        record[k_clean] = v.strip()
                records.append(record)

        self._data = records
        return self._data

    def to_arrays(self) -> Tuple[List[List[float]], List[int]]:
        """Converts loaded records into numeric feature matrix X and target labels y."""
        if not self._data:
            self.load()
        records = self._data
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
