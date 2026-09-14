"""Data Pillar: modelkit/data.py

Standardizes dataset ingestion, schema validation, and partition contracts.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from modelkit.config import BaseConfig


class Dataset:
    """Standardizes dataset ingestion, schema validation, and partition contracts.

    Subclass this class to define your dataset, just like models.Model in Django.
    Importing or defining an empty Dataset subclass does nothing until load() or
    training is executed.
    """

    source: Optional[Union[str, Path]] = None
    filename: Optional[str] = None
    combine_all: bool = False

    def __init_subclass__(cls, name: Optional[str] = None, **kwargs: Any) -> None:
        """Automatically registers Dataset subclasses into the ModelKit registry."""
        super().__init_subclass__(**kwargs)
        from modelkit.registry import register_class

        register_class("dataset", cls, name=name)

    def __init__(
        self,
        name: str = "dataset",
        source: Optional[Union[str, Path]] = None,
        config: Optional[BaseConfig] = None,
        filename: Optional[str] = None,
        combine_all: Optional[bool] = None,
    ) -> None:
        """Initializes dataset metadata, source paths, and configuration binds.

        Args:
            name: Dataset naming identifier.
            source: Optional file path or URI anchor.
            config: Optional BaseConfig instance.
            filename: Optional explicit file name within data/ directory.
            combine_all: If True, combines all matching tabular files in data/.
        """
        self.name: str = name
        self.config: Optional[BaseConfig] = config
        if source is not None:
            self.source = Path(source)
        elif self.source is not None:
            self.source = Path(self.source)

        if filename is not None:
            self.filename = filename
        if combine_all is not None:
            self.combine_all = combine_all

        self.columns: List[str] = []
        self._data: List[Dict[str, Any]] = []
        self._train_data: Optional[List[Dict[str, Any]]] = None
        self._val_data: Optional[List[Dict[str, Any]]] = None
        self._test_data: Optional[List[Dict[str, Any]]] = None

    def _get_data_dir(self) -> Path:
        """Resolves the data directory from config or fallback cwd/data."""
        if self.config is not None and hasattr(self.config, "data_dir"):
            return Path(self.config.data_dir)
        return Path.cwd() / "data"

    def _resolve_file_path(self, source: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """Resolves the dataset file path using explicit source, filename attribute, or convention."""
        if source is not None:
            s_path = Path(source)
            if s_path.is_file():
                return s_path
            data_dir = self._get_data_dir()
            if (data_dir / s_path.name).is_file():
                return data_dir / s_path.name
            return s_path

        if self.source is not None:
            s_path = Path(self.source)
            if s_path.is_file():
                return s_path
            data_dir = self._get_data_dir()
            if (data_dir / s_path.name).is_file():
                return data_dir / s_path.name
            if self.config is not None and getattr(self.config, "config_path", None):
                root = Path(self.config.config_path).parent
                if (root / s_path).is_file():
                    return root / s_path
            return s_path

        data_dir = self._get_data_dir()

        if self.filename is not None:
            return data_dir / self.filename

        # Convention checks in data_dir
        if data_dir.is_dir():
            for name in ["dataset.csv", "train.csv", "data.csv"]:
                candidate = data_dir / name
                if candidate.is_file():
                    return candidate

            csv_files = sorted(data_dir.glob("*.csv"))
            if csv_files:
                return csv_files[0]

        return None

    def _read_file(self, file_path: Path, **kwargs: Any) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Reads a tabular data file (CSV) and returns (columns, records)."""
        if not file_path.is_file():
            return [], []

        try:
            import pandas as pd

            df = pd.read_csv(file_path, **kwargs)
            return list(df.columns), df.to_dict(orient="records")
        except ImportError:
            pass

        with open(file_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            records = list(reader)
        return columns, records

    def load(self, source: Optional[Union[str, Path]] = None, **kwargs: Any) -> Any:
        """Ingests data into memory. Returns loaded records."""
        data_dir = self._get_data_dir()

        # 1. Combine all CSV files in data_dir if combine_all is True
        if self.combine_all and data_dir.is_dir():
            csv_files = sorted(data_dir.glob("*.csv"))
            all_records: List[Dict[str, Any]] = []
            cols: List[str] = []
            for cf in csv_files:
                c, r = self._read_file(cf, **kwargs)
                if not cols:
                    cols = c
                all_records.extend(r)
            self.columns = cols
            self._data = all_records
            return self._data

        # 2. Check for pre-split convention: train.csv and test.csv in data_dir
        train_file = data_dir / "train.csv"
        if train_file.is_file() and not self.filename and source is None and self.source is None:
            cols, train_records = self._read_file(train_file, **kwargs)
            self.columns = cols
            self._train_data = train_records
            self._data = list(train_records)

            test_file = data_dir / "test.csv"
            if test_file.is_file():
                _, test_records = self._read_file(test_file, **kwargs)
                self._test_data = test_records

            val_file = data_dir / "val.csv"
            if not val_file.is_file():
                val_file = data_dir / "validation.csv"
            if val_file.is_file():
                _, val_records = self._read_file(val_file, **kwargs)
                self._val_data = val_records

            return self._data

        # 3. Standard file resolution
        resolved_path = self._resolve_file_path(source)
        if resolved_path is not None and resolved_path.is_file():
            self.columns, self._data = self._read_file(resolved_path, **kwargs)
            return self._data

        self._data = []
        return self._data

    def split(
        self,
        train: float = 0.8,
        validation: float = 0.1,
        test: float = 0.1,
        **kwargs: Any,
    ) -> Tuple[Any, Any, Any]:
        """Partitions the dataset rows into (train, validation, test) subsets."""
        if self._train_data is not None and (self._test_data is not None or self._val_data is not None):
            train_data = self._train_data
            test_data = self._test_data or []
            val_data = self._val_data

            if val_data is None and validation > 0 and len(train_data) > 1:
                n_val = max(1, int(len(train_data) * validation))
                val_data = train_data[-n_val:]
                train_data = train_data[:-n_val]
            elif val_data is None:
                val_data = []

            return train_data, val_data, test_data

        if not self._data:
            self.load()

        n = len(self._data)
        if n == 0:
            return [], [], []

        n_train = max(1, int(n * train))
        n_val = int(n * validation)

        train_data = self._data[:n_train]
        val_data = self._data[n_train : n_train + n_val]
        test_data = self._data[n_train + n_val :]
        return train_data, val_data, test_data

    def validate(self) -> bool:
        """Verifies that dataset exists and contains records or valid source."""
        if self._data and len(self._data) > 0:
            return True
        if self.source is not None or self.filename is not None:
            return True
        self.load()
        return bool(self._data and len(self._data) > 0)
