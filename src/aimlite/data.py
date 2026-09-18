"""Data Pillar: aimlite/data.py

Standardizes dataset ingestion, schema validation, and partition contracts.
Supports CSV, TSV, JSON, JSONL, Parquet, TXT, and custom data formats.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from aimlite.config import BaseConfig

SUPPORTED_EXTENSIONS = [
    ".csv",
    ".tsv",
    ".tab",
    ".json",
    ".jsonl",
    ".parquet",
    ".pq",
    ".txt",
]


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
        """Automatically registers Dataset subclasses into the AIMLite registry."""
        super().__init_subclass__(**kwargs)
        from aimlite.registry import register_class

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

        self.filename: Optional[str] = filename or getattr(self.__class__, "filename", None)
        if combine_all is not None:
            self.combine_all = combine_all

        self.resolved_path: Optional[Path] = None
        self.columns: List[str] = []
        self._data: List[Dict[str, Any]] = []
        self._train_data: Optional[List[Dict[str, Any]]] = None
        self._val_data: Optional[List[Dict[str, Any]]] = None
        self._test_data: Optional[List[Dict[str, Any]]] = None

    def __len__(self) -> int:
        """Returns the number of ingested records."""
        return len(self._data)

    def __iter__(self):
        """Allows iteration over records."""
        return iter(self._data)

    def __getitem__(self, index: Any) -> Any:
        """Enables indexing over dataset records."""
        return self._data[index]

    def _get_data_dir(self) -> Path:
        """Resolves the data directory from config or fallback cwd/data."""
        if self.config is not None and hasattr(self.config, "data_dir"):
            return Path(self.config.data_dir)
        return Path.cwd() / "data"

    def get_file_path(self) -> Optional[Path]:
        """Returns the resolved file path for this dataset."""
        if self.resolved_path is not None and self.resolved_path.is_file():
            return self.resolved_path
        return self._resolve_file_path()

    def _resolve_file_path(self, source: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """Resolves the dataset file path using explicit source, filename attribute, or convention."""
        data_dir = self._get_data_dir()

        # 1. Explicit source passed to load()
        if source is not None:
            s_path = Path(source)
            if s_path.is_file():
                self.resolved_path = s_path
                return s_path
            if data_dir.is_dir() and (data_dir / s_path.name).is_file():
                self.resolved_path = data_dir / s_path.name
                return self.resolved_path
            self.resolved_path = s_path
            return s_path

        # 2. Explicit self.source attribute on class or instance
        if self.source is not None:
            s_path = Path(self.source)
            if s_path.is_file():
                self.resolved_path = s_path
                return s_path
            if data_dir.is_dir() and (data_dir / s_path.name).is_file():
                self.resolved_path = data_dir / s_path.name
                return self.resolved_path
            if self.config is not None and getattr(self.config, "config_path", None):
                root = Path(self.config.config_path).parent
                if (root / s_path).is_file():
                    self.resolved_path = root / s_path
                    return self.resolved_path
            self.resolved_path = s_path
            return s_path

        # 3. Explicit self.filename attribute on class or instance
        fn = self.filename or getattr(self.__class__, "filename", None)
        if fn:
            target = data_dir / fn
            if target.is_file():
                self.resolved_path = target
                return target
            if Path(fn).is_file():
                self.resolved_path = Path(fn)
                return self.resolved_path

        # Developer must specify filename or source; do not auto-load unmapped files
        return None

    def _read_file(self, file_path: Path, **kwargs: Any) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Reads a data file of any supported format (CSV, TSV, JSON, JSONL, Parquet, TXT) and returns (columns, records)."""
        if not file_path.is_file():
            return [], []

        # Safe guard: empty file check
        try:
            if file_path.stat().st_size == 0:
                return [], []
        except OSError:
            return [], []

        ext = file_path.suffix.lower()

        # 1. JSON (array of objects or key-value object)
        if ext == ".json":
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = json.load(f)
                if isinstance(content, list):
                    if content and isinstance(content[0], dict):
                        return list(content[0].keys()), content
                    return ["value"], [{"value": item} for item in content]
                elif isinstance(content, dict):
                    return list(content.keys()), [content]
            except Exception:
                pass

        # 2. JSONL (newline-delimited JSON)
        elif ext == ".jsonl":
            try:
                records: List[Dict[str, Any]] = []
                cols: List[str] = []
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            obj = json.loads(line)
                            if isinstance(obj, dict):
                                if not cols:
                                    cols = list(obj.keys())
                                records.append(obj)
                return cols, records
            except Exception:
                pass

        # 3. Parquet
        elif ext in [".parquet", ".pq"]:
            try:
                import pandas as pd

                df = pd.read_parquet(file_path, **kwargs)
                return list(df.columns), df.to_dict(orient="records")
            except Exception:
                pass

        # 4. TSV / Tab-delimited
        elif ext in [".tsv", ".tab"]:
            try:
                import pandas as pd

                df = pd.read_csv(file_path, sep="\t", **kwargs)
                if not df.empty:
                    return list(df.columns), df.to_dict(orient="records")
            except Exception:
                pass

            try:
                with open(file_path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    cols = list(reader.fieldnames or [])
                    records = list(reader)
                    return cols, records
            except Exception:
                pass

        # 5. CSV
        elif ext == ".csv":
            try:
                import pandas as pd

                df = pd.read_csv(file_path, **kwargs)
                if not df.empty or len(df.columns) > 0:
                    return list(df.columns), df.to_dict(orient="records")
            except Exception:
                pass

            try:
                with open(file_path, "r", encoding="utf-8", newline="") as f:
                    reader = csv.DictReader(f)
                    cols = list(reader.fieldnames or [])
                    records = list(reader)
                    return cols, records
            except Exception:
                pass

        # 6. TXT / Raw text
        elif ext == ".txt":
            try:
                lines = file_path.read_text(encoding="utf-8").splitlines()
                records = [{"text": line} for line in lines if line.strip()]
                return ["text"], records
            except Exception:
                pass

        # 7. Generic tabular fallback
        try:
            import pandas as pd

            df = pd.read_csv(file_path, **kwargs)
            return list(df.columns), df.to_dict(orient="records")
        except Exception:
            pass

        return [], []

    def load(self, source: Optional[Union[str, Path]] = None, **kwargs: Any) -> Any:
        """Ingests data into memory. Returns loaded records."""
        data_dir = self._get_data_dir()

        # 1. Combine all data files in data_dir if combine_all is True
        if self.combine_all and data_dir.is_dir():
            data_files = sorted([
                f for f in data_dir.iterdir()
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS and not f.name.startswith(".")
            ])
            all_records: List[Dict[str, Any]] = []
            cols: List[str] = []
            for cf in data_files:
                c, r = self._read_file(cf, **kwargs)
                if not cols and c:
                    cols = c
                all_records.extend(r)
            self.columns = cols
            self._data = all_records
            return self._data

        # 2. Check for pre-split companion files if filename is train.*
        fn = self.filename or (Path(self.source).name if self.source else None)
        if fn and Path(fn).stem == "train" and data_dir.is_dir():
            target = data_dir / fn if not Path(fn).is_file() else Path(fn)
            if target.is_file() and target.stat().st_size > 0:
                cols, train_records = self._read_file(target, **kwargs)
                self.columns = cols
                self._train_data = train_records
                self._data = list(train_records)

                for ext in SUPPORTED_EXTENSIONS:
                    test_file = data_dir / f"test{ext}"
                    if test_file.is_file():
                        _, test_records = self._read_file(test_file, **kwargs)
                        self._test_data = test_records
                        break

                for base in ["val", "validation"]:
                    for ext in SUPPORTED_EXTENSIONS:
                        val_file = data_dir / f"{base}{ext}"
                        if val_file.is_file():
                            _, val_records = self._read_file(val_file, **kwargs)
                            self._val_data = val_records
                            break

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
        """Verifies that dataset exists and contains records."""
        if not self.filename and self.source is None and self.resolved_path is None:
            return False
        if not self._data:
            records = self.load()
            if records and not self._data:
                self._data = records
        return bool(self._data and len(self._data) > 0)
