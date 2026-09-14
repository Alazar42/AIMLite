"""Data Pillar Header: modelkit/data.py

Standardizes dataset ingestion, schema validation, and partition contracts
across tabular, vision, text, or streaming inputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

if TYPE_CHECKING:
    from modelkit.config import BaseConfig


class Dataset(ABC):
    """Standardizes dataset ingestion, schema validation, and partition contracts."""

    def __init__(
        self,
        name: str,
        source: Optional[Union[str, Path]] = None,
        config: Optional[BaseConfig] = None,
    ) -> None:
        """Sets dataset naming metadata, file/URI anchors, and configuration binds.

        Args:
            name: Dataset naming metadata identifier.
            source: Optional file path or URI anchor to the data source.
            config: Optional BaseConfig instance binding project configurations.
        """
        self.name: str = name
        self.source: Optional[Path] = Path(source) if source is not None else None
        self.config: Optional[BaseConfig] = config
        self._data: Any = None

    @abstractmethod
    def load(self, source: Optional[Union[str, Path]] = None, **kwargs: Any) -> Any:
        """Ingests raw data into memory, buffers, or generator references.

        Args:
            source: Optional source URI/path override.
            **kwargs: Backend-specific ingestion flags.

        Returns:
            Reference to the loaded internal dataset instance.
        """
        ...

    @abstractmethod
    def split(
        self,
        train: float = 0.8,
        validation: float = 0.1,
        test: float = 0.1,
        **kwargs: Any,
    ) -> Tuple[Any, Any, Any]:
        """Partitions the dataset into train, validation, and test subsets based on fractional ratios.

        Args:
            train: Fractional ratio for training partition.
            validation: Fractional ratio for validation partition.
            test: Fractional ratio for test partition.
            **kwargs: Optional random seeds, stratification rules, or backend arguments.

        Returns:
            Tuple of (train_split, validation_split, test_split).
        """
        ...

    def validate(self) -> bool:
        """Verifies schema integrity, null-value thresholds, or required column/feature presence.

        Returns:
            Boolean indicating readiness for training or evaluation.
        """
        # Baseline validation checks if data has been loaded
        return self._data is not None or self.source is not None
