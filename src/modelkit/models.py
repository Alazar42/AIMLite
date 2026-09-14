"""Model Pillar Header: modelkit/models.py

Defines model initialization, checkpoint persistence, forward inference,
and standalone evaluation hooks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union


class Model(ABC):
    """Defines model initialization, checkpoint persistence, forward inference, and evaluation hooks."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None) -> None:
        """Initializes model identifiers, hyperparameter configurations, and internal instance handles.

        Args:
            name: Model identifier name.
            config: Optional hyperparameter and runtime configuration dictionary.
        """
        self.name: str = name
        self.config: Dict[str, Any] = config or {}

    @abstractmethod
    def predict(self, inputs: Any, **kwargs: Any) -> Any:
        """Executes forward inference across batch or single-instance inputs.

        Args:
            inputs: Raw or transformed input payloads, tensors, arrays, or text strings.
            **kwargs: Additional inference flags or parameters.

        Returns:
            Model predictions, logits, token streams, or class labels.
        """
        ...

    @abstractmethod
    def save(self, destination: Union[str, Path], **kwargs: Any) -> None:
        """Serializes model parameters, checkpoints, and weights to disk.

        Args:
            destination: Target directory or file destination path.
            **kwargs: Additional serialization parameters.
        """
        ...

    @abstractmethod
    def load(self, source: Union[str, Path], **kwargs: Any) -> None:
        """Restores weights from storage or connects to an external local runtime process.

        Args:
            source: Checkpoint directory, file path, or service connection string.
            **kwargs: Additional restoration parameters.
        """
        ...

    def evaluate(self, dataset: Any, **kwargs: Any) -> Dict[str, float]:
        """Optional baseline hook computing performance metrics directly on a target dataset.

        Args:
            dataset: Evaluation partition or dataset instance.
            **kwargs: Additional evaluation arguments.

        Returns:
            Dictionary of metric keys mapped to numerical scores.
        """
        return {}
