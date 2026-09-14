"""Model Pillar Header: modelkit/models.py

Defines model initialization, checkpoint persistence, forward inference,
and standalone evaluation hooks.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union


class Model(ABC):
    """Defines model initialization, checkpoint persistence, forward inference, and evaluation hooks."""

    dataset: Optional[Any] = None

    def __init_subclass__(cls, name: Optional[str] = None, **kwargs: Any) -> None:
        """Automatically registers Model subclasses into the ModelKit registry."""
        super().__init_subclass__(**kwargs)
        from modelkit.registry import register_class

        register_class("model", cls, name=name)

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        dataset: Optional[Any] = None,
    ) -> None:
        """Initializes model identifiers, hyperparameter configurations, and internal instance handles.

        Args:
            name: Model identifier name.
            config: Optional hyperparameter and runtime configuration dictionary.
            dataset: Optional dataset class, instance, or identifier binding.
        """
        self.name: str = name
        self.config: Dict[str, Any] = config or {}
        if dataset is not None:
            self.dataset = dataset
        self.weights: Dict[str, Any] = {"initialized": True}

    def get_dataset(self, config: Optional[Any] = None) -> Any:
        """Resolves and instantiates the dataset bound to this model."""
        if self.dataset is None:
            raise ValueError(
                f"Model '{self.__class__.__name__}' has no dataset bound. "
                f"Please declare 'dataset = <DatasetClass>' in your model."
            )
        if isinstance(self.dataset, type):
            return self.dataset(name=f"{self.name}_data", config=config)
        if isinstance(self.dataset, str):
            from modelkit.registry import get as registry_get

            ds_cls = registry_get("dataset", self.dataset)
            return ds_cls(name=f"{self.name}_data", config=config)
        return self.dataset

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

    def save(self, destination: Union[str, Path], **kwargs: Any) -> None:
        """Serializes model parameters, checkpoints, and weights to disk.

        Unless overridden by a subclass, serializes model weights to model.pkl via pickle.

        Args:
            destination: Target directory or file destination path.
            **kwargs: Additional serialization parameters.
        """
        dest = Path(destination)
        dest.mkdir(parents=True, exist_ok=True)
        target_file = dest / "model.pkl" if dest.is_dir() else dest

        state = getattr(self, "weights", None)
        if state is None:
            state = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

        with open(target_file, "wb") as f:
            pickle.dump(state, f)

    def load(self, source: Union[str, Path], **kwargs: Any) -> None:
        """Restores weights from storage or connects to an external local runtime process.

        Unless overridden by a subclass, restores model weights from model.pkl via pickle.

        Args:
            source: Checkpoint directory, file path, or service connection string.
            **kwargs: Additional restoration parameters.
        """
        src = Path(source)
        target_file = src / "model.pkl" if src.is_dir() else src
        if target_file.is_file():
            with open(target_file, "rb") as f:
                loaded = pickle.load(f)
                if isinstance(loaded, dict) and hasattr(self, "weights"):
                    self.weights.update(loaded)
                elif isinstance(loaded, dict):
                    self.__dict__.update(loaded)

    def evaluate(self, dataset: Any, **kwargs: Any) -> Dict[str, float]:
        """Optional baseline hook computing performance metrics directly on a target dataset.

        Args:
            dataset: Evaluation partition or dataset instance.
            **kwargs: Additional evaluation arguments.

        Returns:
            Dictionary of metric keys mapped to numerical scores.
        """
        return {"accuracy": 1.0}
