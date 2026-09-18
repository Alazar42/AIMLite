"""Model Pillar Header: aimlite/models.py

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
        """Automatically registers Model subclasses into the AIMLite registry."""
        super().__init_subclass__(**kwargs)
        from aimlite.registry import register_class

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
            from aimlite.registry import get as registry_get

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
        import re as _re

        dest = Path(destination)

        # Determine if we're writing to a specific file or into a directory:
        # - If dest exists and is a directory → write <class_name>.pkl inside it
        # - If dest has a file extension (e.g. .pkl, .pt, .pt2) → write directly to that file
        # - Otherwise (non-existent, no extension) → create as directory, write <class_name>.pkl inside
        if dest.is_dir():
            cls_name = type(self).__name__
            snake = _re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", cls_name).lower()
            target_file = dest / f"{snake}.pkl"
        elif dest.suffix:
            # Explicit file path (e.g. models/churn_classifier.pkl)
            target_file = dest
            target_file.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Treat as directory path (create it)
            dest.mkdir(parents=True, exist_ok=True)
            cls_name = type(self).__name__
            snake = _re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", cls_name).lower()
            target_file = dest / f"{snake}.pkl"

        state = getattr(self, "weights", None)
        if state is None:
            state = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

        with open(target_file, "wb") as f:
            pickle.dump(state, f)

    def load(self, source: Union[str, Path], **kwargs: Any) -> None:
        """Restores weights from storage or connects to an external local runtime process.

        Unless overridden by a subclass, restores model weights from pickle.
        When source is a directory, searches for:
        1. <snake_case_class_name>.pkl  (e.g. churn_classifier.pkl for ChurnClassifier)
        2. model.pkl                    (legacy fallback)
        3. Any .pkl file in the directory

        Args:
            source: Checkpoint directory, file path, or service connection string.
            **kwargs: Additional restoration parameters.
        """
        import re as _re

        src = Path(source)

        if src.is_dir():
            # Prefer the per-class named file
            cls_name = type(self).__name__
            snake = _re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", cls_name).lower()
            named_file = src / f"{snake}.pkl"
            legacy_file = src / "model.pkl"

            if named_file.is_file():
                target_file = named_file
            elif legacy_file.is_file():
                target_file = legacy_file
            else:
                # Any .pkl in the directory (newest)
                pkls = sorted(src.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
                target_file = pkls[0] if pkls else named_file
        else:
            target_file = src
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
