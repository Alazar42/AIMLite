"""Lifecycle Pillar Header: aimlite/lifecycle.py

Defines BaseTrainer, BaseEvaluator, and BaseInference abstract base classes
for training orchestration, evaluation benchmarking, and live model serving.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    from aimlite.data import Dataset
    from aimlite.models import Model


def _model_weights_filename(model: Any) -> str:
    """Derives a per-model weights filename from the model class name.

    Converts CamelCase class names to snake_case.pkl so each model class
    gets a unique, predictable weights file:

        ChurnClassifier     → churn_classifier.pkl
        UserModel           → user_model.pkl
        BertEncoder         → bert_encoder.pkl
        AppModel            → app_model.pkl
    """
    cls_name = type(model).__name__
    # Insert underscore before each uppercase letter that follows a lowercase letter or digit
    snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", cls_name)
    return f"{snake.lower()}.pkl"


class BaseTrainer(ABC):
    """Coordinates optimization steps, data consumption, convergence tracking, and checkpointing."""

    def __init_subclass__(cls, name: Optional[str] = None, **kwargs: Any) -> None:
        """Automatically registers BaseTrainer subclasses into the AIMLite registry."""
        super().__init_subclass__(**kwargs)
        from aimlite.registry import register_class

        register_class("trainer", cls, name=name)

    @abstractmethod
    def fit(self, model: Model, dataset: Dataset, **kwargs: Any) -> Dict[str, Any]:
        """Coordinates optimization steps, data consumption, and convergence tracking.

        Args:
            model: Active Model instance.
            dataset: Active Dataset instance.
            **kwargs: Optimizer, epoch, batch size, or backend keyword arguments.

        Returns:
            Dictionary reporting training status, loss curves, and execution metrics.
        """
        ...

    def checkpoint(
        self,
        model: Model,
        destination: Union[str, Path],
        step: Optional[int] = None,
        experiments_dir: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Writes an experiment snapshot containing weights, config state, and metrics.

        Args:
            model: Active Model instance to serialize.
            destination: Path to target model checkpoint directory.
            step: Optional training step or epoch number.
            experiments_dir: Optional path to experiments directory for metadata tracking.

        Returns:
            Path to the saved checkpoint directory.
        """
        dest_path = Path(destination)
        if step is not None:
            checkpoint_dir = dest_path / f"checkpoint-step-{step}"
        else:
            checkpoint_dir = dest_path

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 1. Derive a per-model weights filename from the model class name.
        # Each trained model class gets its own uniquely-named checkpoint file:
        #   ChurnClassifier  → models/churn_classifier.pkl
        #   UserModel        → models/user_model.pkl
        weights_filename = _model_weights_filename(model)
        weights_file = checkpoint_dir / weights_filename

        # 2. Trigger model weights persistence.
        # First try passing the directory (allows save() to name files itself).
        # If save() tries to open() the directory as a file (IsADirectoryError),
        # gracefully retry with the explicit per-model .pkl file path.
        try:
            model.save(weights_file)
        except IsADirectoryError:
            # Custom save() opened the destination as a directory;
            # retry with a parent dir so it can create its own files.
            model.save(checkpoint_dir)

        # 3. Write experiment snapshot metadata into experiments/ folder
        exp_target_dir = None
        if experiments_dir is not None:
            exp_target_dir = Path(experiments_dir)
        elif (dest_path.parent / "experiments").is_dir():
            exp_target_dir = dest_path.parent / "experiments"
        else:
            exp_target_dir = checkpoint_dir

        exp_target_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "model_name": getattr(model, "name", "unknown"),
            "model_class": type(model).__name__,
            "weights_file": str(weights_file),
            "step": step,
            "timestamp": time.time(),
            "config": getattr(model, "config", {}),
        }
        # Name snapshot after the model class so multiple models don't clobber each other
        snapshot_name = f"{type(model).__name__.lower()}_snapshot.json"
        metadata_file = exp_target_dir / snapshot_name
        try:
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except OSError:
            pass

        return checkpoint_dir


class BaseEvaluator(ABC):
    """Evaluates model quality against held-out splits (validation or test)."""

    def __init_subclass__(cls, name: Optional[str] = None, **kwargs: Any) -> None:
        """Automatically registers BaseEvaluator subclasses into the AIMLite registry."""
        super().__init_subclass__(**kwargs)
        from aimlite.registry import register_class

        register_class("evaluator", cls, name=name)

    @abstractmethod
    def evaluate(self, model: Model, dataset: Dataset, **kwargs: Any) -> Dict[str, float]:
        """Evaluates model quality against held-out splits (validation or test).

        Args:
            model: Active Model instance.
            dataset: Partitioned Dataset instance.
            **kwargs: Additional evaluation flags or metrics configurations.

        Returns:
            Structured dictionary of metric names mapped to numerical scores (e.g. accuracy, loss, F1).
        """
        ...


class BaseInference(ABC):
    """Normalizes raw payloads, triggers model inference, and packages structured responses."""

    def __init_subclass__(cls, name: Optional[str] = None, **kwargs: Any) -> None:
        """Automatically registers BaseInference subclasses into the AIMLite registry."""
        super().__init_subclass__(**kwargs)
        from aimlite.registry import register_class

        register_class("inference", cls, name=name)

    @abstractmethod
    def run(self, model: Model, raw_input: Any, **kwargs: Any) -> Any:
        """Normalizes raw payloads, triggers model inference, and packages structured responses for local execution or serving endpoints.

        Args:
            model: Active Model instance.
            raw_input: Unprocessed incoming input payload or request dictionary.
            **kwargs: Additional serving or decoding arguments.

        Returns:
            Formatted prediction output dictionary or list.
        """
        ...

    def get_routes(self) -> Dict[str, Any]:
        """Returns a mapping of HTTP routes ('METHOD /path') to handler callables.

        Developers can override this method to expose custom endpoints,
        such as 'POST /api/v1/predict', 'GET /status', etc.

        Returns:
            Dictionary mapping route strings (e.g. 'POST /predict') to handler functions.
        """
        return {
            "POST /predict": self.run,
            "POST /": self.run,
            "GET /health": self.health,
            "GET /info": self.info,
        }

    def health(self, model: Optional[Model] = None, **kwargs: Any) -> Dict[str, Any]:
        """Health check probe handler."""
        return {
            "status": "healthy",
            "model": getattr(model, "name", "model") if model else "model",
        }

    def info(self, model: Optional[Model] = None, **kwargs: Any) -> Dict[str, Any]:
        """Metadata info endpoint handler."""
        return {
            "model": getattr(model, "name", "model") if model else "model",
            "version": "0.1.0",
        }
