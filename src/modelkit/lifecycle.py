"""Lifecycle Pillar Header: modelkit/lifecycle.py

Coordinates the execution pipeline across training workflows,
evaluation runs, checkpointing, and local serving.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    from modelkit.data import Dataset
    from modelkit.models import Model


class BaseTrainer(ABC):
    """Coordinates optimization steps, data consumption, convergence tracking, and checkpointing."""

    def __init_subclass__(cls, name: Optional[str] = None, **kwargs: Any) -> None:
        """Automatically registers BaseTrainer subclasses into the ModelKit registry."""
        super().__init_subclass__(**kwargs)
        from modelkit.registry import register_class

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

        # 1. Trigger model weights persistence into models/ folder
        model.save(checkpoint_dir)

        # 2. Write experiment snapshot metadata into experiments/ folder
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
            "step": step,
            "timestamp": time.time(),
            "config": getattr(model, "config", {}),
        }
        metadata_file = exp_target_dir / "experiment_snapshot.json"
        try:
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        except OSError:
            pass

        return checkpoint_dir


class BaseEvaluator(ABC):
    """Evaluates model quality against held-out splits (validation or test)."""

    def __init_subclass__(cls, name: Optional[str] = None, **kwargs: Any) -> None:
        """Automatically registers BaseEvaluator subclasses into the ModelKit registry."""
        super().__init_subclass__(**kwargs)
        from modelkit.registry import register_class

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
        """Automatically registers BaseInference subclasses into the ModelKit registry."""
        super().__init_subclass__(**kwargs)
        from modelkit.registry import register_class

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
