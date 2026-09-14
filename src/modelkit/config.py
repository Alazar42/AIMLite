"""BaseConfig interface and contract specification.

Manages project configuration parsing, directory path resolution,
and hardware device discovery.
"""

from __future__ import annotations

import copy
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class BaseConfig:
    """Manages project configuration parsing, directory path resolution, and hardware device discovery."""

    def __init_subclass__(cls, name: Optional[str] = None, **kwargs: Any) -> None:
        """Automatically registers BaseConfig subclasses into the ModelKit registry."""
        super().__init_subclass__(**kwargs)
        from modelkit.registry import register_class

        register_class("config", cls, name=name)

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Initializes project settings by parsing mlkit.json or fallback defaults.

        Args:
            config_path: Optional file path to the project configuration manifest.
        """
        self.config_path: Optional[Path] = Path(config_path) if config_path is not None else None
        self._config: Dict[str, Any] = self._load_config()

    @property
    def root_dir(self) -> Path:
        """Returns the project root directory."""
        if self.config_path is not None:
            if self.config_path.is_dir():
                return self.config_path
            return self.config_path.parent
        return Path.cwd()

    @property
    def data_dir(self) -> Path:
        """Returns the resolved data directory."""
        data_path = self._config.get("paths", {}).get("data", "data")
        return self.root_dir / data_path

    @property
    def models_dir(self) -> Path:
        """Returns the resolved models directory."""
        models_path = self._config.get("paths", {}).get("models", "models")
        return self.root_dir / models_path

    @property
    def experiments_dir(self) -> Path:
        """Returns the resolved experiments directory."""
        exp_path = self._config.get("paths", {}).get("experiments", "experiments")
        return self.root_dir / exp_path

    @property
    def artifacts_dir(self) -> Path:
        """Returns the resolved artifacts directory."""
        art_path = self._config.get("paths", {}).get("artifacts", "artifacts")
        return self.root_dir / art_path

    @property
    def checkpoints_dir(self) -> Path:
        """Returns the resolved checkpoints directory."""
        ckpt_path = self._config.get("paths", {}).get("checkpoints", "checkpoints")
        return self.root_dir / ckpt_path

    @property
    def apps(self) -> List[str]:
        """Returns the list of installed application names declared in mlkit.json."""
        return list(self._config.get("apps", []))

    @property
    def is_multi_app(self) -> bool:
        """Returns True if the project defines one or more modular apps."""
        return len(self.apps) > 0

    def get_app_dir(self, app_name: str) -> Path:
        """Returns the directory path for a specific app within the project."""
        return self.root_dir / app_name

    def get_app_config(self, app_name: str) -> Dict[str, Any]:
        """Returns configuration dictionary for a specific app if defined, or empty dict."""
        app_configs = self._config.get("app_configs", {})
        return dict(app_configs.get(app_name, {}))

    def _load_config(self) -> Dict[str, Any]:
        """Loads configuration from config_path or searches for mlkit.json in parent directories."""
        target_path: Optional[Path] = self.config_path

        if target_path is None:
            # Search upwards for mlkit.json starting from current working directory
            current_dir = Path.cwd()
            for directory in [current_dir, *current_dir.parents]:
                candidate = directory / "mlkit.json"
                if candidate.is_file():
                    target_path = candidate
                    self.config_path = candidate
                    break

        if target_path is not None and target_path.is_file():
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
            except (json.JSONDecodeError, OSError):
                pass

        # Fallback default configuration
        return {
            "name": "unnamed_project",
            "version": "0.1.0",
            "task_type": "generic",
            "entrypoint": "my_ai",
            "hardware": {
                "device": "auto",
            },
            "paths": {
                "data": "data",
                "models": "models",
                "experiments": "experiments",
                "artifacts": "artifacts",
                "checkpoints": "checkpoints",
            },
        }

    def resolve_device(self) -> str:
        """Inspects available local hardware and resolves target devices.

        Returns:
            Standardized string identifier for the execution device ("cuda", "mps", or "cpu").
        """
        configured_device = self._config.get("hardware", {}).get("device", "auto")
        if configured_device and configured_device != "auto":
            return str(configured_device)

        # Check PyTorch device availability if installed
        if "torch" in sys.modules or self._can_import("torch"):
            try:
                import torch

                if torch.cuda.is_available():
                    return "cuda"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
            except Exception:
                pass

        # Check Apple Silicon MPS capability without PyTorch
        if sys.platform == "darwin" and platform.machine() == "arm64":
            return "mps"

        # Check CUDA availability without PyTorch via environment or nvidia-smi
        if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"] != "-1":
            return "cuda"

        return "cpu"

    @staticmethod
    def _can_import(module_name: str) -> bool:
        """Checks if a module is importable without raising an exception."""
        import importlib.util

        try:
            return importlib.util.find_spec(module_name) is not None
        except (ModuleNotFoundError, ValueError):
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the active configuration into a dictionary for experiment tracking and manifest exports.

        Returns:
            Key-value mapping of current configuration parameters.
        """
        return copy.deepcopy(self._config)
