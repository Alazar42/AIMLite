"""Zero-Path Discovery & Execution Engine for AIMLite.

Discovers aimlite.json, resolves project conventions,
imports component modules, and extracts registered classes from aimlite.registry.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from aimlite.config import BaseConfig
from aimlite.data import Dataset
from aimlite.lifecycle import BaseEvaluator, BaseInference, BaseTrainer
from aimlite.models import Model
from aimlite.registry import get_all


@dataclass
class ProjectContext:
    """Encapsulates the discovered AIMLite project environment."""

    root_dir: Path
    manifest: Dict[str, Any]
    config: BaseConfig
    target_module: str
    entrypoint_dir: Path
    data_dir: Path
    models_dir: Path
    experiments_dir: Path
    artifacts_dir: Path
    checkpoints_dir: Path
    dataset_cls: Optional[Type[Dataset]] = None
    dataset_classes: Dict[str, Type[Dataset]] = field(default_factory=dict)
    model_cls: Optional[Type[Model]] = None
    model_classes: Dict[str, Type[Model]] = field(default_factory=dict)
    trainer_cls: Optional[Type[BaseTrainer]] = None
    trainer_classes: Dict[str, Type[BaseTrainer]] = field(default_factory=dict)
    evaluator_cls: Optional[Type[BaseEvaluator]] = None
    inference_cls: Optional[Type[BaseInference]] = None


FRAMEWORK_MODULES = ("aimlite.", "aimlite")
FRAMEWORK_CLASS_NAMES = {
    "Model",
    "AdapterModel",
    "RAGModel",
    "KnowledgeModel",
    "Dataset",
    "BaseTrainer",
    "AdapterTrainer",
    "RAGTrainer",
    "BaseEvaluator",
    "BaseInference",
}


def _is_framework_class(cls: Type[Any]) -> bool:
    """Returns True if cls is an internal AIMLite framework base class."""
    mod = getattr(cls, "__module__", "")
    if any(mod == m or mod.startswith(m) for m in FRAMEWORK_MODULES):
        return True
    if getattr(cls, "__name__", "") in FRAMEWORK_CLASS_NAMES:
        return True
    return False


def find_project_root(start_dir: Optional[Path] = None) -> Optional[Path]:
    """Traverses up from start_dir to find the directory containing aimlite.json."""
    current = (start_dir or Path.cwd()).resolve()
    for parent in [current, *current.parents]:
        if (parent / "aimlite.json").is_file():
            return parent
    return None


def get_manifest_path(project_root: Path) -> Path:
    """Returns the manifest path (aimlite.json)."""
    return project_root / "aimlite.json"


def inject_venv_site_packages(project_root: Path) -> None:
    """Injects the project .venv site-packages into sys.path so installed libraries are available.

    This enables the standalone aimlite CLI to use libraries (pandas, scikit-learn, torch, etc.)
    installed via `aimlite install` in the project's virtual environment without needing
    to activate the venv manually.

    Works on Linux/macOS (.venv/lib/pythonX.Y/site-packages) and Windows (.venv/Lib/site-packages).
    """
    venv_dir = project_root / ".venv"
    if not venv_dir.is_dir():
        return

    # Discover all site-packages directories inside .venv
    site_pkgs_candidates: List[Path] = []

    # Unix / macOS: .venv/lib/python3.X/site-packages
    lib_dir = venv_dir / "lib"
    if lib_dir.is_dir():
        for py_dir in sorted(lib_dir.iterdir()):
            if py_dir.is_dir() and py_dir.name.startswith("python"):
                sp = py_dir / "site-packages"
                if sp.is_dir():
                    site_pkgs_candidates.append(sp)

    # Windows: .venv/Lib/site-packages
    win_lib = venv_dir / "Lib" / "site-packages"
    if win_lib.is_dir():
        site_pkgs_candidates.append(win_lib)

    for sp in site_pkgs_candidates:
        sp_str = str(sp)
        if sp_str not in sys.path:
            sys.path.insert(1, sp_str)


def load_project_manifest(project_root: Path) -> Dict[str, Any]:
    """Loads and parses aimlite.json from the project root."""
    manifest_path = get_manifest_path(project_root)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing project manifest at: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid manifest format: expected a JSON object at {manifest_path}")
    return data


def resolve_project_context(
    start_dir: Optional[Path] = None,
    require_manifest: bool = True,
    **kwargs: Any,
) -> ProjectContext:
    """Discovers project root, loads manifest, and dynamically imports project code.

    Args:
        start_dir: Optional starting directory for discovery.
        require_manifest: If True, raises RuntimeError if aimlite.json is not found.

    Returns:
        ProjectContext with loaded classes and convention paths.
    """
    root_dir = find_project_root(start_dir)
    if root_dir is None:
        if require_manifest:
            raise RuntimeError(
                "Zero-Path error: 'aimlite.json' not found in current or parent directories.\n"
                "Please run this command from within an initialized AIMLite project, "
                "or run 'aimlite init <project_name>' to create one."
            )
        root_dir = (start_dir or Path.cwd()).resolve()
        manifest: Dict[str, Any] = {}
    else:
        manifest = load_project_manifest(root_dir)

    # Prepend project root to sys.path so modules can be imported directly
    root_str = str(root_dir)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    # Inject project .venv site-packages so installed libraries are available
    inject_venv_site_packages(root_dir)

    # Initialize BaseConfig bound to root
    manifest_file = get_manifest_path(root_dir)
    config_file = manifest_file if manifest_file.is_file() else None
    config = BaseConfig(config_path=config_file)

    # Determine target module
    target_module = manifest.get("entrypoint") or manifest.get("name") or "my_ai"
    entrypoint_dir = root_dir / target_module
    if not entrypoint_dir.is_dir() and (root_dir / "data.py").is_file():
        # Flat project structure where modules reside in root
        entrypoint_dir = root_dir

    # Import target modules to trigger automatic registry discovery
    _import_project_modules(root_dir, target_module, entrypoint_dir)

    # Register bare module aliases in sys.modules so both `from data import X`
    # and `from telecom_churn.data import X` resolve correctly everywhere.
    _register_bare_module_aliases(target_module, entrypoint_dir)

    # Extract registered subclasses scoped to the active project (excluding framework base classes)
    project_datasets = {}
    for name, cls in get_all("dataset").items():
        if issubclass(cls, Dataset) and cls is not Dataset and not _is_framework_class(cls):
            mod = getattr(cls, "__module__", "")
            if mod.startswith(f"{target_module}.") or mod == target_module or mod in ("data", "__main__"):
                project_datasets[cls.__name__] = cls

    if not project_datasets and not manifest:
        project_datasets = {
            cls.__name__: cls for name, cls in get_all("dataset").items()
            if issubclass(cls, Dataset) and cls is not Dataset and not _is_framework_class(cls)
        }

    project_models = {}
    for name, cls in get_all("model").items():
        if issubclass(cls, Model) and cls is not Model and not _is_framework_class(cls):
            mod = getattr(cls, "__module__", "")
            if mod.startswith(f"{target_module}.") or mod == target_module or mod in ("model", "__main__"):
                project_models[cls.__name__] = cls

    if not project_models and not manifest:
        project_models = {
            cls.__name__: cls for name, cls in get_all("model").items()
            if issubclass(cls, Model) and cls is not Model and not _is_framework_class(cls)
        }

    project_trainers = {}
    for name, cls in get_all("trainer").items():
        if issubclass(cls, BaseTrainer) and cls is not BaseTrainer and not _is_framework_class(cls):
            mod = getattr(cls, "__module__", "")
            if mod.startswith(f"{target_module}.") or mod == target_module or mod in ("trainer", "__main__"):
                project_trainers[cls.__name__] = cls

    if not project_trainers and not manifest:
        project_trainers = {
            cls.__name__: cls for name, cls in get_all("trainer").items()
            if issubclass(cls, BaseTrainer) and cls is not BaseTrainer and not _is_framework_class(cls)
        }

    all_datasets = project_datasets
    all_models = project_models
    all_trainers = project_trainers

    # Resolve default model: prioritize custom user model over default AppModel
    custom_models = {k: v for k, v in all_models.items() if k != "AppModel"}
    if custom_models:
        model_cls = list(custom_models.values())[0]
    elif all_models:
        model_cls = list(all_models.values())[-1]
    else:
        model_cls = None

    # Resolve default dataset:
    dataset_cls = None
    if model_cls and getattr(model_cls, "dataset", None) is not None:
        binding = getattr(model_cls, "dataset")
        if isinstance(binding, type) and issubclass(binding, Dataset):
            dataset_cls = binding
        elif isinstance(binding, str) and binding in all_datasets:
            dataset_cls = all_datasets[binding]

    if dataset_cls is None and all_datasets:
        # Check custom dataset matching model name
        if model_cls:
            model_base = model_cls.__name__.lower().replace("model", "")
            for d_name, d_cls in all_datasets.items():
                if model_base in d_name.lower():
                    dataset_cls = d_cls
                    break

        if dataset_cls is None:
            for ds_name, cls in all_datasets.items():
                fn = getattr(cls, "filename", None)
                if fn and (config.data_dir / fn).is_file():
                    dataset_cls = cls
                    break

    if dataset_cls is None and all_datasets:
        custom_ds = {k: v for k, v in all_datasets.items() if k != "AppDataset"}
        dataset_cls = list(custom_ds.values())[0] if custom_ds else list(all_datasets.values())[-1]

    # Resolve trainer:
    trainer_cls = list(all_trainers.values())[-1] if all_trainers else None
    evaluator_cls = _resolve_registered_class("evaluator", BaseEvaluator)
    inference_cls = _resolve_registered_class("inference", BaseInference)

    return ProjectContext(
        root_dir=root_dir,
        manifest=manifest,
        config=config,
        target_module=target_module,
        entrypoint_dir=entrypoint_dir,
        data_dir=config.data_dir,
        models_dir=config.models_dir,
        experiments_dir=config.experiments_dir,
        artifacts_dir=config.artifacts_dir,
        checkpoints_dir=config.checkpoints_dir,
        dataset_cls=dataset_cls,
        dataset_classes=all_datasets,
        model_cls=model_cls,
        model_classes=all_models,
        trainer_cls=trainer_cls,
        trainer_classes=all_trainers,
        evaluator_cls=evaluator_cls,
        inference_cls=inference_cls,
    )


def _import_project_modules(root_dir: Path, module_name: str, entrypoint_dir: Path) -> None:
    """Attempts to import standard project component modules."""
    module_candidates = [
        "config",
        "data",
        "model",
        "trainer",
        "evaluator",
        "inference",
        "serve",
    ]

    for mod in module_candidates:
        full_name = f"{module_name}.{mod}" if entrypoint_dir != root_dir else mod
        try:
            importlib.import_module(full_name)
        except ModuleNotFoundError as exc:
            target_file = entrypoint_dir / f"{mod}.py"
            if target_file.is_file():
                _import_file_directly(target_file, full_name, parent_package=module_name)
        except Exception:
            pass


def _import_file_directly(
    file_path: Path,
    module_name: str,
    parent_package: Optional[str] = None,
) -> None:
    """Directly loads a Python file as a module into sys.modules.

    Registers the module under `module_name` (e.g. `telecom_churn.data`) so that
    subsequent imports like `from telecom_churn.data import X` resolve correctly.
    """
    spec = importlib.util.spec_from_file_location(
        module_name, file_path, submodule_search_locations=[]
    )
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        # Set __package__ so relative imports inside the file can resolve
        if parent_package:
            module.__package__ = parent_package
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:
            # Keep module registered so partial imports don't error on re-import,
            # but surface the error so it can be debugged if needed.
            import os
            if os.environ.get("AIMLITE_DEBUG"):
                import traceback
                traceback.print_exc()


def _register_bare_module_aliases(module_name: str, entrypoint_dir: Path) -> None:
    """Creates bare-name aliases in sys.modules so user code can use either style:

        from data import FEATURE_COLUMNS          # bare import (direct alias)
        from telecom_churn.data import FEATURE_COLUMNS  # full qualified import

    Without this, bare `from data import X` can accidentally resolve to an
    unrelated system 'data' module (e.g. matplotlib.dates shim) instead of
    the project's data.py.
    """
    bare_names = ["config", "data", "model", "trainer", "evaluator", "inference", "serve"]
    for mod in bare_names:
        full_name = f"{module_name}.{mod}"
        # Only alias if the full-qualified module was successfully loaded
        if full_name in sys.modules and mod not in sys.modules:
            sys.modules[mod] = sys.modules[full_name]


def _resolve_registered_class(category: str, base_cls: Type[Any]) -> Optional[Type[Any]]:
    """Retrieves the most recently registered custom subclass, ignoring base classes."""
    classes = get_all(category)
    for cls in reversed(list(classes.values())):
        if issubclass(cls, base_cls) and cls is not base_cls and not _is_framework_class(cls):
            return cls
    return None
