"""Runs the ModelKit zero-path model training cycle."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Optional, Tuple, Type

from cli.discovery import ProjectContext, resolve_project_context
from cli.ui import C, arrow, check, cross, vite_header
from modelkit.data import Dataset
from modelkit.lifecycle import BaseTrainer
from modelkit.models import Model


def _resolve_target_model_and_dataset(
    ctx: ProjectContext,
    target: Optional[str] = None,
) -> Tuple[Optional[Type[Model]], Optional[Type[Dataset]], Optional[str]]:
    """Resolves the specific Model and Dataset classes for training based on target name or conventions.

    Returns:
        (model_cls, dataset_cls, error_message)
    """
    models = ctx.model_classes
    datasets = ctx.dataset_classes

    # Filter out starter default AppModel / AppDataset to distinguish custom implementations
    custom_models = {k: v for k, v in models.items() if k != "AppModel"}
    custom_datasets = {k: v for k, v in datasets.items() if k != "AppDataset"}

    # Case 1: Specific class target provided via CLI argument (e.g. modelkit train UserModel)
    if target:
        target_lower = target.lower()

        # Check if target matches a Model class name
        matched_model: Optional[Type[Model]] = None
        for m_name, m_cls in models.items():
            if m_name.lower() == target_lower or m_name.lower().replace("model", "") == target_lower:
                matched_model = m_cls
                break

        if matched_model is not None:
            # Resolve dataset for this model
            bound_ds = getattr(matched_model, "dataset", None)
            if bound_ds:
                if isinstance(bound_ds, type) and issubclass(bound_ds, Dataset):
                    return matched_model, bound_ds, None
                if isinstance(bound_ds, str) and bound_ds in datasets:
                    return matched_model, datasets[bound_ds], None

            # Name matching between model and dataset (e.g. UserModel <-> UserDataset)
            m_base = matched_model.__name__.lower().replace("model", "")
            for d_name, d_cls in datasets.items():
                if m_base in d_name.lower():
                    return matched_model, d_cls, None

            # Fallback to single available dataset
            if len(datasets) == 1:
                return matched_model, list(datasets.values())[0], None

            avail_ds = ", ".join(datasets.keys()) or "None"
            err = (
                f"Model '{matched_model.__name__}' has no dataset bound.\n"
                f"  Available datasets: {avail_ds}\n"
                f"  Declare 'dataset = <DatasetClass>' in {matched_model.__name__}."
            )
            return None, None, err

        # Check if target matches a Dataset class name (e.g. modelkit train UserDataset)
        matched_ds: Optional[Type[Dataset]] = None
        for d_name, d_cls in datasets.items():
            if d_name.lower() == target_lower or d_name.lower().replace("dataset", "") == target_lower:
                matched_ds = d_cls
                break

        if matched_ds is not None:
            # Check which model binds this dataset
            for m_name, m_cls in models.items():
                b = getattr(m_cls, "dataset", None)
                if b == matched_ds or b == matched_ds.__name__:
                    return m_cls, matched_ds, None
                if m_name.lower().replace("model", "") == matched_ds.__name__.lower().replace("dataset", ""):
                    return m_cls, matched_ds, None

            suggested_model = matched_ds.__name__.replace("Dataset", "") + "Model"
            err = (
                f"No Model class in model.py is bound to dataset '{matched_ds.__name__}'.\n"
                f"  Available models in model.py: {', '.join(models.keys()) or 'None'}\n"
                f"  Please define your model in model.py first:\n"
                f"    class {suggested_model}(Model):\n"
                f"        dataset = {matched_ds.__name__}"
            )
            return None, None, err

        avail_models = ", ".join(models.keys()) or "None"
        avail_ds = ", ".join(datasets.keys()) or "None"
        err = (
            f"No Model or Dataset class named '{target}' found in project.\n"
            f"  Available models in model.py: {avail_models}\n"
            f"  Available datasets in data.py: {avail_ds}"
        )
        return None, None, err

    # Case 2: No specific target provided (modelkit train)
    # Check if developer defined custom datasets, but has not created a custom model class
    if custom_datasets and not custom_models:
        # User defined a dataset in data.py but left model.py with AppModel or empty
        ds_name = list(custom_datasets.keys())[0]
        suggested_model = ds_name.replace("Dataset", "") + "Model"
        err = (
            f"Found custom dataset(s) ({', '.join(custom_datasets.keys())}) in data.py, but no matching Model class in model.py.\n"
            f"  Please define your model in model.py before training:\n"
            f"    class {suggested_model}(Model):\n"
            f"        dataset = {ds_name}\n\n"
            f"  Or specify which model to train: modelkit train <ModelName>"
        )
        return None, None, err

    # Check if multiple custom models exist: require specifying which to train
    if len(custom_models) > 1:
        models_list = "\n".join(f"    - {name}" for name in custom_models.keys())
        err = (
            f"Multiple models found in model.py:\n"
            f"{models_list}\n\n"
            f"  Please specify which model to train:\n"
            f"    modelkit train <ModelName>"
        )
        return None, None, err

    # Exactly one custom model exists
    if len(custom_models) == 1:
        m_cls = list(custom_models.values())[0]
        b = getattr(m_cls, "dataset", None)
        if b:
            if isinstance(b, type) and issubclass(b, Dataset):
                return m_cls, b, None
            if isinstance(b, str) and b in datasets:
                return m_cls, datasets[b], None

        m_base = m_cls.__name__.lower().replace("model", "")
        for d_name, d_cls in datasets.items():
            if m_base in d_name.lower():
                return m_cls, d_cls, None

        if custom_datasets:
            return m_cls, list(custom_datasets.values())[0], None
        if datasets:
            return m_cls, list(datasets.values())[0], None

        return m_cls, None, f"Model '{m_cls.__name__}' has no dataset available in data.py."

    # Starter fallback: AppModel and AppDataset
    if "AppModel" in models and "AppDataset" in datasets:
        return models["AppModel"], datasets["AppDataset"], None

    if ctx.model_cls and ctx.dataset_cls:
        return ctx.model_cls, ctx.dataset_cls, None

    return None, None, "No Model subclass found in model.py."


def run_train(
    target: Optional[str] = None,
    project_root: Optional[Path] = None,
    **kwargs: Any,
) -> int:
    """Executes the training cycle for a specific model class or single project model.

    Args:
        target: Optional specific Model or Dataset class name to train.
        project_root: Optional project root path.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    print(vite_header("training"))

    # 1. Resolve project context
    try:
        ctx = resolve_project_context(start_dir=project_root)
    except Exception as e:
        print(f"{cross(str(e))}\n")
        return 1

    device = ctx.config.resolve_device()

    # 2. Resolve specific model and dataset to train
    model_cls, dataset_cls, err = _resolve_target_model_and_dataset(ctx, target=target)
    if err is not None or model_cls is None or dataset_cls is None:
        print(f"{cross(err or 'Could not resolve model and dataset for training.')}\n")
        return 1

    # 3. Instantiate model
    try:
        model = model_cls(name=model_cls.__name__, config=ctx.config.to_dict())
    except Exception as e:
        print(f"{cross(f'Model instantiation error ({model_cls.__name__}): {e}')}\n")
        return 1

    # 4. Ingest and validate dataset
    try:
        dataset = dataset_cls(name=f"{model_cls.__name__.lower()}_data", config=ctx.config)
        records = dataset.load()
    except Exception as e:
        print(f"{cross(f'Error loading dataset ({dataset_cls.__name__}): {e}')}\n")
        return 1

    if not records or not dataset.validate():
        has_any_file = ctx.data_dir.is_dir() and any(
            f.is_file() and not f.name.startswith(".") for f in ctx.data_dir.iterdir()
        )
        if not has_any_file:
            print(f"{cross(f'Data validation failed for {dataset_cls.__name__}: No data found in data/ directory.')}")
            print(f"  {C.DIM}Checked path: {ctx.data_dir}{C.RESET}")
            print(f"  {C.YELLOW}Please add your data file before running train.{C.RESET}\n")
        else:
            target_fn = getattr(dataset, "filename", None) or "data file"
            print(f"{cross(f'Data validation failed for {dataset_cls.__name__}: No valid records loaded.')}")
            print(f"  {C.DIM}Loaded 0 rows from {target_fn} in {ctx.data_dir}.{C.RESET}")
            print(f"  {C.YELLOW}Ensure your data file contains valid rows and headers.{C.RESET}\n")
        return 1

    # 5. Resolve trainer (custom or default)
    trainer_cls = None
    if ctx.trainer_classes:
        # Check matching trainer (e.g. UserModel -> UserTrainer)
        m_base = model_cls.__name__.lower().replace("model", "")
        for t_name, t_cls in ctx.trainer_classes.items():
            if m_base in t_name.lower():
                trainer_cls = t_cls
                break
        if trainer_cls is None:
            # Pick custom trainer over AppTrainer if available
            custom_t = {k: v for k, v in ctx.trainer_classes.items() if k != "AppTrainer"}
            trainer_cls = list(custom_t.values())[0] if custom_t else list(ctx.trainer_classes.values())[-1]

    if trainer_cls is None:
        trainer_cls = ctx.trainer_cls or BaseTrainer

    try:
        trainer = trainer_cls()
    except Exception as e:
        print(f"{cross(f'Trainer instantiation error ({trainer_cls.__name__}): {e}')}\n")
        return 1

    # 6. Execute training fit loop
    start_time = time.time()
    try:
        metrics = trainer.fit(model, dataset)
    except NotImplementedError:
        print(f"{cross(f'{trainer_cls.__name__}.fit() is not implemented.')}\n")
        return 1
    except Exception as e:
        print(f"{cross(f'Training error in {trainer_cls.__name__}: {e}')}\n")
        return 1

    elapsed = round(time.time() - start_time, 2)

    # 7. Save model weights and experiment snapshot
    ckpt_dir = trainer.checkpoint(
        model=model,
        destination=ctx.models_dir,
        experiments_dir=ctx.experiments_dir,
    )
    weights_file = ctx.models_dir / "model.pkl"
    snapshot_file = ctx.experiments_dir / "experiment_snapshot.json"

    metrics_str = " | ".join(f"{k}: {v}" for k, v in (metrics or {}).items()) if metrics else "status: completed"

    print(arrow("Model", model_cls.__name__))
    print(arrow("Dataset", f"{dataset_cls.__name__} ({len(records):,} records)"))
    print(arrow("Device", device.upper()))
    print(arrow("Weights", str(weights_file if weights_file.is_file() else ckpt_dir)))
    print(arrow("Snapshot", str(snapshot_file)))
    print(arrow("Metrics", metrics_str))
    print(f"\n{check(f'Training finished in {elapsed}s.')}\n")
    return 0
