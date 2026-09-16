"""Assesses model performance against held-out validation or test partitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from cli.discovery import resolve_project_context
from cli.ui import C, arrow, check, cross, vite_header
from modelkit.data import Dataset
from modelkit.lifecycle import BaseEvaluator, _model_weights_filename
from modelkit.models import Model


def run_evaluate(
    target: Optional[str] = None,
    project_root: Optional[Path] = None,
    **kwargs: Any,
) -> int:
    """Discovers latest checkpoint and runs model evaluation for target model class.

    Args:
        target: Optional specific Model class name to evaluate.
        project_root: Optional project root path.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    print(vite_header("evaluate"))

    try:
        ctx = resolve_project_context(start_dir=project_root)
    except Exception as e:
        print(f"{cross(str(e))}\n")
        return 1

    # 1. Resolve specific model class
    model_cls = None
    if target:
        for m_name, m_c in ctx.model_classes.items():
            if m_name.lower() == target.lower():
                model_cls = m_c
                break
        if model_cls is None:
            print(f"{cross(f'Model class \'{target}\' not found in model.py.')}")
            print(f"  Available models: {', '.join(ctx.model_classes.keys()) or 'None'}\n")
            return 1
    else:
        custom_models = {k: v for k, v in ctx.model_classes.items() if k != "AppModel"}
        if len(custom_models) > 1:
            print(f"{cross('Multiple models found in model.py.')}")
            print(f"  Please specify which model to evaluate: modelkit evaluate <ModelName>\n")
            return 1
        elif len(custom_models) == 1:
            model_cls = list(custom_models.values())[0]
        else:
            model_cls = ctx.model_classes.get("AppModel", ctx.model_cls)

    if model_cls is None:
        print(f"{cross('No Model subclass found in model.py')}\n")
        return 1

    # 2. Locate checkpoint or serialized weights for this specific model class
    checkpoint_file = _discover_checkpoint(ctx, model_cls=model_cls)
    if checkpoint_file is None:
        print(f"{cross('No model checkpoint found in models/ or artifacts/.')}")
        print(f"  {C.YELLOW}Run 'modelkit train' first to generate model weights.{C.RESET}\n")
        return 1

    evaluator_cls = ctx.evaluator_cls
    if evaluator_cls is None:
        print(f"{cross(f'No BaseEvaluator subclass found in {ctx.target_module}')}\n")
        return 1

    # 3. Restore model weights
    model = model_cls(name=ctx.target_module, config=ctx.config.to_dict())
    try:
        model.load(checkpoint_file)
    except Exception as e:
        print(f"  {C.YELLOW}! Notice: Model.load returned: {e}. Proceeding with initialized model.{C.RESET}")

    # 4. Ingest and partition dataset
    dataset_cls = ctx.dataset_cls or Dataset
    dataset = dataset_cls(name=f"{ctx.target_module}_data", config=ctx.config)
    dataset.load()

    # 5. Invoke evaluation
    evaluator = evaluator_cls()
    try:
        metrics = evaluator.evaluate(model, dataset)
    except Exception as e:
        print(f"{cross(f'Error during evaluation: {e}')}\n")
        return 1

    print(arrow("Target", ctx.target_module))
    print(arrow("Checkpoint", str(checkpoint_file)))
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
            print(arrow(k, val_str))
    else:
        print(arrow("Result", str(metrics)))

    print(f"\n{check('Evaluation completed.')}\n")
    return 0


def _discover_checkpoint(
    ctx,
    model_cls: Optional[Any] = None,
) -> Optional[Path]:
    """Finds the most recent model checkpoint or weights file.

    Search order:
    1. Per-model named file (e.g. churn_classifier.pkl) in models/
    2. Any .pkl file in models/ (newest first)
    3. Common ML checkpoint extensions (.pt, .pth, .joblib, .bin) in models/
    4. Repeat search in checkpoints/ and artifacts/
    """
    candidate_dirs = [ctx.models_dir, ctx.checkpoints_dir, ctx.artifacts_dir]

    # 1. Look for per-model named checkpoint first (highest priority)
    if model_cls is not None:
        try:
            tmp_instance = object.__new__(model_cls)
            tmp_instance.name = "_"
            weights_name = _model_weights_filename(tmp_instance)
        except Exception:
            # Fallback: derive name from class name directly
            import re
            snake = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", model_cls.__name__).lower()
            weights_name = f"{snake}.pkl"

        for cdir in candidate_dirs:
            if cdir.is_dir():
                candidate = cdir / weights_name
                if candidate.is_file():
                    return candidate
                # Also search in subdirs (e.g. checkpoint-step-N/)
                matches = sorted(
                    cdir.glob(f"**/{weights_name}"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if matches:
                    return matches[0]

    # 2. Any .pkl file (newest first)
    for cdir in candidate_dirs:
        if cdir.is_dir():
            pkls = sorted(
                cdir.glob("**/*.pkl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if pkls:
                return pkls[0]

    # 3. Other ML checkpoint extensions
    for cdir in candidate_dirs:
        if cdir.is_dir():
            for ext in ["*.pt", "*.pth", "*.joblib", "*.bin"]:
                matches = sorted(
                    cdir.glob(f"**/{ext}"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if matches:
                    return matches[0]

    return None
