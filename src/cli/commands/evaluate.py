"""Assesses model performance against held-out validation or test partitions."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from cli.discovery import resolve_project_context
from cli.ui import C, arrow, check, cross, vite_header
from modelkit.data import Dataset
from modelkit.lifecycle import BaseEvaluator
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

    # 2. Locate checkpoint or serialized weights
    checkpoint_file = _discover_checkpoint(ctx)
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


def _discover_checkpoint(ctx) -> Optional[Path]:
    """Finds the most recent model checkpoint or weights file."""
    direct_model = ctx.models_dir / "model.pkl"
    if direct_model.is_file():
        return direct_model

    candidate_dirs = [ctx.checkpoints_dir, ctx.models_dir, ctx.artifacts_dir]
    for cdir in candidate_dirs:
        if cdir.is_dir():
            ckpts = sorted(
                cdir.glob("**/model.pkl"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if ckpts:
                return ckpts[0]
            for ext in ["*.pt", "*.pth", "*.joblib", "*.bin"]:
                generic = sorted(
                    cdir.glob(f"**/{ext}"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True,
                )
                if generic:
                    return generic[0]

    return None
