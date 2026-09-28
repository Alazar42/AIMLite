"""Launches the developer-customizable HTTP inference & frontend server."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from cli.commands.evaluate import _discover_checkpoint
from cli.discovery import _is_framework_class, resolve_project_context
from cli.server import run_inference_server
from cli.ui import C, cross


class FallbackEchoModel:
    """Fallback model structure."""

    def __init__(self, name: str = "model", config=None):
        self.name = name
        self.config = config or {}

    def predict(self, inputs, **kwargs):
        return inputs

    def load(self, path):
        pass


def detect_model_type(model_cls: type, model_instance: Optional[Any] = None) -> str:
    """Detects whether model is 'rag', 'adapter', or 'ml'.

    Returns:
        'rag': KnowledgeModel, RAGModel, or has retrieve/vector_store.
        'adapter': AdapterModel, or has adapter_manager/merge_weights.
        'ml': Standard classical / deep ML classification or regression model.
    """
    for base in getattr(model_cls, "__mro__", []):
        base_name = getattr(base, "__name__", "")
        if base_name in ("KnowledgeModel", "RAGModel"):
            return "rag"
        if base_name in ("AdapterModel", "LoRAModel"):
            return "adapter"

    target = model_instance or model_cls
    if hasattr(target, "vector_store") or hasattr(target, "retrieve") or hasattr(target, "knowledge_dir"):
        return "rag"
    if hasattr(target, "adapter_manager") or hasattr(target, "merge_weights") or hasattr(target, "generate_with_adapter"):
        return "adapter"

    return "ml"


def discover_feature_names(ctx: Any, model_cls: type, model_instance: Optional[Any] = None) -> List[str]:
    """Attempts to extract feature column names for an ML model."""
    # 1. Direct attribute on model
    for attr in ("feature_names", "features", "columns", "feature_columns"):
        val = getattr(model_instance, attr, None) or getattr(model_cls, attr, None)
        if val and isinstance(val, (list, tuple)):
            return [str(x) for x in val]

    # 2. Check dataset class bound to model or context
    ds_cls = getattr(model_cls, "dataset", None) or getattr(ctx, "dataset_cls", None)
    if ds_cls is not None:
        for attr in ("FEATURE_COLUMNS", "feature_columns", "features", "columns"):
            val = getattr(ds_cls, attr, None)
            if val and isinstance(val, (list, tuple)):
                return [str(x) for x in val]

    # 3. Check data module global variables
    if hasattr(ctx, "target_module") and ctx.target_module in sys.modules:
        mod = sys.modules[ctx.target_module]
        data_mod = getattr(mod, "data", None)
        if data_mod:
            for attr in ("FEATURE_COLUMNS", "feature_columns", "features"):
                val = getattr(data_mod, attr, None)
                if val and isinstance(val, (list, tuple)):
                    return [str(x) for x in val]

    # 4. Check CSV files in data/
    if hasattr(ctx, "data_dir") and ctx.data_dir.is_dir():
        for csv_path in sorted(ctx.data_dir.glob("*.csv")):
            try:
                with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header:
                        target_names = {"target", "label", "churn", "class", "y", "output", "id", "row_id"}
                        features = [c.strip() for c in header if c.strip().lower() not in target_names]
                        if features:
                            return features
            except Exception:
                pass

    return []


def run_serve(
    target: Optional[str] = None,
    port: int = 8000,
    host: str = "127.0.0.1",
    frontend: Optional[str] = None,
    checkpoint: Optional[str] = None,
    api_only: bool = False,
    project_root: Optional[Path] = None,
    **kwargs: Any,
) -> int:
    """Validates trained model presence, loads custom routes/frontend, and launches server.

    Args:
        target: Optional specific Model class name to serve.
        port: Listening port (default 8000).
        host: Listening host (default 127.0.0.1).
        frontend: Optional custom frontend directory (e.g. dist/, frontend/).
        checkpoint: Optional explicit path to past checkpoint to serve.
        api_only: If True, serves headless API endpoints without web interfaces.
        project_root: Optional project root path.

    Returns:
        Process exit code (0 for clean run/shutdown, 1 on error).
    """
    try:
        ctx = resolve_project_context(start_dir=project_root, require_manifest=True)
    except Exception as e:
        print(f"\n{cross(str(e))}\n")
        return 1

    # Check if developer defined a custom WSGI app or custom serve() function
    custom_app = None
    if ctx.target_module in sys.modules:
        mod = sys.modules[ctx.target_module]
        custom_app = getattr(mod, "app", None) or getattr(mod, "serve", None)

    # 1. Resolve candidate model classes
    candidate_classes: Dict[str, type] = {}
    if target:
        target_lower = target.lower()
        matched_cls = None
        matched_name = None
        for m_name, m_c in ctx.model_classes.items():
            if m_name.lower() == target_lower or m_name.lower().replace("model", "") == target_lower:
                matched_cls = m_c
                matched_name = m_name
                break
        if matched_cls is None:
            print(f"\n{cross(f'Model class \'{target}\' not found in model.py.')}")
            avail = [k for k, v in ctx.model_classes.items() if not _is_framework_class(v)]
            print(f"  Available models: {', '.join(avail) or 'None'}\n")
            return 1
        candidate_classes[matched_name] = matched_cls
    else:
        # Discover all registered user models from model.py
        candidate_classes = {
            k: v for k, v in ctx.model_classes.items()
            if not _is_framework_class(v)
        }
        if not candidate_classes and ctx.model_cls and not _is_framework_class(ctx.model_cls):
            candidate_classes[ctx.model_cls.__name__] = ctx.model_cls

    # 2. Check trained status and instantiate ready models
    ready_models: Dict[str, Dict[str, Any]] = {}
    for m_name, m_cls in candidate_classes.items():
        ckpt_file = None
        if checkpoint and (target or len(candidate_classes) == 1):
            explicit_p = Path(checkpoint)
            if not explicit_p.is_absolute():
                explicit_p = ctx.root_dir / explicit_p
            if not explicit_p.exists():
                print(f"\n{cross(f'Specified checkpoint path does not exist: {checkpoint}')}\n")
                return 1
            ckpt_file = explicit_p
        else:
            ckpt_file = _discover_checkpoint(ctx, model_cls=m_cls)

        try:
            instance = m_cls(name=m_name, config=ctx.config.to_dict())
        except Exception:
            try:
                instance = m_cls(name=m_name)
            except Exception:
                instance = m_cls()

        is_ready = False
        if ckpt_file:
            try:
                instance.load(ckpt_file)
                is_ready = True
            except Exception as e:
                if target:
                    print(f"\n{cross(f'Failed to load model checkpoint {ckpt_file}: {e}')}\n")
                    return 1
        else:
            # Check if model has weights or is self-contained / trained
            if getattr(instance, "is_trained", False) or getattr(instance, "estimator", None) is not None:
                is_ready = True
            elif detect_model_type(m_cls, instance) == "rag" and (ctx.artifacts_dir / "rag_index.json").is_file():
                is_ready = True
            elif target is not None:
                # If target was explicitly requested, allow it if it has an estimator or weights
                is_ready = getattr(instance, "weights", None) is not None

        if is_ready or (target and custom_app is not None):
            m_type = detect_model_type(m_cls, instance)
            features = discover_feature_names(ctx, m_cls, instance) if m_type == "ml" else []
            doc = m_cls.__doc__.strip().split("\n")[0] if m_cls.__doc__ else f"{m_name} ({m_type.upper()})"
            inference_handler = ctx.inference_cls() if ctx.inference_cls else None
            ready_models[m_name] = {
                "class": m_cls,
                "instance": instance,
                "type": m_type,
                "checkpoint": str(ckpt_file) if ckpt_file else None,
                "description": doc,
                "features": features,
                "inference": inference_handler,
            }

    # 3. If no models are ready and no custom app, display helpful guide
    if not ready_models and custom_app is None:
        has_data = False
        if ctx.data_dir.is_dir():
            data_files = [f for f in ctx.data_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
            has_data = len(data_files) > 0

        if not has_data:
            print(f"\n{cross('Cannot serve: No trained model checkpoint found.')}")
            print(f"  {C.DIM}No model weights exist in models/, and no dataset was found in data/.{C.RESET}")
            print(f"\n  {C.BOLD}To get started:{C.RESET}")
            print(f"    1. Place your dataset in {C.CYAN}{ctx.data_dir.name}/{C.RESET} (e.g. {ctx.data_dir.name}/dataset.csv)")
            print(f"    2. Run {C.CYAN}aimlite train{C.RESET} to train your model")
            print(f"    3. Run {C.CYAN}aimlite serve{C.RESET} to launch the inference server\n")
        else:
            print(f"\n{cross('Cannot serve: No trained model checkpoint found in models/.')}")
            print(f"  {C.YELLOW}Dataset exists in data/, but the model has not been trained yet.{C.RESET}")
            print(f"\n  {C.BOLD}Run training first:{C.RESET}")
            print(f"    {C.CYAN}aimlite train{C.RESET}\n")

        return 1

    # 4. Resolve primary model & inference handler
    active_model_name = list(ready_models.keys())[0] if ready_models else (target or "model")
    primary_model_info = ready_models.get(active_model_name)
    primary_instance = primary_model_info["instance"] if primary_model_info else FallbackEchoModel()
    primary_ckpt = primary_model_info.get("checkpoint") if primary_model_info else None
    inference_handler = ctx.inference_cls() if ctx.inference_cls else None

    # 5. Resolve custom developer frontend directory (if specified or detected)
    resolved_frontend: Optional[Path] = None
    if frontend:
        cand = Path(frontend).resolve()
        if cand.is_dir():
            resolved_frontend = cand
        elif (ctx.root_dir / frontend).is_dir():
            resolved_frontend = ctx.root_dir / frontend
        else:
            print(f"  {C.YELLOW}! Specified frontend directory '{frontend}' not found.{C.RESET}")
    else:
        for folder_name in ["frontend/dist", "dist", "frontend", "static", "web"]:
            cand = ctx.root_dir / folder_name
            if cand.is_dir() and (cand / "index.html").is_file():
                resolved_frontend = cand
                break

    # 6. Check for developer custom templates directory (<project_root>/templates/)
    template_dir: Optional[Path] = None
    proj_templates = ctx.root_dir / "templates"
    if proj_templates.is_dir():
        template_dir = proj_templates

    return run_inference_server(
        model=primary_instance,
        inference=inference_handler,
        model_name=active_model_name,
        host=host,
        port=port,
        frontend_dir=resolved_frontend,
        template_dir=template_dir,
        custom_app=custom_app,
        checkpoint_path=Path(primary_ckpt) if primary_ckpt else None,
        registered_models=ready_models,
        active_model_name=active_model_name,
        api_only=api_only,
    )
