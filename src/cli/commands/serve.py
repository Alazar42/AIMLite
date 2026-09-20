"""Launches the developer-customizable HTTP inference & frontend server."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from cli.commands.evaluate import _discover_checkpoint
from cli.discovery import resolve_project_context
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


def run_serve(
    target: Optional[str] = None,
    port: int = 8000,
    host: str = "127.0.0.1",
    frontend: Optional[str] = None,
    checkpoint: Optional[str] = None,
    project_root: Optional[Path] = None,
    **kwargs,
) -> int:
    """Validates trained model presence, loads custom routes/frontend, and launches server.

    Args:
        target: Optional specific Model class name to serve.
        port: Listening port (default 8000).
        host: Listening host (default 127.0.0.1).
        frontend: Optional custom frontend directory (e.g. dist/, frontend/).
        checkpoint: Optional explicit path to past checkpoint to serve.
        project_root: Optional project root path.

    Returns:
        Process exit code (0 for clean run/shutdown, 1 on error).
    """
    try:
        ctx = resolve_project_context(start_dir=project_root, require_manifest=True)
    except Exception as e:
        print(f"\n{cross(str(e))}\n")
        return 1

    # Resolve specific model class if requested
    if target:
        model_cls = None
        for m_name, m_c in ctx.model_classes.items():
            if m_name.lower() == target.lower():
                model_cls = m_c
                break
        if model_cls is None:
            print(f"\n{cross(f'Model class \'{target}\' not found in model.py.')}")
            print(f"  Available models: {', '.join(ctx.model_classes.keys()) or 'None'}\n")
            return 1
        model_name = model_cls.__name__
    else:
        model_cls = ctx.model_cls or FallbackEchoModel
        model_name = model_cls.__name__ if hasattr(model_cls, "__name__") else ctx.target_module

    # Check if developer defined a custom WSGI app or custom serve() function
    custom_app = None
    if ctx.target_module in sys.modules:
        mod = sys.modules[ctx.target_module]
        custom_app = getattr(mod, "app", None) or getattr(mod, "serve", None)

    # 1. Require a trained model checkpoint (unless custom app is explicitly defined)
    if checkpoint:
        explicit_p = Path(checkpoint)
        if not explicit_p.is_absolute():
            explicit_p = ctx.root_dir / explicit_p
        if not explicit_p.exists():
            print(f"\n{cross(f'Specified checkpoint path does not exist: {checkpoint}')}\n")
            return 1
        ckpt_file = explicit_p
    else:
        ckpt_file = _discover_checkpoint(ctx, model_cls=model_cls)
    if ckpt_file is None and custom_app is None:
        # Inspect data directory to see if dataset exists
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

    # 2. Instantiate and load model weights
    try:
        model = model_cls(name=model_name, config=ctx.config.to_dict())
    except Exception:
        model = model_cls(name=model_name)

    if ckpt_file:
        try:
            model.load(ckpt_file)
        except Exception as e:
            print(f"\n{cross(f'Failed to load model checkpoint {ckpt_file}: {e}')}\n")
            return 1

    # 3. Resolve developer's custom BaseInference instance
    inference_handler = ctx.inference_cls() if ctx.inference_cls else None

    # 4. Resolve custom frontend directory (if developer specified or has one)
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
        # Auto-detect common frontend build folders
        for folder_name in ["api_docs/dist", "frontend/dist", "dist", "frontend", "static", "web"]:
            cand = ctx.root_dir / folder_name
            if cand.is_dir() and (cand / "index.html").is_file():
                resolved_frontend = cand
                break

    # 5. Extract real dataset sample if data exists (never hallucinate fake features)
    sample_payload, dataset_info = _extract_real_data_sample(ctx)

    # 6. Extract Voxide API key from project root .env if present
    voxide_key = _extract_voxide_key(ctx.root_dir)

    return run_inference_server(
        model=model,
        inference=inference_handler,
        model_name=model_name,
        host=host,
        port=port,
        frontend_dir=resolved_frontend,
        custom_app=custom_app,
        checkpoint_path=ckpt_file,
        sample_payload=sample_payload,
        dataset_info=dataset_info,
        voxide_api_key=voxide_key,
    )


def _extract_real_data_sample(ctx) -> tuple[Optional[str], dict]:
    """Inspects the project data directory to extract the REAL schema and sample record.

    Never hallucinates or invents fake features. Returns actual records from the dataset.
    """
    import json

    if not ctx.data_dir or not ctx.data_dir.is_dir():
        return None, {"has_data": False, "filename": None, "columns": [], "record_count": 0}

    data_files = [f for f in ctx.data_dir.iterdir() if f.is_file() and not f.name.startswith(".")]
    if not data_files:
        return None, {"has_data": False, "filename": None, "columns": [], "record_count": 0}

    # If dataset_cls declared filename, try that first
    target = data_files[0]
    if ctx.dataset_cls and getattr(ctx.dataset_cls, "filename", None):
        cand = ctx.data_dir / ctx.dataset_cls.filename
        if cand.is_file():
            target = cand
    else:
        for f in data_files:
            if f.name.lower() in ["dataset.csv", "data.csv", "train.csv", "customers.csv"]:
                target = f
                break

    try:
        if target.suffix.lower() == ".csv":
            import csv

            with open(target, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    first = rows[0]
                    cleaned = {}
                    for k, v in first.items():
                        if k is None:
                            continue
                        k_s = k.strip().strip('"').strip("'")
                        val_str = v.strip().strip('"').strip("'") if isinstance(v, str) else v
                        try:
                            if "." in str(val_str):
                                cleaned[k_s] = float(val_str)
                            else:
                                cleaned[k_s] = int(val_str)
                        except (ValueError, TypeError):
                            cleaned[k_s] = val_str
                    payload_str = json.dumps(cleaned, indent=2)
                    info = {
                        "has_data": True,
                        "filename": target.name,
                        "columns": list(cleaned.keys()),
                        "record_count": len(rows),
                    }
                    return payload_str, info
        elif target.suffix.lower() == ".json":
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
                if isinstance(data, list) and data:
                    sample = data[0]
                    rows_count = len(data)
                elif isinstance(data, dict):
                    sample = data
                    rows_count = 1
                else:
                    sample = {"data": data}
                    rows_count = 1
                payload_str = json.dumps(sample, indent=2)
                info = {
                    "has_data": True,
                    "filename": target.name,
                    "columns": list(sample.keys()) if isinstance(sample, dict) else [],
                    "record_count": rows_count,
                }
                return payload_str, info
    except Exception:
        pass

    return None, {"has_data": False, "filename": target.name, "columns": [], "record_count": 0}


def _extract_voxide_key(root_dir: Optional[Path]) -> str:
    """Reads Voxide API key from .env in project root or environment variables."""
    import os

    # 1. Process environment variables first
    for var in [
        "VOXIDE_API_KEY",
        "VOXIDE_PUBLIC_KEY",
        "VOXIDE_KEY",
        "VITE_VOXIDE_PUBLIC_KEY",
        "NEXT_PUBLIC_VOXIDE_KEY",
    ]:
        val = os.environ.get(var)
        if val and val.strip():
            return val.strip()

    # 2. Candidate .env paths: root_dir, root_dir parents up to 4 levels, and cwd
    candidates: list[Path] = []
    if root_dir:
        candidates.append(root_dir / ".env")
        curr = root_dir.resolve()
        for _ in range(4):
            parent = curr.parent
            if parent == curr:
                break
            candidates.append(parent / ".env")
            curr = parent

    candidates.append(Path.cwd().resolve() / ".env")

    # Remove duplicates while preserving order
    seen: set[Path] = set()
    unique_candidates: list[Path] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    for env_path in unique_candidates:
        if env_path.is_file():
            try:
                with open(env_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            if k in [
                                "VOXIDE_API_KEY",
                                "VOXIDE_PUBLIC_KEY",
                                "VOXIDE_KEY",
                                "VITE_VOXIDE_PUBLIC_KEY",
                                "NEXT_PUBLIC_VOXIDE_KEY",
                            ]:
                                cleaned_val = v.strip().strip('"').strip("'")
                                if cleaned_val:
                                    return cleaned_val
            except Exception:
                pass
    return ""

