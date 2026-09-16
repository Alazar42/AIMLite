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
    project_root: Optional[Path] = None,
    **kwargs,
) -> int:
    """Validates trained model presence, loads custom routes/frontend, and launches server.

    Args:
        target: Optional specific Model class name to serve.
        port: Listening port (default 8000).
        host: Listening host (default 127.0.0.1).
        frontend: Optional custom frontend directory (e.g. dist/, frontend/).
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
    ckpt_file = _discover_checkpoint(ctx)
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
            print(f"    2. Run {C.CYAN}modelkit train{C.RESET} to train your model")
            print(f"    3. Run {C.CYAN}modelkit serve{C.RESET} to launch the inference server\n")
        else:
            print(f"\n{cross('Cannot serve: No trained model checkpoint found in models/.')}")
            print(f"  {C.YELLOW}Dataset exists in data/, but the model has not been trained yet.{C.RESET}")
            print(f"\n  {C.BOLD}Run training first:{C.RESET}")
            print(f"    {C.CYAN}modelkit train{C.RESET}\n")

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
        for folder_name in ["frontend/dist", "dist", "frontend", "static", "web"]:
            cand = ctx.root_dir / folder_name
            if cand.is_dir() and (cand / "index.html").is_file():
                resolved_frontend = cand
                break

    return run_inference_server(
        model=model,
        inference=inference_handler,
        model_name=model_name,
        host=host,
        port=port,
        frontend_dir=resolved_frontend,
        custom_app=custom_app,
        checkpoint_path=ckpt_file,
    )
