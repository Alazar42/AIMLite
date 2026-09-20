"""Scaffolds a new AIMLite project with Vite-style zero-code convention layout."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from cli.ui import C, check, cross, next_steps


def run_init(
    project_name: Optional[str] = None,
    target_dir: Optional[str] = None,
    create_venv: bool = True,
) -> int:
    """Initializes a new AIMLite project directory structure and manifest.

    Args:
        project_name: Project name. If None, prompts interactively.
        target_dir: Optional custom parent target directory.
        create_venv: Whether to automatically create .venv and install aimlite.

    Returns:
        Process exit code (0 for success).
    """
    if not project_name:
        try:
            print(f"\n  {C.BOLD}{C.BRIGHT_CYAN}AIMLite{C.RESET} {C.DIM}create project{C.RESET}")
            prompt_str = f"  {C.GREEN}?{C.RESET} {C.BOLD}Project name:{C.RESET} {C.DIM}»{C.RESET} "
            prompted = input(prompt_str).strip()
            project_name = prompted or "my_ai"
        except (EOFError, KeyboardInterrupt):
            project_name = "my_ai"

    # Resolve target directory
    if target_dir:
        dest_root = Path(target_dir).resolve() / project_name
    elif project_name == ".":
        dest_root = Path.cwd()
        project_name = dest_root.name
    else:
        dest_root = Path.cwd() / project_name

    dest_root.mkdir(parents=True, exist_ok=True)

    # 1. Create convention directories
    convention_dirs = ["data", "models", "experiments", "artifacts", "checkpoints"]
    for d in convention_dirs:
        (dest_root / d).mkdir(parents=True, exist_ok=True)

    # 2. Generate aimlite.json manifest (clean, no task_type or apps)
    manifest_data = {
        "name": project_name,
        "version": "0.1.0",
        "entrypoint": project_name,
        "dependencies": [],
        "config": {
            "device": "auto",
            "batch_size": 32,
        },
        "paths": {
            "data": "data",
            "models": "models",
            "experiments": "experiments",
            "artifacts": "artifacts",
            "checkpoints": "checkpoints",
        },
    }

    manifest_path = dest_root / "aimlite.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # 3. Create starter package directory
    package_dir = dest_root / project_name
    package_dir.mkdir(parents=True, exist_ok=True)

    _create_starter_files(package_dir, project_name)

    print(f"\n  {C.DIM}Scaffolding project in{C.RESET} {dest_root}...")

    # 4. Create .venv and install aimlite into it
    if create_venv:
        _setup_project_venv(dest_root)

    steps = []
    if dest_root != Path.cwd():
        steps.append(f"cd {project_name}")
    steps.extend([
        "aimlite install <pandas scikit-learn ...>",
        "aimlite doctor",
        "aimlite train",
    ])
    print(next_steps(steps))

    return 0


def _setup_project_venv(dest_root: Path) -> bool:
    """Creates an isolated virtual environment (.venv) and installs aimlite into it."""
    venv_dir = dest_root / ".venv"
    uv_bin = shutil.which("uv")

    # 1. Create .venv if not already present
    if not venv_dir.is_dir():
        created = False
        if uv_bin:
            res = subprocess.run([uv_bin, "venv", str(venv_dir)], cwd=str(dest_root), capture_output=True)
            if res.returncode == 0:
                created = True
        if not created:
            try:
                import venv
                venv.create(str(venv_dir), with_pip=True)
                created = True
            except Exception:
                res = subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], cwd=str(dest_root), capture_output=True)
                created = res.returncode == 0
        if not created:
            print(f"  {cross('Could not create virtual environment (.venv) automatically.')}")
            return False
        print(f"  {check('Created virtual environment (.venv).')}")

    # 2. Resolve venv python
    venv_python = venv_dir / "bin" / "python"
    if not venv_python.exists():
        venv_python = venv_dir / "Scripts" / "python.exe"

    if not venv_python.exists():
        return False

    # 3. Install aimlite into .venv
    installed = False

    # Try uv pip install aimlite
    if uv_bin:
        res = subprocess.run([uv_bin, "pip", "install", "--python", str(venv_python), "aimlite"], cwd=str(dest_root), capture_output=True)
        if res.returncode == 0:
            installed = True

    # Fallback to pip install aimlite
    if not installed:
        res = subprocess.run([str(venv_python), "-m", "pip", "install", "aimlite"], cwd=str(dest_root), capture_output=True)
        if res.returncode == 0:
            installed = True

    # Fallback for local development or prior to PyPI indexing: check if local wheel or package exists
    if not installed:
        dist_dir = Path(__file__).resolve().parents[3] / "dist"
        wheels = list(dist_dir.glob("aimlite-*.whl")) if dist_dir.is_dir() else []
        if wheels:
            latest_wheel = sorted(wheels)[-1]
            if uv_bin:
                res = subprocess.run([uv_bin, "pip", "install", "--python", str(venv_python), str(latest_wheel)], cwd=str(dest_root), capture_output=True)
                if res.returncode == 0:
                    installed = True
            if not installed:
                res = subprocess.run([str(venv_python), "-m", "pip", "install", str(latest_wheel)], cwd=str(dest_root), capture_output=True)
                if res.returncode == 0:
                    installed = True

    if installed:
        print(f"  {check('Installed aimlite into .venv.')}")
    else:
        print(f"  {C.DIM}Note: aimlite will be installed into .venv once connected to PyPI.{C.RESET}")

    return True


def _create_starter_files(package_dir: Path, project_name: str) -> None:
    """Generates starter files from aimlite templates with clean boilerplate."""
    try:
        from aimlite import templates

        template_dir = Path(templates.__file__).parent / "app_template"
    except Exception:
        template_dir = Path(__file__).resolve().parent.parent.parent / "aimlite" / "templates" / "app_template"

    if template_dir.is_dir():
        for filename in ["__init__.py", "data.py", "model.py", "trainer.py", "evaluator.py", "inference.py"]:
            src = template_dir / filename
            dst = package_dir / filename
            if src.is_file() and not dst.exists():
                shutil.copy2(src, dst)
    else:
        _write_fallback_files(package_dir)

    config_file = package_dir / "config.py"
    if not config_file.exists():
        config_content = f'''"""AIMLite App Configuration: config.py"""

from pathlib import Path
from aimlite import BaseConfig

BASE_DIR = Path(__file__).resolve().parent.parent


class Config(BaseConfig):
    """Project configuration declaring paths and hardware settings."""

    name: str = "{project_name}"
    data_dir: Path = BASE_DIR / "data"
    models_dir: Path = BASE_DIR / "models"
    experiments_dir: Path = BASE_DIR / "experiments"
    artifacts_dir: Path = BASE_DIR / "artifacts"
    checkpoints_dir: Path = BASE_DIR / "checkpoints"
    device: str = "auto"
    batch_size: int = 32
'''
        with open(config_file, "w", encoding="utf-8") as f:
            f.write(config_content)


def _write_fallback_files(package_dir: Path) -> None:
    """Fallback generator for starter files."""
    files = {
        "__init__.py": '"""AIMLite Package."""\n',
        "data.py": '''"""AIMLite App: data.py"""
from aimlite import Dataset


class AppDataset(Dataset):
    """Application dataset definition."""
    filename = "dataset.csv"
''',
        "model.py": '''"""AIMLite App: model.py"""
from typing import Any
from aimlite import Model


class AppModel(Model):
    """Application model definition."""

    def predict(self, inputs: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "Model prediction logic has not been implemented yet. "
            "Define your forward pass in model.py or train your model with 'aimlite train'."
        )
''',
        "trainer.py": '''"""AIMLite App: trainer.py"""
from typing import Any, Dict
from aimlite import BaseTrainer, Dataset, Model


class AppTrainer(BaseTrainer):
    """Application training orchestrator."""

    def fit(self, model: Model, dataset: Dataset, **kwargs: Any) -> Dict[str, Any]:
        return {"status": "completed"}
''',
        "evaluator.py": '''"""AIMLite App: evaluator.py"""
from typing import Any, Dict
from aimlite import BaseEvaluator, Dataset, Model


class AppEvaluator(BaseEvaluator):
    """Application evaluation benchmark."""

    def evaluate(self, model: Model, dataset: Dataset, **kwargs: Any) -> Dict[str, float]:
        return {"accuracy": 1.0}
''',
        "inference.py": '''"""AIMLite App: inference.py"""
from typing import Any
from aimlite import BaseInference, Model


class AppInference(BaseInference):
    """Application inference handler."""

    def run(self, model: Model, raw_input: Any, **kwargs: Any) -> Any:
        return model.predict(raw_input, **kwargs)
''',
    }
    for fname, content in files.items():
        p = package_dir / fname
        if not p.exists():
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
