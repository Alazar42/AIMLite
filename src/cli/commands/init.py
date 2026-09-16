"""Scaffolds a new ModelKit project with Vite-style zero-code convention layout."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

from cli.ui import C, next_steps


def run_init(
    project_name: Optional[str] = None,
    target_dir: Optional[str] = None,
) -> int:
    """Initializes a new ModelKit project directory structure and manifest.

    Args:
        project_name: Project name. If None, prompts interactively.
        target_dir: Optional custom parent target directory.

    Returns:
        Process exit code (0 for success).
    """
    if not project_name:
        try:
            print(f"\n  {C.BOLD}{C.BRIGHT_CYAN}ModelKit{C.RESET} {C.DIM}create project{C.RESET}")
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

    # 2. Generate mlkit.json manifest (clean, no task_type or apps)
    manifest_data = {
        "name": project_name,
        "version": "0.1.0",
        "entrypoint": project_name,
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

    manifest_path = dest_root / "mlkit.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # 3. Create starter package directory
    package_dir = dest_root / project_name
    package_dir.mkdir(parents=True, exist_ok=True)

    _create_starter_files(package_dir, project_name)

    print(f"\n  {C.DIM}Scaffolding project in{C.RESET} {dest_root}...")
    steps = []
    if dest_root != Path.cwd():
        steps.append(f"cd {project_name}")
    steps.extend([
        "modelkit doctor",
        "modelkit train",
    ])
    print(next_steps(steps))

    return 0


def _create_starter_files(package_dir: Path, project_name: str) -> None:
    """Generates starter files from modelkit templates with clean boilerplate."""
    try:
        from modelkit import templates

        template_dir = Path(templates.__file__).parent / "app_template"
    except Exception:
        template_dir = Path(__file__).resolve().parent.parent.parent / "modelkit" / "templates" / "app_template"

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
        config_content = f'''"""ModelKit App Configuration: config.py"""

from pathlib import Path
from modelkit import BaseConfig

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
        "__init__.py": '"""ModelKit Package."""\n',
        "data.py": '''"""ModelKit App: data.py"""
from modelkit import Dataset


class AppDataset(Dataset):
    """Application dataset definition."""
    pass
''',
        "model.py": '''"""ModelKit App: model.py"""
from typing import Any
from modelkit import Model


class AppModel(Model):
    """Application model definition."""

    def predict(self, inputs: Any, **kwargs: Any) -> Any:
        return inputs
''',
        "trainer.py": '''"""ModelKit App: trainer.py"""
from typing import Any, Dict
from modelkit import BaseTrainer, Dataset, Model


class AppTrainer(BaseTrainer):
    """Application training orchestrator."""

    def fit(self, model: Model, dataset: Dataset, **kwargs: Any) -> Dict[str, Any]:
        return {"status": "completed"}
''',
        "evaluator.py": '''"""ModelKit App: evaluator.py"""
from typing import Any, Dict
from modelkit import BaseEvaluator, Dataset, Model


class AppEvaluator(BaseEvaluator):
    """Application evaluation benchmark."""

    def evaluate(self, model: Model, dataset: Dataset, **kwargs: Any) -> Dict[str, float]:
        return {"accuracy": 1.0}
''',
        "inference.py": '''"""ModelKit App: inference.py"""
from typing import Any
from modelkit import BaseInference, Model


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
