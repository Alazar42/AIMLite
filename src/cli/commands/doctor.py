"""Inspects runtime environment health, hardware accelerators, and directory permissions."""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Optional

from cli.discovery import find_project_root, get_manifest_path
from cli.ui import C, arrow, check, cross, vite_header
from aimlite.config import BaseConfig


def run_doctor(project_root: Optional[Path] = None) -> int:
    """Executes AIMLite environment and hardware diagnostics.

    Returns:
        Exit code: 0 if healthy, 1 if critical issues found.
    """
    print(vite_header("doctor", extra="(AIMLite Doctor)"))

    py_ver = sys.version.split()[0]
    py_major, py_minor = sys.version_info.major, sys.version_info.minor
    is_supported_py = (py_major == 3 and py_minor >= 14) or (py_major > 3)

    # Hardware accelerator resolution
    root = project_root or find_project_root()
    manifest_path = get_manifest_path(root) if root else None
    config = BaseConfig(config_path=manifest_path)
    device = config.resolve_device()

    # Project manifest check
    if root and manifest_path and manifest_path.is_file():
        name = config.to_dict().get("name", "unnamed")
        manifest_str = f"{manifest_path.name} ({name})"
    else:
        manifest_str = "Not in project root (cwd fallback)"

    print(arrow("Python", f"{py_ver} ({platform.machine()} {sys.platform})"))
    print(arrow("Device", f"{device.upper()} (resolved via BaseConfig)"))
    print(arrow("Manifest", manifest_str))

    # Directory read/write permissions
    check_dirs = {
        "data/": config.data_dir,
        "models/": config.models_dir,
        "experiments/": config.experiments_dir,
        "artifacts/": config.artifacts_dir,
        "checkpoints/": config.checkpoints_dir,
    }

    print(f"\n  {C.BOLD}Directories:{C.RESET}")
    all_ok = True
    cwd = Path.cwd()
    home = Path.home()

    for label, path in check_dirs.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
            test_file = path / ".write_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()

            try:
                rel = f"./{path.relative_to(cwd)}"
            except ValueError:
                try:
                    rel = f"~/{path.relative_to(home)}"
                except ValueError:
                    rel = str(path)

            print(f"  {check(f'{label:<14} ready ({rel})')}")
        except Exception as e:
            print(f"  {cross(f'{label:<14} cannot write ({e})')}")
            all_ok = False

    if all_ok and is_supported_py:
        print(f"\n{check('Environment is ready for zero-path execution.')}\n")
        return 0
    elif all_ok:
        print(f"\n{check('Environment is ready (Notice: Python >= 3.14 recommended).')}\n")
        return 0
    else:
        print(f"\n{cross('Directory permission issues detected.')}\n")
        return 1
