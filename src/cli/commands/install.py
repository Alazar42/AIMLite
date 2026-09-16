"""Installs dependencies into the project virtual environment using uv or pip."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from cli.discovery import find_project_root
from cli.ui import C, arrow, check, cross, vite_header


def run_install(
    packages: List[str],
    upgrade: bool = False,
    project_root: Optional[Path] = None,
) -> int:
    """Safely installs packages into the project .venv using uv or pip.

    Args:
        packages: List of package names/specs to install.
        upgrade: Whether to upgrade packages.
        project_root: Optional project root path override.

    Returns:
        Process exit code.
    """
    if not packages:
        print(f"\n  {C.YELLOW}Usage:{C.RESET} modelkit install <package_name ...> [--upgrade]\n")
        return 1

    print(vite_header("install"))

    root = project_root or find_project_root() or Path.cwd()
    venv_dir = root / ".venv"

    # 1. Ensure .venv exists, create if missing
    if not venv_dir.is_dir():
        print(f"  {C.DIM}Creating .venv in{C.RESET} {root}...")
        uv_bin = shutil.which("uv")
        if uv_bin:
            res = subprocess.run([uv_bin, "venv", str(venv_dir)], cwd=str(root))
            if res.returncode != 0:
                res = subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], cwd=str(root))
                if res.returncode != 0:
                    print(f"{cross('Failed to create virtual environment.')}\n")
                    return res.returncode
        else:
            res = subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], cwd=str(root))
            if res.returncode != 0:
                print(f"{cross('Failed to create virtual environment.')}\n")
                return res.returncode

    # 2. Resolve python executable in .venv
    venv_python = venv_dir / "bin" / "python"
    if not venv_python.exists():
        venv_python = venv_dir / "Scripts" / "python.exe"

    # 3. Prefer uv pip install if available
    uv_bin = shutil.which("uv")
    backend_name = "uv pip install" if uv_bin else "pip install"

    print(arrow("Environment", ".venv"))
    print(arrow("Packages", ", ".join(packages)))
    print(arrow("Backend", backend_name))
    print()

    if uv_bin:
        cmd = [uv_bin, "pip", "install", "--python", str(venv_python)]
        if upgrade:
            cmd.append("--upgrade")
        cmd.extend(packages)
        res = subprocess.run(cmd, cwd=str(root))
        if res.returncode == 0:
            print(f"\n{check(f'Installed {len(packages)} packages.')}\n")
            return 0

    # 4. Fallback to venv python -m pip
    pip_cmd = [str(venv_python), "-m", "pip", "install"]
    if upgrade:
        cmd.append("--upgrade")
    pip_cmd.extend(packages)
    res = subprocess.run(pip_cmd, cwd=str(root))
    if res.returncode == 0:
        print(f"\n{check(f'Installed {len(packages)} packages.')}\n")
    else:
        print(f"\n{cross('Package installation encountered errors.')}\n")
    return res.returncode
