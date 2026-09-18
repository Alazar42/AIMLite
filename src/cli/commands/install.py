"""Installs dependencies into the project virtual environment and tracks them in aimlite.json."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from cli.discovery import find_project_root, get_manifest_path
from cli.ui import C, arrow, check, cross, vite_header


def run_install(
    packages: List[str],
    upgrade: bool = False,
    project_root: Optional[Path] = None,
) -> int:
    """Safely installs packages into the project .venv and updates aimlite.json.

    Args:
        packages: List of package names/specs to install. If empty, installs from aimlite.json.
        upgrade: Whether to upgrade packages.
        project_root: Optional project root path override.

    Returns:
        Process exit code.
    """
    root = project_root or find_project_root() or Path.cwd()
    manifest_path = get_manifest_path(root)

    # If no packages specified, attempt to install existing dependencies from manifest
    if not packages:
        if manifest_path.is_file():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                manifest_deps = manifest.get("dependencies", [])
                if isinstance(manifest_deps, list) and manifest_deps:
                    packages = manifest_deps
                elif isinstance(manifest_deps, dict) and manifest_deps:
                    packages = list(manifest_deps.keys())
            except Exception:
                pass

        if not packages:
            print(f"\n  {C.YELLOW}Usage:{C.RESET} aimlite install <package_name ...> [--upgrade]\n")
            print(f"  {C.DIM}No dependencies found in {manifest_path.name}. Specify packages to install.{C.RESET}\n")
            return 1

    print(vite_header("install"))

    venv_dir = root / ".venv"

    # 1. Ensure .venv exists, create if missing (essential for modern Python and PEP 668 managed environments)
    if not venv_dir.is_dir():
        print(f"  {C.DIM}Creating virtual environment (.venv) in{C.RESET} {root}...")
        uv_bin = shutil.which("uv")
        created = False
        if uv_bin:
            res = subprocess.run([uv_bin, "venv", str(venv_dir)], cwd=str(root))
            if res.returncode == 0:
                created = True

        if not created:
            res = subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], cwd=str(root))
            if res.returncode != 0:
                print(f"{cross('Failed to create virtual environment (.venv).')}\n")
                return res.returncode

    # 2. Resolve python and pip executable in .venv across operating systems (Unix vs Windows)
    venv_python = venv_dir / "bin" / "python"
    if not venv_python.exists():
        venv_python = venv_dir / "Scripts" / "python.exe"

    if not venv_python.exists():
        print(f"{cross(f'Python executable not found in virtual environment: {venv_dir}')}\n")
        return 1

    # 3. Prefer uv pip install if available for speed and determinism
    uv_bin = shutil.which("uv")
    backend_name = "uv pip install" if uv_bin else "pip install"

    print(arrow("Environment", ".venv"))
    print(arrow("Packages", ", ".join(packages)))
    print(arrow("Backend", backend_name))
    print()

    installed_successfully = False

    if uv_bin:
        cmd = [uv_bin, "pip", "install", "--python", str(venv_python)]
        if upgrade:
            cmd.append("--upgrade")
        cmd.extend(packages)
        res = subprocess.run(cmd, cwd=str(root))
        if res.returncode == 0:
            installed_successfully = True
        else:
            print(f"\n  {C.DIM}Retrying with fallback pip backend...{C.RESET}")

    if not installed_successfully:
        pip_cmd = [str(venv_python), "-m", "pip", "install"]
        if upgrade:
            pip_cmd.append("--upgrade")
        pip_cmd.extend(packages)
        res = subprocess.run(pip_cmd, cwd=str(root))
        if res.returncode == 0:
            installed_successfully = True

    if not installed_successfully:
        print(f"\n{cross('Package installation encountered errors.')}\n")
        return 1

    # 4. Update manifest (aimlite.json) dependencies
    manifest_file = get_manifest_path(root)
    if not manifest_file.is_file():
        default_data = {
            "name": root.name,
            "version": "0.1.0",
            "entrypoint": root.name,
            "dependencies": [],
            "config": {
                "device": "auto",
                "batch_size": 32,
            },
        }
        try:
            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(default_data, f, indent=2)
        except Exception:
            pass

    try:
        if manifest_file.is_file():
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)

            current_deps = manifest_data.get("dependencies", [])
            if isinstance(current_deps, list):
                dep_set = set(current_deps)
                for p in packages:
                    clean_pkg = p.strip()
                    if clean_pkg and not clean_pkg.startswith("-"):
                        dep_set.add(clean_pkg)
                manifest_data["dependencies"] = sorted(list(dep_set))
            elif isinstance(current_deps, dict):
                for p in packages:
                    clean_pkg = p.strip()
                    if clean_pkg and not clean_pkg.startswith("-"):
                        current_deps[clean_pkg] = "latest"
                manifest_data["dependencies"] = current_deps

            with open(manifest_file, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2)

            print(f"  {check(f'Updated {manifest_file.name} with dependencies.')}")
    except Exception as e:
        print(f"  {C.DIM}Note: Could not update {manifest_file.name}: {e}{C.RESET}")

    print(f"\n{check(f'Installed {len(packages)} package(s) into .venv.')}\n")
    return 0
