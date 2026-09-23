"""Runs a Python script inside the project context with the project root on sys.path.

This command solves the common 'ModuleNotFoundError: No module named <project>' error
when running benchmark or experiment scripts directly. It is equivalent to:

    PYTHONPATH=. .venv/bin/python experiments/benchmark.py

but works automatically via aimlite.json discovery — no manual path setup needed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from cli.discovery import find_project_root, inject_venv_site_packages
from cli.ui import C, arrow, check, cross, vite_header


def run_benchmark(
    script: Optional[str] = None,
    args: Optional[List[str]] = None,
) -> int:
    """Runs a Python script with the project root injected into sys.path.

    Discovers the project root via aimlite.json, adds it (and the .venv
    site-packages) to PYTHONPATH, then executes the given script using the
    project .venv Python if available, falling back to the current interpreter.

    Args:
        script: Path to the Python script to run (e.g. experiments/benchmark.py).
        args:   Extra arguments forwarded to the script.

    Returns:
        Process exit code.
    """
    extra_args: List[str] = args or []

    print(vite_header("benchmark"))

    # 1. Resolve project root
    root = find_project_root()
    if root is None:
        print(cross("aimlite.json not found. Run from inside an AIMLite project."))
        return 1

    # 2. Decide which script to run
    if not script:
        # Auto-discover: look for scripts in experiments/ then project root
        candidates = sorted((root / "experiments").glob("*.py")) if (root / "experiments").is_dir() else []
        candidates += sorted(root.glob("benchmark*.py"))
        if not candidates:
            print(f"\n  {C.YELLOW}Usage:{C.RESET} aimlite benchmark <script.py> [args...]")
            print(f"  {C.DIM}No script supplied and none auto-discovered in experiments/{C.RESET}\n")
            return 1
        script_path = candidates[0]
        print(f"  {C.DIM}Auto-discovered:{C.RESET} {script_path.relative_to(root)}")
    else:
        script_path = Path(script)
        if not script_path.is_absolute():
            script_path = Path.cwd() / script_path
        if not script_path.is_file():
            print(cross(f"Script not found: {script_path}"))
            return 1

    # 3. Resolve .venv Python (prefer project venv, fall back to current interpreter)
    venv_python = root / ".venv" / "bin" / "python"
    if not venv_python.exists():
        venv_python = root / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        venv_python = Path(sys.executable)

    print(arrow("Root", str(root)))
    print(arrow("Script", str(script_path.relative_to(root) if script_path.is_relative_to(root) else script_path)))
    print(arrow("Python", str(venv_python)))
    print()

    # 4. Build env: extend PYTHONPATH with the project root so `from <project> import ...` works
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    project_root_str = str(root)
    if project_root_str not in existing_pythonpath.split(os.pathsep):
        env["PYTHONPATH"] = project_root_str + (os.pathsep + existing_pythonpath if existing_pythonpath else "")

    # 5. Execute
    cmd = [str(venv_python), str(script_path)] + extra_args
    try:
        result = subprocess.run(cmd, cwd=str(root), env=env)
        if result.returncode == 0:
            print(f"\n  {check(f'Benchmark completed successfully.')}\n")
        else:
            print(f"\n  {cross(f'Script exited with code {result.returncode}.')}\n")
        return result.returncode
    except KeyboardInterrupt:
        print(f"\n  {C.DIM}Benchmark interrupted.{C.RESET}\n")
        return 130
    except Exception as e:
        print(cross(f"Failed to run script: {e}"))
        return 1
