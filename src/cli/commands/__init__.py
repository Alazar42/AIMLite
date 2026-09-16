"""ModelKit CLI command modules."""

from cli.commands.data import run_data_validate
from cli.commands.doctor import run_doctor
from cli.commands.evaluate import run_evaluate
from cli.commands.init import run_init
from cli.commands.install import run_install
from cli.commands.serve import run_serve
from cli.commands.train import run_train


def run_headers() -> int:
    """Generates/refreshes ModelKit project header files for IDE integration.

    Generates:
    - ``py.typed``: PEP 561 marker enabling mypy/pyright/pylance type checking.
    - ``.modelkit/workspace.json``: Workspace config for ModelKit IDE plugins.
    - ``.vscode/settings.json``: VS Code Python extension integration.
    - ``pyrightconfig.json``: Pyright/Pylance configuration.
    """
    from pathlib import Path
    from cli.commands.init import _generate_project_headers
    from cli.discovery import find_project_root, load_project_manifest
    from cli.ui import C, check, cross, vite_header

    print(vite_header("headers"))

    root_dir = find_project_root()
    if root_dir is None:
        print(f"{cross('modelkit.json not found. Run from within a ModelKit project, or run modelkit init first.')}\n")
        return 1

    try:
        manifest = load_project_manifest(root_dir)
    except Exception as e:
        print(f"{cross(str(e))}\n")
        return 1

    project_name = manifest.get("entrypoint") or manifest.get("name") or root_dir.name
    package_dir = root_dir / project_name
    if not package_dir.is_dir():
        package_dir = root_dir

    try:
        _generate_project_headers(root_dir, package_dir, project_name)
    except Exception as e:
        print(f"{cross(f'Error generating headers: {e}')}\n")
        return 1

    print(f"  {C.GREEN}✔{C.RESET}  {C.BOLD}ModelKit project headers generated/updated:{C.RESET}")
    print(f"    {C.DIM}•{C.RESET} {package_dir.name}/py.typed {C.DIM}(PEP 561 type checking marker){C.RESET}")
    print(f"    {C.DIM}•{C.RESET} .modelkit/workspace.json {C.DIM}(ModelKit workspace config){C.RESET}")
    print(f"    {C.DIM}•{C.RESET} .vscode/settings.json {C.DIM}(VS Code Python integration){C.RESET}")
    print(f"    {C.DIM}•{C.RESET} pyrightconfig.json {C.DIM}(Pyright/Pylance path config){C.RESET}")
    print()
    print(f"  {C.DIM}IDE tip: Open the project root in VS Code and select the .venv interpreter{C.RESET}")
    print(f"  {C.DIM}         Ctrl+Shift+P → 'Python: Select Interpreter' → ./.venv/bin/python{C.RESET}\n")
    return 0


__all__ = [
    "run_init",
    "run_install",
    "run_data_validate",
    "run_train",
    "run_evaluate",
    "run_serve",
    "run_doctor",
    "run_headers",
]
