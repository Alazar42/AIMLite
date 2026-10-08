"""Updates the globally or locally installed AIMLite package and CLI."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, List, Optional, Tuple

from aimlite.cli.ui import C, arrow, check, cross, vite_header


def parse_version(v: Optional[str]) -> Tuple[Any, ...]:
    """Converts a SemVer or PEP 440 version string into a comparable tuple."""
    if not v:
        return ()
    clean = str(v).strip().lstrip("v")
    parts: List[Any] = []
    for token in re.split(r"([0-9]+)", clean):
        if not token or token in (".", "-", "_"):
            continue
        if token.isdigit():
            parts.append(int(token))
        else:
            parts.append(token.lower())
    return tuple(parts)


def get_installed_version(fresh: bool = False) -> Optional[str]:
    """Determines currently installed AIMLite distribution version.
    
    Args:
        fresh: If True, bypasses in-memory caches by querying pip show and invalidating importlib caches.
    """
    if fresh:
        try:
            import importlib
            importlib.invalidate_caches()
        except Exception:
            pass

        try:
            res = subprocess.run(
                [sys.executable, "-m", "pip", "show", "aimlite"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if res.returncode == 0:
                for line in res.stdout.splitlines():
                    if line.startswith("Version:"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass

    # 1. Try standard library importlib.metadata
    try:
        import importlib.metadata

        return importlib.metadata.version("aimlite")
    except Exception:
        pass

    # 2. Try pip show aimlite
    try:
        res = subprocess.run(
            [sys.executable, "-m", "pip", "show", "aimlite"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            for line in res.stdout.splitlines():
                if line.startswith("Version:"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass

    # 3. Fallback to aimlite.__version__ if running within the package
    try:
        import aimlite

        return getattr(aimlite, "__version__", None)
    except Exception:
        pass

    return None


def get_latest_pypi_version(include_pre: bool = False, timeout: float = 6.0) -> Optional[str]:
    """Queries PyPI JSON API for latest published release of aimlite."""
    url = "https://pypi.org/pypi/aimlite/json"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "aimlite-cli-update"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if not include_pre:
                v = data.get("info", {}).get("version")
                if v:
                    return str(v).strip()
            releases = list(data.get("releases", {}).keys())
            if releases:
                return str(sorted(releases, key=parse_version)[-1]).strip()
            return data.get("info", {}).get("version")
    except Exception:
        pass

    # Fallback to pip index versions aimlite
    try:
        res = subprocess.run(
            [sys.executable, "-m", "pip", "index", "versions", "aimlite"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if res.returncode == 0:
            m = re.search(r"aimlite\s*\(([^)]+)\)", res.stdout)
            if m:
                return m.group(1).strip()
    except Exception:
        pass

    return None


def is_virtual_env() -> bool:
    """Checks whether execution is within a Python virtual environment."""
    return (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or bool(os.environ.get("VIRTUAL_ENV"))
    )


def detect_environment_description() -> Tuple[str, bool]:
    """Returns human-readable environment label and virtualenv status."""
    if is_virtual_env():
        venv_name = Path(sys.prefix).name
        return f"Virtual Environment ({venv_name})", True

    if sys.platform.startswith("win"):
        return "Windows System / User Site", False
    elif sys.platform.startswith("darwin"):
        return "macOS System / User Site", False
    else:
        return "Linux System / User Site", False


def resolve_candidate_commands(
    pkg_spec: str,
    user_flag: Optional[bool] = None,
    pre: bool = False,
) -> List[List[str]]:
    """Builds a prioritized list of installer commands suitable for the platform & environment."""
    in_venv = is_virtual_env()
    uv_bin = shutil.which("uv")
    pip_base = [sys.executable, "-m", "pip", "install"]

    commands: List[List[str]] = []

    # Case A: Explicit user flag passed (--user)
    if user_flag is True:
        cmd = [*pip_base, "--upgrade", "--user", pkg_spec]
        if pre:
            cmd.append("--pre")
        commands.append(cmd)

        cmd_break = [*pip_base, "--upgrade", "--user", "--break-system-packages", pkg_spec]
        if pre:
            cmd_break.append("--pre")
        commands.append(cmd_break)
        return commands

    # Case B: Explicit global flag passed (--global)
    if user_flag is False:
        cmd = [*pip_base, "--upgrade", pkg_spec]
        if pre:
            cmd.append("--pre")
        commands.append(cmd)

        cmd_break = [*pip_base, "--upgrade", "--break-system-packages", pkg_spec]
        if pre:
            cmd_break.append("--pre")
        commands.append(cmd_break)
        return commands

    # Case C: Running inside a virtual environment
    if in_venv:
        if uv_bin:
            uv_cmd = [uv_bin, "pip", "install", "--upgrade", pkg_spec]
            if pre:
                uv_cmd.append("--prerelease=allow")
            commands.append(uv_cmd)

        cmd = [*pip_base, "--upgrade", pkg_spec]
        if pre:
            cmd.append("--pre")
        commands.append(cmd)
        return commands

    # Case D: Outside virtual environment (Global / User on Windows, macOS, Linux)
    # 1. On Windows or Unix: Try --user first (safest and avoids root/permission issues)
    user_cmd = [*pip_base, "--upgrade", "--user", pkg_spec]
    if pre:
        user_cmd.append("--pre")
    commands.append(user_cmd)

    # 2. Try --user with --break-system-packages (for Debian 12+, Ubuntu 23.04+, Fedora PEP 668)
    user_break_cmd = [*pip_base, "--upgrade", "--user", "--break-system-packages", pkg_spec]
    if pre:
        user_break_cmd.append("--pre")
    commands.append(user_break_cmd)

    # 3. Try global install without --user (for root/containers or Windows elevated installs)
    global_cmd = [*pip_base, "--upgrade", pkg_spec]
    if pre:
        global_cmd.append("--pre")
    commands.append(global_cmd)

    # 4. Try global with --break-system-packages
    global_break_cmd = [*pip_base, "--upgrade", "--break-system-packages", pkg_spec]
    if pre:
        global_break_cmd.append("--pre")
    commands.append(global_break_cmd)

    return commands


def run_update(
    force: bool = False,
    user: Optional[bool] = None,
    target_version: Optional[str] = None,
    pre: bool = False,
    timeout: float = 6.0,
) -> int:
    """Executes the aimlite update routine."""
    print(vite_header("update"))

    env_label, in_venv = detect_environment_description()
    installed_ver = get_installed_version()

    print(arrow("Target", "AIMLite CLI & Python SDK"))
    print(arrow("Environment", env_label))
    if installed_ver:
        print(arrow("Installed", f"v{installed_ver}"))
    else:
        print(arrow("Installed", f"{C.YELLOW}Not installed in current environment{C.RESET}"))

    # Determine desired version
    latest_ver: Optional[str] = None
    if target_version:
        latest_ver = target_version.lstrip("v")
        pkg_spec = f"aimlite=={latest_ver}"
        print(arrow("Requested", f"v{latest_ver}"))
    else:
        pkg_spec = "aimlite"
        print(f"  {C.DIM}Checking PyPI for latest release...{C.RESET}")
        latest_ver = get_latest_pypi_version(include_pre=pre, timeout=timeout)
        if latest_ver:
            print(arrow("Latest PyPI", f"v{latest_ver}"))
        else:
            print(f"  {C.YELLOW}! Could not query PyPI. Proceeding with pip index search.{C.RESET}")

    # Check if already up to date
    if not force and installed_ver and latest_ver:
        if parse_version(installed_ver) >= parse_version(latest_ver):
            print(f"\n{check(f'AIMLite is already up to date (v{installed_ver}).')}\n")
            return 0

    print()
    if installed_ver and latest_ver and parse_version(installed_ver) < parse_version(latest_ver):
        print(f"  {C.CYAN}Upgrading aimlite:{C.RESET} {C.DIM}v{installed_ver}{C.RESET} {C.BRIGHT_GREEN}❯{C.RESET} {C.BOLD}v{latest_ver}{C.RESET}\n")
    elif not installed_ver:
        print(f"  {C.CYAN}Installing aimlite:{C.RESET} {C.BOLD}v{latest_ver or 'latest'}{C.RESET}\n")
    else:
        print(f"  {C.CYAN}Reinstalling aimlite...{C.RESET}\n")

    candidates = resolve_candidate_commands(pkg_spec=pkg_spec, user_flag=user, pre=pre)

    last_error = ""
    success = False

    for cmd in candidates:
        cmd_display = " ".join(cmd)
        print(f"  {C.DIM}Running:{C.RESET} {cmd_display}")
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                success = True
                break
            else:
                err_text = (res.stderr or res.stdout).strip()
                last_error = err_text.splitlines()[-1] if err_text else f"Exit code {res.returncode}"
        except Exception as e:
            last_error = str(e)

    if not success:
        print(f"\n{cross('Failed to install/upgrade aimlite.')}")
        if last_error:
            print(f"  {C.RED}Error details:{C.RESET} {last_error}")
        print(f"\n  {C.DIM}Try running manually:{C.RESET} {C.CYAN}pip install --upgrade aimlite{C.RESET}\n")
        return 1

    # Verify newly installed version
    new_ver = get_installed_version(fresh=True)
    print()
    if new_ver:
        if installed_ver and installed_ver != new_ver:
            print(f"{check(f'Successfully updated AIMLite from v{installed_ver} to v{new_ver}!')}\n")
        elif installed_ver:
            print(f"{check(f'Successfully reinstalled AIMLite v{new_ver}!')}\n")
        else:
            print(f"{check(f'Successfully installed AIMLite v{new_ver}!')}\n")
    else:
        print(f"{check('AIMLite installation completed successfully.')}\n")

    # Helpful hint for user site bin directory if not on PATH
    if not in_venv:
        _check_and_hint_user_path()

    return 0


def _check_and_hint_user_path() -> None:
    """Prints a friendly note if Python's user script directory is not currently in PATH."""
    try:
        import site

        user_base = getattr(site, "USER_BASE", None)
        if not user_base:
            return

        if sys.platform.startswith("win"):
            scripts_dir = Path(user_base) / "Scripts"
        else:
            scripts_dir = Path(user_base) / "bin"

        if scripts_dir.is_dir():
            path_env = os.environ.get("PATH", "")
            resolved_scripts = str(scripts_dir.resolve())
            if resolved_scripts not in path_env and str(scripts_dir) not in path_env:
                print(f"  {C.DIM}Note: Ensure '{scripts_dir}' is in your PATH to invoke the 'aimlite' command globally.{C.RESET}\n")
    except Exception:
        pass
