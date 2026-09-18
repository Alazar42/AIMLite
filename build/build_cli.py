"""Automated Build Script for AIMLite CLI Executable.

Bundles the AIMLite library and CLI into a single, standalone executable binary
using Python's native zipapp packaging with compressed bytecode and shebang.
"""

from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile
import zipapp
from pathlib import Path


def build_executable(output_name: str = "aimlite") -> Path:
    """Compiles src/ into a standalone executable file in build/ directory."""
    project_root = Path(__file__).resolve().parent.parent
    src_dir = project_root / "src"
    build_dir = project_root / "build"
    output_bin = build_dir / output_name

    build_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Building AIMLite CLI standalone executable...")
    print(f"    Source directory: {src_dir}")
    print(f"    Target output:    {output_bin}")

    # Stage files into a clean temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Copy src/cli and src/aimlite into staging directory
        for pkg in ["cli", "aimlite"]:
            src_pkg = src_dir / pkg
            dst_pkg = tmp_path / pkg
            if src_pkg.is_dir():
                shutil.copytree(src_pkg, dst_pkg)

        # Create zipapp archive with compressed bytecode and interpreter shebang
        zipapp.create_archive(
            source=tmp_path,
            target=output_bin,
            interpreter="/usr/bin/env python3",
            main="cli.main:main",
            compressed=True,
        )

    # Set executable permissions (chmod +x)
    current_mode = os.stat(output_bin).st_mode
    os.chmod(output_bin, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    file_size_kb = round(output_bin.stat().st_size / 1024, 1)
    print(f"[OK] Successfully built standalone executable: {output_bin} ({file_size_kb} KB)")
    print(f"[*] You can run it directly: {output_bin} --help")
    return output_bin


if __name__ == "__main__":
    out = build_executable()
    sys.exit(0 if out.is_file() else 1)
