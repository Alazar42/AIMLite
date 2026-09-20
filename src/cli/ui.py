"""Vite-inspired minimalistic, ultra-clean Terminal UI engine for AIMLite CLI.

Zero dependencies, zero emojis. Follows modern Vite / Astral styling conventions:
subbrand banners, arrow prompts, clean whitespace, and colored status indicators.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional


class Color:
    """ANSI color codes with automatic NO_COLOR and TTY detection."""

    _ENABLED = sys.stdout.isatty() and "NO_COLOR" not in os.environ

    RESET = "\033[0m" if _ENABLED else ""
    BOLD = "\033[1m" if _ENABLED else ""
    DIM = "\033[2m" if _ENABLED else ""
    ITALIC = "\033[3m" if _ENABLED else ""
    UNDERLINE = "\033[4m" if _ENABLED else ""

    # Foreground
    BLACK = "\033[30m" if _ENABLED else ""
    RED = "\033[31m" if _ENABLED else ""
    GREEN = "\033[32m" if _ENABLED else ""
    YELLOW = "\033[33m" if _ENABLED else ""
    BLUE = "\033[34m" if _ENABLED else ""
    MAGENTA = "\033[35m" if _ENABLED else ""
    CYAN = "\033[36m" if _ENABLED else ""
    WHITE = "\033[37m" if _ENABLED else ""

    # Bright / Light
    BRIGHT_BLACK = "\033[90m" if _ENABLED else ""
    BRIGHT_RED = "\033[91m" if _ENABLED else ""
    BRIGHT_GREEN = "\033[92m" if _ENABLED else ""
    BRIGHT_YELLOW = "\033[93m" if _ENABLED else ""
    BRIGHT_BLUE = "\033[94m" if _ENABLED else ""
    BRIGHT_MAGENTA = "\033[95m" if _ENABLED else ""
    BRIGHT_CYAN = "\033[96m" if _ENABLED else ""
    BRIGHT_WHITE = "\033[97m" if _ENABLED else ""


C = Color


def vite_header(subcommand: str = "", extra: str = "") -> str:
    """Renders a Vite-style brand banner."""
    brand = f"{C.BOLD}{C.BRIGHT_CYAN}AIMLITE{C.RESET} {C.DIM}v1.0.0{C.RESET}"
    if subcommand:
        tag = f" {C.GREEN}{subcommand}{C.RESET}"
    else:
        tag = ""
    if extra:
        details = f"  {C.DIM}{extra}{C.RESET}"
    else:
        details = ""
    return f"\n  {brand}{tag}{details}\n"


def arrow(label: str, value: str, label_width: int = 14) -> str:
    """Renders a Vite-style arrow bullet line: '  ➜  Local:   http://...'"""
    arrow_sym = f"{C.BRIGHT_GREEN}➜{C.RESET}"
    lbl = f"{C.BOLD}{label:<{label_width}}{C.RESET}"
    return f"  {arrow_sym}  {lbl}: {value}"


def check(label: str, value: str = "", label_width: int = 14) -> str:
    """Renders a clean green check bullet."""
    sym = f"{C.GREEN}✔{C.RESET}"
    if value:
        lbl = f"{C.BOLD}{label:<{label_width}}{C.RESET}"
        return f"  {sym}  {lbl}  {C.DIM}{value}{C.RESET}"
    return f"  {sym}  {label}"


def cross(label: str, value: str = "", label_width: int = 14) -> str:
    """Renders a clean red error bullet."""
    sym = f"{C.RED}✖{C.RESET}"
    if value:
        lbl = f"{C.BOLD}{label:<{label_width}}{C.RESET}"
        return f"  {sym}  {lbl}  {C.RED}{value}{C.RESET}"
    return f"  {sym}  {C.RED}{label}{C.RESET}"


def next_steps(commands: List[str], intro: str = "Done. Now run:") -> str:
    """Renders a Vite-style 'Done. Now run:' instruction block."""
    lines = [f"\n  {C.GREEN}Done.{C.RESET} Now run:\n"]
    for cmd in commands:
        lines.append(f"    {C.CYAN}{cmd}{C.RESET}")
    lines.append("")
    return "\n".join(lines)
