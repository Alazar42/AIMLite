"""Vite-inspired modern Terminal UI engine for AIMLite CLI.

Delivers a state-of-the-art developer experience modeled directly after Vite / Astral:
subbrand banners, interactive arrow-key selection (via questionary & prompt_toolkit),
smooth spinners (via rich), clean whitespace, and colored status indicators.
"""

from __future__ import annotations

import contextlib
import os
import sys
from typing import Any, Generator, List, Optional, Tuple, Union

try:
    import questionary
    from questionary import Choice, Style

    HAS_QUESTIONARY = True
except ImportError:
    questionary = None  # type: ignore
    Choice = None  # type: ignore
    Style = None  # type: ignore
    HAS_QUESTIONARY = False

try:
    from rich.console import Console

    HAS_RICH = True
    rich_console = Console(highlight=False)
except ImportError:
    rich_console = None  # type: ignore
    HAS_RICH = False


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

# Custom Vite theme for questionary prompts
if HAS_QUESTIONARY:
    VITE_STYLE = Style([
        ("qmark", "fg:#00d8ff bold"),        # Cyan ? mark
        ("question", "bold"),                 # Question title
        ("answer", "fg:#00d8ff bold"),        # Selected answer text
        ("pointer", "fg:#00d8ff bold"),       # Arrow indicator (❯)
        ("highlighted", "fg:#00d8ff bold"),   # Currently highlighted item
        ("selected", "fg:#00d8ff"),
        ("separator", "fg:#6c7086"),
        ("instruction", "fg:#6c7086 italic"), # Muted instructions
        ("text", ""),
        ("disabled", "fg:#6c7086 italic"),
    ])
else:
    VITE_STYLE = None


def vite_header(subcommand: str = "", extra: str = "") -> str:
    """Renders a Vite-style brand banner."""
    brand = f"{C.BOLD}{C.BRIGHT_CYAN}AIMLITE{C.RESET} {C.DIM}v1.0.6{C.RESET}"
    tag = f" {C.GREEN}{subcommand}{C.RESET}" if subcommand else ""
    details = f"  {C.DIM}{extra}{C.RESET}" if extra else ""
    return f"\n  {brand}{tag}{details}\n"


try:
    import pyfiglet
    HAS_PYFIGLET = True
except ImportError:
    pyfiglet = None  # type: ignore
    HAS_PYFIGLET = False


def big_header(subcommand: str = "", tagline: str = "The Django for AI & Machine Learning") -> str:
    """Renders a Vite-inspired large title banner using pyfiglet (slant font)."""
    lines: List[str] = []
    lines.append("")

    if HAS_PYFIGLET:
        raw = pyfiglet.figlet_format("AIMLITE", font="basic").rstrip()
        for line in raw.splitlines():
            lines.append(f"  {C.BOLD}{C.BRIGHT_CYAN}{line}{C.RESET}")
    else:
        # Plain fallback if pyfiglet is not installed
        lines.append(f"  {C.BOLD}{C.BRIGHT_CYAN}AIMLITE{C.RESET}")

    lines.append("")
    lines.append(f"  {C.DIM}v1.0.6{C.RESET}  {C.BRIGHT_CYAN}❯{C.RESET}  {C.BOLD}{tagline}{C.RESET}")
    if subcommand:
        lines.append(f"  {C.GREEN}{subcommand}{C.RESET}")
    lines.append("")
    return "\n".join(lines)


def prompt_package_search(
    question: str = "Add extra packages",
    hint: str = "(space-separated, or press Enter to skip)",
    is_tty: Optional[bool] = None,
) -> List[str]:
    """Interactive prompt allowing user to type/search package names to add.

    Returns:
        List of package name strings entered by the user.
    """
    if is_tty is None:
        is_tty = sys.stdin.isatty()
    if not is_tty:
        return []

    if HAS_QUESTIONARY:
        try:
            answer = questionary.text(
                f"{question}  {hint}",
                default="",
                style=VITE_STYLE,
                qmark="+",
            ).ask()
            if answer is None:
                return []
            return [p.strip() for p in answer.split() if p.strip()]
        except (KeyboardInterrupt, EOFError):
            return []

    # Fallback
    prompt_str = (
        f"  {C.GREEN}+{C.RESET} {C.BOLD}{question}{C.RESET} "
        f"{C.DIM}{hint}{C.RESET} » "
    )
    try:
        raw = input(prompt_str).strip()
        return [p.strip() for p in raw.split() if p.strip()]
    except (KeyboardInterrupt, EOFError):
        return []


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


def prompt_text(question: str, default: str = "", is_tty: Optional[bool] = None) -> str:
    """Prompts for text input with Vite styling and keyboard interrupt handling."""
    if is_tty is None:
        is_tty = sys.stdin.isatty()
    if not is_tty:
        return default

    if HAS_QUESTIONARY:
        try:
            answer = questionary.text(
                f"{question}",
                default=default,
                style=VITE_STYLE,
                qmark="?",
            ).ask()
            if answer is None:  # User cancelled via Ctrl+C / EOF
                print(f"\n  {C.DIM}Operation cancelled.{C.RESET}\n")
                sys.exit(130)
            return answer.strip() or default
        except (KeyboardInterrupt, EOFError):
            print(f"\n  {C.DIM}Operation cancelled.{C.RESET}\n")
            sys.exit(130)

    # Basic fallback
    prompt_str = f"  {C.GREEN}?{C.RESET} {C.BOLD}{question}{C.RESET} {C.DIM}(default: {default}){C.RESET} » "
    try:
        val = input(prompt_str).strip()
        return val or default
    except (KeyboardInterrupt, EOFError):
        print(f"\n  {C.DIM}Operation cancelled.{C.RESET}\n")
        sys.exit(130)


def prompt_select(
    question: str,
    choices: List[Union[Tuple[str, str], Any, str]],
    default: Optional[str] = None,
    is_tty: Optional[bool] = None,
) -> str:
    """Prompts the user to select from an interactive arrow-navigated list with Vite styling."""
    raw_choices = []
    default_val = None

    for item in choices:
        if isinstance(item, tuple):
            val, label = item
            raw_choices.append((val, label))
        elif hasattr(item, "value") and hasattr(item, "title"):
            raw_choices.append((item.value, item.title))
        else:
            raw_choices.append((str(item), str(item)))

    if not raw_choices:
        return ""

    if default:
        default_val = default
    else:
        default_val = raw_choices[0][0]

    if is_tty is None:
        is_tty = sys.stdin.isatty()
    if not is_tty:
        return default_val

    if HAS_QUESTIONARY:
        try:
            q_choices = [
                questionary.Choice(title=label, value=val)
                for val, label in raw_choices
            ]
            answer = questionary.select(
                question,
                choices=q_choices,
                default=default_val,
                style=VITE_STYLE,
                pointer="❯",
                instruction="- Use arrow-keys. Return to submit.",
                qmark="?",
            ).ask()
            if answer is None:
                print(f"\n  {C.DIM}Operation cancelled.{C.RESET}\n")
                sys.exit(130)
            return answer
        except (KeyboardInterrupt, EOFError):
            print(f"\n  {C.DIM}Operation cancelled.{C.RESET}\n")
            sys.exit(130)

    # Clean fallback for non-questionary environments
    print(f"\n  {C.GREEN}?{C.RESET} {C.BOLD}{question}{C.RESET}")
    for idx, (val, label) in enumerate(raw_choices, start=1):
        print(f"    {C.CYAN}{idx}){C.RESET} {label}")
    prompt_str = f"  {C.DIM}Select option [1-{len(raw_choices)}]:{C.RESET} » "
    try:
        raw = input(prompt_str).strip()
        if raw and raw.isdigit():
            num = int(raw)
            if 1 <= num <= len(raw_choices):
                return raw_choices[num - 1][0]
    except (KeyboardInterrupt, EOFError):
        print(f"\n  {C.DIM}Operation cancelled.{C.RESET}\n")
        sys.exit(130)
    return default_val


def prompt_confirm(question: str, default: bool = False, is_tty: Optional[bool] = None) -> bool:
    """Prompts for a yes/no confirmation with Vite styling."""
    if is_tty is None:
        is_tty = sys.stdin.isatty()
    if not is_tty:
        return default

    if HAS_QUESTIONARY:
        try:
            answer = questionary.confirm(
                question,
                default=default,
                style=VITE_STYLE,
                qmark="?",
            ).ask()
            if answer is None:
                print(f"\n  {C.DIM}Operation cancelled.{C.RESET}\n")
                sys.exit(130)
            return bool(answer)
        except (KeyboardInterrupt, EOFError):
            print(f"\n  {C.DIM}Operation cancelled.{C.RESET}\n")
            sys.exit(130)

    yn = "Y/n" if default else "y/N"
    prompt_str = f"  {C.GREEN}?{C.RESET} {C.BOLD}{question}{C.RESET} {C.DIM}({yn}):{C.RESET} » "
    try:
        raw = input(prompt_str).strip().lower()
        if not raw:
            return default
        return raw in ("y", "yes", "true", "1")
    except (KeyboardInterrupt, EOFError):
        print(f"\n  {C.DIM}Operation cancelled.{C.RESET}\n")
        sys.exit(130)


@contextlib.contextmanager
def status_spinner(message: str) -> Generator[None, None, None]:
    """Displays a modern spinner during long-running tasks, with graceful fallback."""
    if HAS_RICH and rich_console and sys.stdout.isatty():
        with rich_console.status(f"[cyan]{message}[/cyan]", spinner="dots"):
            yield
    else:
        print(f"  {C.DIM}{message}{C.RESET}")
        yield
