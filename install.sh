#!/usr/bin/env bash
# ==============================================================================
# AIMLite Universal Installer
# The Django for AI & Machine Learning — zero-configuration, zero-path execution.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/Alazar42/AIMLite/main/install.sh | bash
#   ./install.sh [options]
#
# Options:
#   --version <version>    Install a specific release version (e.g. 2.1.1)
#   --user                 Force installation into Python user site directory
#   --global               Force system-wide installation (may require sudo/admin)
#   --editable, -e         Install local repository in editable mode (-e .)
#   --force                Force reinstall / upgrade
#   -h, --help             Display this help message
# ==============================================================================

set -eo pipefail

# Colors and styling
if [ -t 1 ]; then
    RESET="\033[0m"
    BOLD="\033[1m"
    DIM="\033[2m"
    CYAN="\033[36m"
    BRIGHT_CYAN="\033[96m"
    GREEN="\033[32m"
    BRIGHT_GREEN="\033[92m"
    YELLOW="\033[33m"
    RED="\033[31m"
else
    RESET=""
    BOLD=""
    DIM=""
    CYAN=""
    BRIGHT_CYAN=""
    GREEN=""
    BRIGHT_GREEN=""
    YELLOW=""
    RED=""
fi

# Print Vite-style brand banner
print_banner() {
    echo ""
    echo -e "  ${BOLD}${BRIGHT_CYAN}AIMLITE${RESET} ${DIM}Installer${RESET}"
    echo -e "  ${DIM}The Django for AI & Machine Learning${RESET}"
    echo ""
}

print_help() {
    print_banner
    echo -e "  ${BOLD}Usage:${RESET}"
    echo -e "    $ curl -fsSL https://raw.githubusercontent.com/Alazar42/AIMLite/main/install.sh | bash"
    echo -e "    $ ./install.sh [options]\n"
    echo -e "  ${BOLD}Options:${RESET}"
    echo -e "    ${GREEN}--version <ver>${RESET}    Install a specific release version (e.g. 2.1.1)"
    echo -e "    ${GREEN}--user${RESET}             Force installation into user site directory (~/.local or %APPDATA%)"
    echo -e "    ${GREEN}--global${RESET}           Force system-wide installation"
    echo -e "    ${GREEN}--editable, -e${RESET}     Install local checkout in editable development mode"
    echo -e "    ${GREEN}--force${RESET}            Force reinstallation"
    echo -e "    ${GREEN}-h, --help${RESET}         Display this help message\n"
}

# Parse command line options
TARGET_VERSION=""
MODE_USER=""
EDITABLE=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)
            TARGET_VERSION="$2"
            shift 2
            ;;
        --user)
            MODE_USER=true
            shift
            ;;
        --global)
            MODE_USER=false
            shift
            ;;
        --editable|-e)
            EDITABLE=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        -h|--help)
            print_help
            exit 0
            ;;
        *)
            echo -e "  ${RED}✖ Unknown option: $1${RESET}"
            print_help
            exit 1
            ;;
    esac
done

print_banner

# 1. Discover Python 3
PYTHON_BIN=""
CANDIDATES=("${PYTHON:-}" "python3" "python")

for cand in "${CANDIDATES[@]}"; do
    if [ -n "$cand" ] && command -v "$cand" >/dev/null 2>&1; then
        # Check if Python is >= 3.10
        if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
            PYTHON_BIN="$cand"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo -e "  ${RED}✖ Error: Python 3.10 or newer is required.${RESET}"
    echo -e "  ${DIM}Please install Python 3.10+ (recommended: 3.10–3.12) and ensure it is available in your PATH.${RESET}\n"
    exit 1
fi

PY_VERSION=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
echo -e "  ${BRIGHT_GREEN}➜${RESET}  ${BOLD}Python${RESET}       : $PYTHON_BIN ${DIM}(v$PY_VERSION)${RESET}"

# 2. Detect virtual environment
IS_VENV=false
if [ -n "${VIRTUAL_ENV:-}" ]; then
    IS_VENV=true
elif "$PYTHON_BIN" -c 'import sys; sys.exit(0 if (hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)) else 1)' 2>/dev/null; then
    IS_VENV=true
fi

# Detect Operating System
OS_NAME="$(uname -s 2>/dev/null || echo "Unknown")"
case "$OS_NAME" in
    Darwin)  OS_DESC="macOS" ;;
    Linux)   OS_DESC="Linux" ;;
    MINGW*|MSYS*|CYGWIN*) OS_DESC="Windows" ;;
    *)       OS_DESC="$OS_NAME" ;;
esac

if [ "$IS_VENV" = true ]; then
    VENV_NAME="$(basename "${VIRTUAL_ENV:-$("$PYTHON_BIN" -c 'import sys; print(sys.prefix)')}")"
    echo -e "  ${BRIGHT_GREEN}➜${RESET}  ${BOLD}Environment${RESET}  : Virtual Environment ${DIM}($VENV_NAME)${RESET}"
else
    echo -e "  ${BRIGHT_GREEN}➜${RESET}  ${BOLD}Environment${RESET}  : $OS_DESC System / User site"
fi

# 3. Determine installation target
INSTALL_SPEC="aimlite"
if [ "$EDITABLE" = true ]; then
    if [ -f "pyproject.toml" ] && [ -d "src/aimlite" ]; then
        INSTALL_SPEC="-e ."
        echo -e "  ${BRIGHT_GREEN}➜${RESET}  ${BOLD}Target${RESET}       : Local source checkout ${DIM}(editable mode)${RESET}"
    else
        echo -e "  ${RED}✖ Error: --editable specified but current directory does not contain AIMLite source.${RESET}\n"
        exit 1
    fi
elif [ -n "$TARGET_VERSION" ]; then
    INSTALL_SPEC="aimlite==${TARGET_VERSION#v}"
    echo -e "  ${BRIGHT_GREEN}➜${RESET}  ${BOLD}Target${RESET}       : $INSTALL_SPEC ${DIM}(from PyPI)${RESET}"
else
    echo -e "  ${BRIGHT_GREEN}➜${RESET}  ${BOLD}Target${RESET}       : aimlite ${DIM}(latest release from PyPI)${RESET}"
fi

# 4. Resolve package manager & install flags
BREAK_FLAG=""
if "$PYTHON_BIN" -m pip install --help 2>&1 | grep -q -- "--break-system-packages"; then
    BREAK_FLAG="--break-system-packages"
fi

UPGRADE_FLAG="--upgrade"
if [ "$FORCE" = true ]; then
    UPGRADE_FLAG="--force-reinstall"
fi

# 5. Execute Installation
echo ""
echo -e "  ${DIM}Installing AIMLite package and CLI binary...${RESET}"

INSTALL_SUCCESS=false

if [ "$IS_VENV" = true ]; then
    # In virtualenv: uv pip install or python -m pip install
    if command -v uv >/dev/null 2>&1; then
        echo -e "  ${DIM}Using backend: uv pip install${RESET}"
        if uv pip install $UPGRADE_FLAG $INSTALL_SPEC 2>/dev/null; then
            INSTALL_SUCCESS=true
        fi
    fi

    if [ "$INSTALL_SUCCESS" = false ]; then
        echo -e "  ${DIM}Using backend: pip install${RESET}"
        if "$PYTHON_BIN" -m pip install $UPGRADE_FLAG $INSTALL_SPEC; then
            INSTALL_SUCCESS=true
        fi
    fi
else
    # Outside virtualenv: respect --user / --global / PEP 668
    if [ "$MODE_USER" = false ]; then
        # Explicit global install requested
        echo -e "  ${DIM}Using backend: pip install (global)${RESET}"
        if "$PYTHON_BIN" -m pip install $UPGRADE_FLAG $INSTALL_SPEC 2>/dev/null; then
            INSTALL_SUCCESS=true
        elif [ -n "$BREAK_FLAG" ]; then
            if "$PYTHON_BIN" -m pip install $UPGRADE_FLAG $BREAK_FLAG $INSTALL_SPEC; then
                INSTALL_SUCCESS=true
            fi
        fi
    else
        # Default or explicit --user: try user-site first
        echo -e "  ${DIM}Using backend: pip install (--user)${RESET}"
        if "$PYTHON_BIN" -m pip install $UPGRADE_FLAG --user $INSTALL_SPEC 2>/dev/null; then
            INSTALL_SUCCESS=true
        elif [ -n "$BREAK_FLAG" ]; then
            # Retry with --break-system-packages (for PEP 668 on Debian 12+, Ubuntu 23.04+, Fedora, Homebrew)
            echo -e "  ${DIM}Retrying with --break-system-packages...${RESET}"
            if "$PYTHON_BIN" -m pip install $UPGRADE_FLAG --user $BREAK_FLAG $INSTALL_SPEC; then
                INSTALL_SUCCESS=true
            fi
        fi

        # Fallback to global if user site fails (e.g. running in Docker container as root)
        if [ "$INSTALL_SUCCESS" = false ]; then
            echo -e "  ${DIM}Retrying system-wide install...${RESET}"
            if "$PYTHON_BIN" -m pip install $UPGRADE_FLAG $BREAK_FLAG $INSTALL_SPEC; then
                INSTALL_SUCCESS=true
            fi
        fi
    fi
fi

if [ "$INSTALL_SUCCESS" = false ]; then
    echo ""
    echo -e "  ${RED}✖ Installation failed.${RESET}"
    echo -e "  ${DIM}Please try running manually:${RESET} ${CYAN}$PYTHON_BIN -m pip install --upgrade aimlite${RESET}\n"
    exit 1
fi

# 6. Verify installation and CLI accessibility
INSTALLED_VER=$("$PYTHON_BIN" -c 'import importlib.metadata; print(importlib.metadata.version("aimlite"))' 2>/dev/null || echo "")

echo ""
if [ -n "$INSTALLED_VER" ]; then
    echo -e "  ${GREEN}✔${RESET}  ${BOLD}Successfully installed AIMLite v$INSTALLED_VER!${RESET}"
else
    echo -e "  ${GREEN}✔${RESET}  ${BOLD}Successfully installed AIMLite!${RESET}"
fi

# Check PATH availability for the CLI executable
USER_BIN_DIR=$("$PYTHON_BIN" -c '
import sys, sysconfig
try:
    print(sysconfig.get_path("scripts", f"{sys.platform}_user"))
except Exception:
    import site
    base = getattr(site, "USER_BASE", "")
    if sys.platform == "win32":
        print(f"{base}/Scripts")
    else:
        print(f"{base}/bin")
' 2>/dev/null || echo "")

if ! command -v aimlite >/dev/null 2>&1; then
    if [ -n "$USER_BIN_DIR" ] && [ -d "$USER_BIN_DIR" ]; then
        echo ""
        echo -e "  ${YELLOW}! Warning:${RESET} The directory ${BOLD}$USER_BIN_DIR${RESET} is not in your current PATH."
        echo -e "  ${DIM}To use the 'aimlite' command from any terminal, add it to your shell configuration:${RESET}"
        echo ""
        echo -e "    ${CYAN}export PATH=\"$USER_BIN_DIR:\$PATH\"${RESET}"
        echo ""
        SHELL_RC=""
        if [ -n "${ZSH_VERSION:-}" ] || [ "$(basename "${SHELL:-}")" = "zsh" ]; then
            SHELL_RC="$HOME/.zshrc"
        elif [ -n "${BASH_VERSION:-}" ] || [ "$(basename "${SHELL:-}")" = "bash" ]; then
            SHELL_RC="$HOME/.bashrc"
        fi
        if [ -n "$SHELL_RC" ]; then
            echo -e "  ${DIM}You can add it automatically by running:${RESET}"
            echo -e "    ${CYAN}echo 'export PATH=\"$USER_BIN_DIR:\$PATH\"' >> $SHELL_RC && source $SHELL_RC${RESET}"
        fi
    fi
fi

# Print next steps
echo ""
echo -e "  ${BOLD}Get started with AIMLite:${RESET}"
echo -e "    1. Scaffold a project : ${CYAN}aimlite init my_first_ai${RESET}"
echo -e "    2. Enter directory     : ${CYAN}cd my_first_ai${RESET}"
echo -e "    3. Launch web app & API: ${CYAN}aimlite serve${RESET}"
echo ""
echo -e "  ${DIM}Documentation: https://github.com/Alazar42/AIMLite${RESET}"
echo ""
