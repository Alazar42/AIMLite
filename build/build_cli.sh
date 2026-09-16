#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

echo "Building ModelKit CLI executable using $PYTHON_BIN..."
"$PYTHON_BIN" "$SCRIPT_DIR/build_cli.py" "$@"
