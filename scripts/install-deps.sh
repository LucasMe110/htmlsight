#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/venv}"
INSTALL_MODEL="${INSTALL_MODEL:-0}"
INSTALL_CHROMIUM="${INSTALL_CHROMIUM:-1}"
PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$VENV_DIR/ms-playwright}"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -e "$ROOT_DIR[dev,render]"

if [ "$INSTALL_MODEL" = "1" ]; then
  "$VENV_DIR/bin/python" -m pip install -e "$ROOT_DIR[model]"
fi

if [ "$INSTALL_CHROMIUM" = "1" ]; then
  PLAYWRIGHT_BROWSERS_PATH="$PLAYWRIGHT_BROWSERS_PATH" \
    "$VENV_DIR/bin/python" -m playwright install chromium
fi

echo "Dependencias instaladas em $VENV_DIR"
