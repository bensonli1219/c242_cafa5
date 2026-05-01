#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ENV_DIR:-$HOME/venvs/cafa5-gpu-cu121}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
KERNEL_NAME="${KERNEL_NAME:-cafa5-gpu-cu121}"
KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME:-Python (CAFA5 GPU cu121)}"
TMPDIR="${TMPDIR:-$HOME/tmp/pip-tmp}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter '$PYTHON_BIN' was not found." >&2
  echo "Load a Python 3.11 module first, for example: module load python/3.11" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
export TMPDIR PIP_CACHE_DIR

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$ROOT_DIR/requirements/notebook-savio.txt"
python -m pip install -r "$ROOT_DIR/requirements/graph-savio-cu121.txt"
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"

cat <<EOF
Savio GPU graph environment is ready.

Environment: $ENV_DIR
Kernel name: $KERNEL_NAME
Kernel label: $KERNEL_DISPLAY_NAME

Use this Python for Slurm training:
  $ENV_DIR/bin/python
EOF
