#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ENV_DIR:-$HOME/venvs/cafa5-notebook}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
KERNEL_NAME="${KERNEL_NAME:-cafa5-notebook}"
KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME:-Python (CAFA5 Notebook)}"
INSTALL_FULL_GRAPH="${INSTALL_FULL_GRAPH:-0}"
INSTALL_MULTIMODAL="${INSTALL_MULTIMODAL:-0}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python interpreter '$PYTHON_BIN' was not found." >&2
  echo "Load a Python 3.11 module first, for example: module load python/3.11" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$ROOT_DIR/requirements-notebook-savio.txt"

if [[ "$INSTALL_FULL_GRAPH" == "1" ]]; then
  python -m pip install -r "$ROOT_DIR/requirements-graph-local.txt"
fi

if [[ "$INSTALL_MULTIMODAL" == "1" ]]; then
  python -m pip install -r "$ROOT_DIR/requirements-remote-multimodal.txt"
fi

python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"

cat <<EOF
Savio notebook environment is ready.

Environment: $ENV_DIR
Kernel name: $KERNEL_NAME
Kernel label: $KERNEL_DISPLAY_NAME

Open this notebook in Jupyter and select the new kernel:
  $ROOT_DIR/output/jupyter-notebook/cafa5-alphafold-preprocessing-walkthrough.ipynb
EOF

