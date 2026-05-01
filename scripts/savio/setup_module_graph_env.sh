#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ENV_DIR:-$HOME/venvs/cafa5-pytorch-2.3.1}"
PYTORCH_MODULE="${PYTORCH_MODULE:-ml/pytorch/2.3.1-py3.11.7}"
KERNEL_NAME="${KERNEL_NAME:-cafa5-pytorch-2.3.1}"
KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME:-Python (CAFA5 PyTorch module 2.3.1)}"
TMPDIR="${TMPDIR:-$HOME/tmp/pip-tmp}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"

if ! command -v module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh
fi

set +u
module load "$PYTORCH_MODULE"
set -u
python -m venv --system-site-packages "$ENV_DIR"
source "$ENV_DIR/bin/activate"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
export TMPDIR PIP_CACHE_DIR
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$ROOT_DIR/requirements/notebook-savio.txt"
python -m pip install -r "$ROOT_DIR/requirements/graph-savio-module.txt"
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY_NAME"

cat <<EOF
Savio module-backed graph environment is ready.

Environment: $ENV_DIR
PyTorch module: $PYTORCH_MODULE
Kernel name: $KERNEL_NAME
Kernel label: $KERNEL_DISPLAY_NAME

Use this Python for Slurm training:
  $ENV_DIR/bin/python
EOF
