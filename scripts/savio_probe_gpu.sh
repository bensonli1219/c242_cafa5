#!/bin/bash
#SBATCH --job-name=probe_gpu
#SBATCH --account=ic_chem242
#SBATCH --partition=savio2_1080ti
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=/global/scratch/users/%u/logs/probe_gpu_%j.out
#SBATCH --error=/global/scratch/users/%u/logs/probe_gpu_%j.err

set -euo pipefail

PYTORCH_MODULE="${PYTORCH_MODULE:-ml/pytorch/2.3.1-py3.11.7}"
PYTHON_BIN="${PYTHON_BIN:-/global/home/users/$USER/venvs/cafa5-pytorch-2.3.1/bin/python}"

echo "started_at=$(date -Iseconds)"
echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-}"
echo "partition=${SLURM_JOB_PARTITION:-}"
echo "node_list=${SLURM_JOB_NODELIST:-}"
echo "slurm_gpus=${SLURM_GPUS:-}"
echo "slurm_gpus_on_node=${SLURM_GPUS_ON_NODE:-}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
echo "gpu_device_ordinals=${SLURM_JOB_GPUS:-}"

if command -v module >/dev/null 2>&1; then
  set +u
  module load "$PYTORCH_MODULE"
  set -u
else
  source /etc/profile.d/modules.sh
  set +u
  module load "$PYTORCH_MODULE"
  set -u
fi

echo "loaded_pytorch_module=$PYTORCH_MODULE"
echo "nvidia_smi_path=$(command -v nvidia-smi || true)"
nvidia-smi || true

echo "python_bin=$PYTHON_BIN"
"$PYTHON_BIN" - <<'PY'
import os
import sys

print("python_executable=" + sys.executable)
print("CUDA_VISIBLE_DEVICES=" + str(os.environ.get("CUDA_VISIBLE_DEVICES")))

try:
    import torch
except Exception as exc:
    print("torch_import_error=" + repr(exc))
    raise

print("torch_version=" + str(getattr(torch, "__version__", "unknown")))
print("torch_cuda_version=" + str(getattr(torch.version, "cuda", None)))
print("torch_cuda_available=" + str(torch.cuda.is_available()))
print("torch_cuda_device_count=" + str(torch.cuda.device_count()))

if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        print(f"torch_cuda_device_{index}_name=" + torch.cuda.get_device_name(index))
    x = torch.ones((1024, 1024), device="cuda")
    y = x @ x
    torch.cuda.synchronize()
    print("torch_cuda_matmul_sum=" + str(float(y.sum().item())))
PY

echo "finished_at=$(date -Iseconds)"
