#!/bin/bash
#SBATCH --job-name=esm_job
#SBATCH --account=ic_chem242
#SBATCH --partition=savio2_1080ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --qos=savio_normal
#SBATCH --output=/global/home/users/yushanfu/logs/esm_%j.out
#SBATCH --error=/global/home/users/yushanfu/logs/esm_%j.err

cd /global/home/users/yushanfu

module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /global/home/users/yushanfu/.conda/envs/esm_env

export PYTHONNOUSERSITE=1
unset PYTHONPATH

echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
which python
python -V

nvidia-smi

python -u /global/home/users/yushanfu/ESM2_embedding.py
