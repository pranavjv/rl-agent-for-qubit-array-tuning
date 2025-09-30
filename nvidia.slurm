#!/bin/bash
#SBATCH --job-name=nvidia_qarray_rl_training
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

mkdir -p logs

echo "Hostname: $(hostname)"
echo "Working Directory: $(pwd)"
echo "Date: $(date)"


module purge
module load Anaconda3

CUDA_VERSION=$(module spider CUDA 2>&1 | grep -oP 'CUDA/12\.[0-9\.]+' | sort -V | tail -n 1)
if [[ -z "$CUDA_VERSION" ]]; then
    echo "No CUDA 12.x modules found. Trying any available version..."
    CUDA_VERSION=$(module spider CUDA 2>&1 | grep -oP 'CUDA/[0-9\.]+' | sort -V | tail -n 1)
fi

if [[ -z "$CUDA_VERSION" ]]; then
    echo "No CUDA modules found — exiting."
    exit 1
else
    module load "$CUDA_VERSION"
fi


CONDA_SH="/apps/system/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh"
if [[ -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"
else
    echo "conda.sh not found at $CONDA_SH"
    exit 1
fi


conda create -y -n rl_train_env python=3.11
conda activate rl_train_env

pip install --upgrade pip
if [[ -f requirements.txt ]]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found. Skipping package installation."
    exit 1
fi


if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "ERROR: nvidia-smi not available — GPU may not be allocated correctly."
    exit 1
fi

srun python src/swarm/training/train.py
