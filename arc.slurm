#!/bin/bash
#SBATCH --job-name=arc_qarray_rl_training
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:8 --constraint='gpu_mem:32GB'
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=24
#SBATCH --mem=16G
#SBATCH --clusters=htc
#SBATCH --partition=long

# -----------------------------
# 1. Load Anaconda and CUDA
# -----------------------------
module purge
module load Anaconda3

echo "Available CUDA modules:"
module spider CUDA

CUDA_VERSION=$(module avail CUDA 2>&1 | grep -oP 'CUDA/\S+' | sort -V | tail -n 1)
if [[ -z "$CUDA_VERSION" ]]; then
    echo "⚠️ No CUDA modules found. Proceeding without CUDA."
else
    echo "Loading CUDA module: $CUDA_VERSION"
    module load $CUDA_VERSION
fi

# -----------------------------
# 2. Activate Conda environment
# -----------------------------
source ~/.bashrc

# Check if environment exists
if conda env list | grep -q "rl_train_env"; then
    echo "Activating existing Conda environment 'rl_train_env'..."
    conda activate rl_train_env
else
    echo "Environment 'rl_train_env' not found. Creating from YAML..."
    if [ -f rl_train_env.yml ]; then
        conda env create -f rl_train_env.yml
        conda activate rl_train_env
    else
        echo "❌ YAML file rl_train_env.yml not found. Cannot create environment."
        exit 1
    fi
fi

# -----------------------------
# 3. Run your training script
# -----------------------------
echo "Running training script..."
srun python src/swarm/train.py
