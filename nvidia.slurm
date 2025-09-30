#!/bin/bash
#SBATCH --job-name=nvidia_qarray_rl_training
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G

set -e
set -u

mkdir -p logs #sbatch logs placed in here

module purge
module load Anaconda3
module load CUDA/12.3.0


ENV_NAME="rl_train_env"

if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "creating new conda env"
    conda create -y -n "$ENV_NAME" python=3.11

    conda run -n "$ENV_NAME" pip install --upgrade pip
    conda run -n "$ENV_NAME" pip install -r requirements.txt
    echo "env setup done"
else
    echo "using existing conda env"
fi

nvidia-smi || { echo "ERROR: nvidia-smi failed"; exit 1; }

srun conda run --no-capture-output -n "$ENV_NAME" python src/swarm/training/train.py
