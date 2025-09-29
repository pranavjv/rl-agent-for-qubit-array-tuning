#!/bin/bash
#SBATCH --job-name=arc_qarray_rl_training
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --clusters=htc
#SBATCH --partition=long

mkdir -p logs

echo "========= SLURM ENVIRONMENT ========="
echo "Hostname: $(hostname)"
echo "Working Directory: $(pwd)"
echo "Date: $(date)"
echo "====================================="

# -----------------------------
# 1. Load Anaconda and CUDA
# -----------------------------
module purge
module load Anaconda3

CUDA_VERSION=$(module spider CUDA 2>&1 | grep -oP 'CUDA/12\.[0-9\.]+' | sort -V | tail -n 1)
if [[ -z "$CUDA_VERSION" ]]; then
    echo "⚠️ No CUDA 12.x modules found. Trying any available version..."
    CUDA_VERSION=$(module spider CUDA 2>&1 | grep -oP 'CUDA/[0-9\.]+' | sort -V | tail -n 1)
fi

if [[ -z "$CUDA_VERSION" ]]; then
    echo "❌ No CUDA modules found — exiting."
    exit 1
else
    echo "✅ Loading CUDA module: $CUDA_VERSION"
    module load "$CUDA_VERSION"
fi

# -----------------------------
# 2. Enable Conda shell & Activate Environment
# -----------------------------
source /apps/system/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh

if conda env list | grep -q "rl_train_env"; then
    echo "Activating existing Conda environment 'rl_train_env'..."
    conda activate rl_train_env
else
    echo "❌ Environment 'rl_train_env' missing — exiting."
    exit 1
fi

echo "✅ Using Python: $(which python)"
echo "✅ Python version: $(python --version)"

# -----------------------------
# 3. Validate GPU is visible
# -----------------------------
echo "========= GPU CHECK ========="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "❌ ERROR: nvidia-smi not available — GPU may not be allocated correctly."
    exit 1
fi

# Optional: Check PyTorch sees GPU
python - << 'EOF'
import torch
print("Torch CUDA Available:", torch.cuda.is_available())
print("Torch Device Count:", torch.cuda.device_count())
print("Torch Current Device:", torch.cuda.current_device() if torch.cuda.is_available() else "N/A")
EOF
echo "============================="

# -----------------------------
# 4. Run training script
# -----------------------------
echo "🚀 Launching training..."
echo "Did not train, debug finished."
# srun python src/swarm/training/train.py
