#!/bin/bash
#SBATCH --job-name=arc_qarray_rl_training
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --clusters=htc
#SBATCH --partition=short
#SBATCH --mail-user=edn.work@proton.me
#SBATCH --mail-type=ALL

mkdir -p logs

echo "========= SLURM ENVIRONMENT ========="
echo "Hostname: $(hostname)"
echo "Working Directory: $(pwd)"
echo "Date: $(date)"
echo "====================================="

# Function to log disk usage
log_disk_usage() {
    echo "--------- Disk Usage & Quota ---------"
    echo "Disk usage of home directory:"
    du -sh ~
    echo "Disk quota and usage:"
    quota -s
    echo "-------------------------------------"
}

log_disk_usage

# -----------------------------
# 1️⃣ Load Anaconda & Latest CUDA
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

log_disk_usage

# -----------------------------
# 2️⃣ Enable Conda shell & Create/Activate Environment
# -----------------------------
CONDA_SH="/apps/system/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh"
if [[ -f "$CONDA_SH" ]]; then
    echo "Sourcing conda.sh..."
    source "$CONDA_SH"
else
    echo "❌ conda.sh not found at $CONDA_SH"
    exit 1
fi

# Clean caches before environment creation
echo "Cleaning Conda and pip caches before environment creation..."
conda clean --all -y
pip cache purge

log_disk_usage

# Create environment
echo "Creating Conda environment 'rl_train_env' with Python 3.11..."
conda create -y -n rl_train_env python=3.11
conda activate rl_train_env

echo "Installing required packages from requirements.txt..."
pip install --upgrade pip
if [[ -f requirements.txt ]]; then
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found. Skipping package installation."
fi

# Clean caches again after installation
echo "Cleaning Conda and pip caches after package installation..."
conda clean --all -y
pip cache purge

log_disk_usage

echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

# -----------------------------
# 3️⃣ Validate GPU visibility
# -----------------------------
echo "========= GPU CHECK ========="
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "❌ ERROR: nvidia-smi not available — GPU may not be allocated correctly."
    exit 1
fi

python - << 'EOF'
try:
    import torch
    print("Torch CUDA Available:", torch.cuda.is_available())
    print("Torch Device Count:", torch.cuda.device_count())
    print("Torch Current Device:", torch.cuda.current_device() if torch.cuda.is_available() else "N/A")
except ModuleNotFoundError:
    print("❌ Torch not installed in this environment")
EOF
echo "============================="

log_disk_usage