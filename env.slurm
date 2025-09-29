#!/bin/bash
#SBATCH --job-name=create_rl_env
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --clusters=htc
#SBATCH --partition=long

echo "=== Loading Anaconda ==="
module purge
module load Anaconda3

echo "=== Checking available CUDA versions ==="
module spider CUDA

# Try to load latest CUDA version automatically
# (grabs highest-numbered version from `module avail` output)
CUDA_VERSION=$(module avail CUDA 2>&1 | grep -oP 'CUDA/\S+' | sort -V | tail -n 1)

if [[ -z "$CUDA_VERSION" ]]; then
  echo "⚠️ No CUDA modules found. Proceeding without CUDA."
else
  echo "Loading CUDA module: $CUDA_VERSION"
  module load $CUDA_VERSION
fi

# Activate conda init
source ~/.bashrc

# Create environment if missing
if conda env list | grep -q "rl_train_env"; then
  echo "Environment rl_train_env already exists. Skipping creation."
else
  echo "Creating Conda env with Python 3.11..."
  conda create -y -n rl_train_env python=3.11
fi

# Activate and install requirements
echo "Activating environment and installing requirements..."
source activate rl_train_env
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Environment setup complete."


# Run with: sbatch env.slurm