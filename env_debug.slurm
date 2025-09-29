#!/bin/bash
#SBATCH --job-name=check_conda_env
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --clusters=htc
#SBATCH --partition=short

# ------- Load Conda -------
module purge
module load Anaconda3

# Source conda init
CONDA_SH="/apps/system/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh"
if [[ -f "$CONDA_SH" ]]; then
    source "$CONDA_SH"
else
    echo "❌ Could not find conda.sh at $CONDA_SH"
    exit 1
fi

# ------- Activate Environment -------
ENV_NAME="rl_train_env"
echo "Activating Conda environment: $ENV_NAME"
conda activate "$ENV_NAME" || { echo "❌ Failed to activate environment"; exit 1; }

echo "✅ Environment activated. Python: $(which python)"
echo "=================================="

# ------- List Packages --------
echo "📦 Installed Packages:"
conda list

echo "=================================="
echo "📏 Package Sizes (Human Readable):"

# Using conda-meta directory to estimate package sizes
PKG_DIR="$CONDA_PREFIX/pkgs"
if [[ -d "$PKG_DIR" ]]; then
    du -sh "$PKG_DIR"/* | sort -h
else
    echo "⚠️ Could not locate package cache at $PKG_DIR"
fi

echo "=================================="
echo "✅ Environment summary complete."
