#!/bin/bash

echo "=============================="
echo "User: $USER"
echo "Hostname: $(hostname)"
echo "Working directory: $PWD"
echo "Date: $(date)"
echo "=============================="

# Load Anaconda module
module purge
module load Anaconda3

# Initialize conda shell functions
source /apps/system/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh

# Activate environment
ENV_NAME="rl_train_env"

if conda env list | grep -q "$ENV_NAME"; then
    echo "Activating environment '$ENV_NAME'..."
    conda activate "$ENV_NAME"
else
    echo "Environment '$ENV_NAME' not found!"
    exit 1
fi

# Test environment
echo "Python version in environment:"
python --version

echo "Python executable location:"
which python

echo "Installed packages (first 10 lines):"
pip list | head -n 10

echo "✅ Environment activation test complete."
