#!/bin/bash

echo "====================================="
echo "🔎 DEBUGGING MODULE & CONDA ACTIVATION"
echo "====================================="

echo "Current Shell: $SHELL"
echo "Is Interactive? [[ \$- == *i* ]] → [[ $- == *i* ]]"
echo "Hostname: $(hostname)"
echo "Working Directory: $(pwd)"
echo "Date: $(date)"

echo "-----------------------------"
echo "1️⃣  Loading Anaconda3 module"
echo "-----------------------------"
module purge
module load Anaconda3

echo "PATH after module load:"
echo "$PATH"

echo "-----------------------------"
echo "2️⃣  Checking CUDA Modules"
echo "-----------------------------"
module spider CUDA

CUDA_VERSION=$(module avail CUDA 2>&1 | grep -oP 'CUDA/\\S+' | sort -V | tail -n 1)
if [[ -z "$CUDA_VERSION" ]]; then
    echo "⚠️ No CUDA modules found."
else
    echo "Loading CUDA module: $CUDA_VERSION"
    module load $CUDA_VERSION
fi

echo "-----------------------------"
echo "3️⃣  Sourcing conda.sh"
echo "-----------------------------"
CONDA_SH="/apps/system/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh"
if [[ -f "$CONDA_SH" ]]; then
    echo "Found $CONDA_SH — sourcing it..."
    source "$CONDA_SH"
else
    echo "❌ ERROR: $CONDA_SH not found — cannot source conda."
    exit 1
fi

echo "-----------------------------"
echo "4️⃣  Checking Conda Availability"
echo "-----------------------------"
if command -v conda >/dev/null 2>&1; then
    echo "✅ Conda command is available."
else
    echo "❌ ERROR: conda still not found after sourcing conda.sh!"
    exit 1
fi

echo "-----------------------------"
echo "5️⃣  Checking for environment"
echo "-----------------------------"
if conda env list | grep -q "rl_train_env"; then
    echo "✅ Environment 'rl_train_env' exists — attempting activation"
    conda activate rl_train_env
else
    echo "⚠️ Environment not found. Skipping activation."
fi

echo "-----------------------------"
echo "6️⃣  Final Python Diagnostics"
echo "-----------------------------"
echo "Using Python: $(which python)"
echo "Python version: $(python --version)"

echo "-----------------------------"
echo "7️⃣  CUDA Runtime Check (if available)"
echo "-----------------------------"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "⚠️ No GPU detected or nvidia-smi unavailable."
fi

echo "====================================="
echo "✅ DEBUG COMPLETE — No training launched"
echo "====================================="
