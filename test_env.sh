#!/bin/bash

echo "=============================="
echo "🔎 SHELL DIAGNOSTICS"
echo "=============================="
echo "Shell: $SHELL"
echo "Is interactive? [[ \$- == *i* ]] → $([[ $- == *i* ]] && echo yes || echo no)"
echo "Is login shell? shopt -q login_shell → $(shopt -q login_shell && echo yes || echo no)"
echo "BASH Version: $BASH_VERSION"
echo "Current User: $USER"
echo "Hostname: $(hostname)"
echo "Working Directory: $PWD"
echo "Date: $(date)"
echo "=============================="

echo "🔎 PATH INFO"
echo "$PATH"
echo "------------------------------"

echo "🔎 CHECK 'conda' COMMAND AVAILABILITY"
type conda 2>/dev/null || echo "'conda' not found in PATH"
which conda || echo "'which conda' returned nothing"
echo "------------------------------"

echo "🔎 Sourcing Anaconda3 Module..."
module purge
module load Anaconda3

echo "PATH after module load:"
echo "$PATH"
echo "------------------------------"

echo "🔎 Is 'conda' available now?"
type conda 2>/dev/null || echo "'conda' still not found after module load"
which conda || echo "'which conda' still returns nothing"

echo "------------------------------"
echo "🔎 Checking for conda.sh at expected path:"
ls -l /apps/system/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh

echo "------------------------------"
echo "🔎 Manually sourcing conda.sh..."
source /apps/system/easybuild/software/Anaconda3/2022.05/etc/profile.d/conda.sh

echo "🔎 After sourcing conda.sh → does conda exist as a function?"
type conda 2>/dev/null || echo "'conda' still not available as a function"

echo "------------------------------"
echo "🔎 Attempting 'conda activate rl_train_env'..."
conda activate rl_train_env 2>&1 || echo "❌ conda activate failed"

echo "Python version after attempted activation:"
python --version
which python

echo "------------------------------"
echo "🔎 Trying 'conda run -n rl_train_env python --version'..."
conda run -n rl_train_env python --version 2>&1 || echo "❌ conda run failed"

echo "=============================="
echo "✅ END OF DEBUG"
