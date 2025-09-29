#!/bin/bash
#SBATCH --job-name=arc_qarray_rl_training           # Job name
#SBATCH --output=logs/%x_%j.out                     # Output file (%x = job name, %j = job ID)
#SBATCH --error=logs/%x_%j.err                      # Error file
#SBATCH --time=100:00:00                            # Time limit (hh:mm:ss)
#SBATCH --gres=gpu:8 --constraint='gpu_mem:32GB'    # GPU specs
#SBATCH --nodes=2                                   # Total number of nodes
#SBATCH --ntasks-per-node=4                         # Number of MPI tasks per node
#SBATCH --cpus-per-task=24                          # Number of CPU cores per task
#SBATCH --mem=16G                                   # Memory per node
#SBATCH --clusters=htc                              # Specify resource cluster
#SBATCH --partition=long                            # medium jobs (max 48h), long jobs (unlimited time)

# Load modules
module purge
module load python/3.11 cuda/12.4

# Activate env
source ~/.bashrc
conda activate rl_train_env

# Run your program
srun python train.py
