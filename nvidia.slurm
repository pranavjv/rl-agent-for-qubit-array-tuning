#!/bin/bash
#SBATCH --job-name=my_job        # Job name
#SBATCH --output=logs/%x_%j.out  # Output file (%x = job name, %j = job ID)
#SBATCH --error=logs/%x_%j.err   # Error file
#SBATCH --time=100:00:00          # Time limit (hh:mm:ss)
#SBATCH --partition=gpu          # Partition/queue 
#SBATCH --gres=gpu:4             # Number of GPUs 
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=16G                # Memory per node

# Load modules
module purge
module load python/3.11 cuda/12.4  

# Activate env
source ~/.bashrc
conda activate rl_train_env

# Run your program
srun python train.py --resume-latest
