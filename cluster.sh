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

# Define the Docker image from your GitLab registry
# TODO: Replace this with your actual GitLab registry image
DOCKER_IMAGE="gitlab-master.nvidia.com/pvaidhyanath/rlqarray/rl-qarray-training:latest"

# Define the working directory (current project directory)
WORK_DIR=$(pwd)

# Define any additional data directories you need to mount
# DATA_DIR="/path/to/your/data"  # Uncomment and set if needed

# Number of GPUs per node
NPROC_PER_NODE=2
TOTAL_GPU=$(($SLURM_JOB_NUM_NODES * $NPROC_PER_NODE))

# Command to run inside the container
RUN_CMD="python src/swarm/training/train.py"

echo "Running on hosts: $(scontrol show hostname)"
echo "Using Docker image: ${DOCKER_IMAGE}"

# Check if nvidia-docker is available, otherwise use regular docker with GPU flags
if command -v nvidia-docker &> /dev/null; then
    DOCKER_CMD="nvidia-docker"
else
    DOCKER_CMD="docker"
fi

srun --ntasks=1 \
     bash -c "
     set -x
     # Export environment variables needed for distributed training
     export WORLD_SIZE=${TOTAL_GPU}
     export WORLD_RANK=\${PMIX_RANK:-0}
     export CUDA_VISIBLE_DEVICES=0,1
     
     # Pull the latest image (will use cached version if already present)
     ${DOCKER_CMD} pull ${DOCKER_IMAGE}
     
     # Run the Docker container
     ${DOCKER_CMD} run --rm \
                   --gpus all \
                   --ipc=host \
                   -v ${WORK_DIR}:${WORK_DIR} \
                   -w ${WORK_DIR} \
                   -e WORLD_SIZE=${WORLD_SIZE} \
                   -e WORLD_RANK=\${WORLD_RANK} \
                   -e CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES} \
                   ${DOCKER_IMAGE} \
                   ${RUN_CMD}
     "


