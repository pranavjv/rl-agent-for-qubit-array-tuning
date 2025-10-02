#!/bin/bash
#SBATCH --job-name=nvidia_qarray_rl_training
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G

set -e
set -u

mkdir -p logs #sbatch logs placed in here

# GitLab Registry Configuration
GITLAB_REGISTRY="gitlab-master.nvidia.com"
DOCKER_IMAGE="${GITLAB_REGISTRY}/pvaidhyanath/rlqarray/rl-qarray-training:latest"

# Authentication method (choose one):
# Option 1: Using GitLab Personal Access Token (recommended)
# Store your token in a file with restricted permissions: chmod 600 ~/.gitlab-token
if [ -f "$HOME/.gitlab-token" ]; then
    GITLAB_TOKEN=$(cat "$HOME/.gitlab-token")
    echo "Using GitLab token for authentication"
fi

# Option 2: Using Docker config (if already logged in)
# Make sure ~/.docker/config.json exists with proper credentials

# Define the working directory (current project directory)
WORK_DIR=$(pwd)

# Number of GPUs per node
NPROC_PER_NODE=2
TOTAL_GPU=$(($SLURM_JOB_NUM_NODES * $NPROC_PER_NODE))

# Command to run inside the container
RUN_CMD="python src/swarm/training/train.py"

echo "Running on hosts: $(scontrol show hostname)"
echo "Using Docker image: ${DOCKER_IMAGE}"

# Check if nvidia-docker is available
if command -v nvidia-docker &> /dev/null; then
    DOCKER_CMD="nvidia-docker"
else
    DOCKER_CMD="docker"
fi

srun --ntasks=1 \
     bash -c "
     set -x
     
     # Authenticate with GitLab registry if token is available
     if [ ! -z \"\${GITLAB_TOKEN:-}\" ]; then
         echo \${GITLAB_TOKEN} | ${DOCKER_CMD} login ${GITLAB_REGISTRY} --username gitlab-ci-token --password-stdin
     fi
     
     # Pull the latest image
     ${DOCKER_CMD} pull ${DOCKER_IMAGE}
     
     # Export environment variables for distributed training
     export WORLD_SIZE=${TOTAL_GPU}
     export WORLD_RANK=\${PMIX_RANK:-0}
     export CUDA_VISIBLE_DEVICES=0,1
     
     # Run the Docker container
     ${DOCKER_CMD} run --rm \
                   --gpus all \
                   --ipc=host \
                   --network=host \
                   -v ${WORK_DIR}:${WORK_DIR} \
                   -v /tmp:/tmp \
                   -w ${WORK_DIR} \
                   -e WORLD_SIZE=\${WORLD_SIZE} \
                   -e WORLD_RANK=\${WORLD_RANK} \
                   -e CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES} \
                   -e NCCL_DEBUG=INFO \
                   -e PYTHONPATH=${WORK_DIR} \
                   ${DOCKER_IMAGE} \
                   ${RUN_CMD}
     "

