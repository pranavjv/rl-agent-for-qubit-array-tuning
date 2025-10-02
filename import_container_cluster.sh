#!/bin/bash
# Script to import Docker container for use with SLURM/pyxis on cluster

# Check if GitLab token exists
if [ ! -f "$HOME/.gitlab-token" ]; then
    echo "Error: GitLab token not found at $HOME/.gitlab-token"
    echo "Please create this file with your GitLab personal access token"
    echo "Example: echo 'your-token-here' > ~/.gitlab-token && chmod 600 ~/.gitlab-token"
    exit 1
fi

GITLAB_TOKEN=$(cat "$HOME/.gitlab-token")

# Define paths - adjust based on your cluster's filesystem
# Common cluster paths:
# - /lustre/fsw/portfolios/nvr/users/$USER/
# - /scratch/$USER/
# - /work/$USER/
# - $HOME/containers/

# Try to detect appropriate path
if [ -d "/lustre/fsw/portfolios/nvr/users/$USER" ]; then
    CONTAINER_DIR="/lustre/fsw/portfolios/nvr/users/$USER/containers"
elif [ -d "/scratch/$USER" ]; then
    CONTAINER_DIR="/scratch/$USER/containers"
elif [ -d "/work/$USER" ]; then
    CONTAINER_DIR="/work/$USER/containers"
else
    CONTAINER_DIR="${HOME}/containers"
fi

CONTAINER_NAME="rl-qarray-training.sqsh"
CONTAINER_PATH="${CONTAINER_DIR}/${CONTAINER_NAME}"

# Create directory if it doesn't exist
mkdir -p "${CONTAINER_DIR}"

echo "Importing container image..."
echo "Source: docker://gitlab-master.nvidia.com/pvaidhyanath/rlqarray/rl-qarray-training:latest"
echo "Destination: ${CONTAINER_PATH}"

# Import the container with authentication
# Method 1: Using --authhdr (most common)
enroot import \
    --output "${CONTAINER_PATH}" \
    --authhdr "Basic $(echo -n "gitlab-ci-token:${GITLAB_TOKEN}" | base64)" \
    docker://gitlab-master.nvidia.com/pvaidhyanath/rlqarray/rl-qarray-training:latest

# If method 1 fails, try method 2 with environment variables
if [ $? -ne 0 ]; then
    echo "Method 1 failed, trying with environment variables..."
    
    # Set authentication environment variables
    export ENROOT_CRED_USER="gitlab-ci-token"
    export ENROOT_CRED_PASS="${GITLAB_TOKEN}"
    
    enroot import \
        --output "${CONTAINER_PATH}" \
        docker://gitlab-master.nvidia.com/pvaidhyanath/rlqarray/rl-qarray-training:latest
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "Container successfully imported to: ${CONTAINER_PATH}"
    echo ""
    echo "Next steps:"
    echo "1. Update cluster_with_auth.sh to use this path:"
    echo "   readonly _cont_image='${CONTAINER_PATH}'"
    echo "2. Submit your job: sbatch cluster_with_auth.sh"
else
    echo "Error: Failed to import container"
    echo ""
    echo "Troubleshooting tips:"
    echo "1. Verify your GitLab token has read access to the container registry"
    echo "2. Try running on a compute node with: srun --pty bash"
    echo "3. Check if enroot is available: which enroot"
    exit 1
fi
