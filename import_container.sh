#!/bin/bash
# Script to import Docker container for use with SLURM/pyxis

# Check if GitLab token exists
if [ ! -f "$HOME/.gitlab-token" ]; then
    echo "Error: GitLab token not found at $HOME/.gitlab-token"
    echo "Please create this file with your GitLab personal access token"
    echo "Example: echo 'your-token-here' > ~/.gitlab-token && chmod 600 ~/.gitlab-token"
    exit 1
fi

GITLAB_TOKEN=$(cat "$HOME/.gitlab-token")

# Define paths
# Adjust this path based on your cluster's filesystem structure
# Common locations are /lustre/fsw/portfolios/nvr/users/$USER/ or similar
CONTAINER_DIR="${HOME}/containers"
CONTAINER_NAME="rl-qarray-training.sqsh"
CONTAINER_PATH="${CONTAINER_DIR}/${CONTAINER_NAME}"

# Create directory if it doesn't exist
mkdir -p "${CONTAINER_DIR}"

echo "Importing container image..."
echo "Source: docker://gitlab-master.nvidia.com/pvaidhyanath/rlqarray/rl-qarray-training:latest"
echo "Destination: ${CONTAINER_PATH}"

# Import the container with authentication
# The --authhdr flag passes authentication header
enroot import \
    --output "${CONTAINER_PATH}" \
    --authhdr "Basic $(echo -n "gitlab-ci-token:${GITLAB_TOKEN}" | base64)" \
    docker://gitlab-master.nvidia.com/pvaidhyanath/rlqarray/rl-qarray-training:latest

if [ $? -eq 0 ]; then
    echo "Container successfully imported to: ${CONTAINER_PATH}"
    echo "You can now use this path in your SLURM script"
else
    echo "Error: Failed to import container"
    exit 1
fi
