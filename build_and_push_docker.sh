#!/bin/bash

# Configuration - Update these variables
GITLAB_REGISTRY="gitlab-master.nvidia.com"
NAMESPACE="pvaidhyanath"
PROJECT="rlqarray"
IMAGE_NAME="rl-qarray-training"
TAG="latest"

# Full image path
FULL_IMAGE_PATH="${GITLAB_REGISTRY}/${NAMESPACE}/${PROJECT}/${IMAGE_NAME}:${TAG}"

echo "Building Docker image: ${FULL_IMAGE_PATH}"

# Login to GitLab registry (you'll need to provide your credentials)
echo "Logging into GitLab registry..."
docker login ${GITLAB_REGISTRY}

# Build the Docker image
echo "Building Docker image..."
docker build -t ${FULL_IMAGE_PATH} .

# Push the image to GitLab registry
echo "Pushing image to GitLab registry..."
docker push ${FULL_IMAGE_PATH}

echo "Successfully built and pushed: ${FULL_IMAGE_PATH}"
echo ""
echo "Update the DOCKER_IMAGE variable in cluster.sh with:"
echo "DOCKER_IMAGE=\"${FULL_IMAGE_PATH}\""

