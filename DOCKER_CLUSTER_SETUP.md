# Docker Cluster Setup Guide

This guide explains how to set up and run your RL training code using Docker images on a SLURM cluster with GitLab registry.

## Prerequisites

1. Docker installed on your local machine for building images
2. Access to GitLab container registry
3. Docker or nvidia-docker installed on cluster nodes
4. GitLab personal access token or deploy token

## Setup Steps

### 1. Configure GitLab Registry Access

Create a GitLab personal access token with `read_registry` and `write_registry` scopes:

1. Go to GitLab → User Settings → Access Tokens
2. Create a token with registry access
3. Save the token securely

Store the token on the cluster:
```bash
# On the cluster node
echo "YOUR_GITLAB_TOKEN" > ~/.gitlab-token
chmod 600 ~/.gitlab-token
```

### 2. Update Configuration Files

The configuration files have been updated with your specific information:

#### build_and_push_docker.sh
```bash
NAMESPACE="pvaidhyanath"    # Your GitLab namespace
PROJECT="rlqarray"          # Your project name
```

#### cluster.sh or cluster_with_auth.sh
```bash
DOCKER_IMAGE="gitlab-master.nvidia.com/pvaidhyanath/rlqarray/rl-qarray-training:latest"
```

The scripts are configured to use 2 GPUs per node (updated from 8).

### 3. Build and Push Docker Image

On your local machine:

```bash
# Make sure you're in the project root directory
cd /path/to/rl-agent-for-qubit-array-tuning

# Login to GitLab registry
docker login gitlab-master.nvidia.com

# Build and push the image
./build_and_push_docker.sh
```

### 4. Submit SLURM Job

On the cluster:

```bash
# Copy your project files to the cluster
scp -r /path/to/rl-agent-for-qubit-array-tuning user@cluster:/path/to/destination

# SSH to the cluster
ssh user@cluster

# Navigate to project directory
cd /path/to/destination

# Submit the job
sbatch cluster.sh
# OR use the version with authentication
sbatch cluster_with_auth.sh
```

## Docker Image Contents

The Dockerfile includes:
- NVIDIA CUDA 12.3.0 base image with cuDNN 9
- Python 3.11
- All dependencies from requirements.txt
- Your complete project code

## Troubleshooting

### Authentication Issues

If you encounter authentication problems:

1. Verify your GitLab token has the correct permissions
2. Check if Docker config exists: `ls ~/.docker/config.json`
3. Try manual login: `docker login gitlab-master.nvidia.com`

### GPU Access Issues

If GPUs aren't accessible in the container:

1. Verify SLURM allocated GPUs: Check the job output for nvidia-smi results
2. Ensure nvidia-docker is installed on cluster nodes
3. Try using `--gpus all` flag explicitly

### Network Issues

If you need to access external resources from within the container:

1. The `--network=host` flag is included in cluster_with_auth.sh
2. You may need to mount additional certificates or configs

### Debugging Container Issues

To debug, you can run an interactive session:

```bash
srun --gres=gpu:1 --pty bash
docker run --rm -it --gpus all -v $(pwd):$(pwd) -w $(pwd) YOUR_IMAGE_NAME bash
```

## Alternative: Using Singularity

If your cluster prefers Singularity over Docker:

```bash
# Convert Docker image to Singularity
singularity pull docker://gitlab-master.nvidia.com/your-namespace/your-project/rl-qarray-training:latest

# Run with Singularity in SLURM script
srun singularity exec --nv \
     --bind ${WORK_DIR}:${WORK_DIR} \
     rl-qarray-training_latest.sif \
     python src/swarm/training/train.py
```

## Security Best Practices

1. Never commit tokens or credentials to git
2. Use deploy tokens for production instead of personal tokens
3. Regularly rotate access tokens
4. Use specific image tags instead of 'latest' for reproducibility
5. Scan images for vulnerabilities before deployment

## Updating the Image

When you need to update the Docker image:

1. Make your code changes
2. Update the TAG in build_and_push_docker.sh (e.g., v1.1, v1.2)
3. Run `./build_and_push_docker.sh`
4. Update the DOCKER_IMAGE in cluster.sh with the new tag
5. Submit a new SLURM job

