# Quick Start Guide - Docker Setup for SLURM Cluster

## Your Configuration
- **GitLab Registry**: gitlab-master.nvidia.com
- **Namespace**: pvaidhyanath  
- **Project**: rlqarray
- **Image**: rl-qarray-training:latest
- **GPUs per node**: 2

## Step 1: Store Your GitLab Token (On Cluster)

On your cluster node, run:
```bash
./setup_gitlab_token.sh
```
When prompted, paste your GitLab token.

**Alternative manual method:**
```bash
echo "YOUR_GITLAB_TOKEN" > ~/.gitlab-token
chmod 600 ~/.gitlab-token
```

## Step 2: Build and Push Docker Image (On Local Machine)

```bash
# Login to GitLab registry
docker login gitlab-master.nvidia.com

# Build and push the image
./build_and_push_docker.sh
```

## Step 3: Submit SLURM Job (On Cluster)

```bash
# Use the version with authentication
sbatch cluster_with_auth.sh

# OR use the basic version (if authentication is already set up)
sbatch cluster.sh
```

## Monitor Your Job

```bash
# Check job status
squeue -u $USER

# View job output
tail -f logs/nvidia_qarray_rl_training_*.out

# View job errors
tail -f logs/nvidia_qarray_rl_training_*.err
```

## Troubleshooting

1. **Authentication Failed**: Verify token is stored correctly
   ```bash
   cat ~/.gitlab-token  # Should show your token
   ls -la ~/.gitlab-token  # Should show -rw------- permissions
   ```

2. **Image Pull Failed**: Test docker login manually
   ```bash
   docker login gitlab-master.nvidia.com -u gitlab-ci-token -p $(cat ~/.gitlab-token)
   ```

3. **GPU Issues**: Check SLURM allocation
   ```bash
   srun --gres=gpu:2 nvidia-smi
   ```

## Important Notes

- Your GitLab token should NEVER be committed to git
- The scripts are configured for 2 GPUs (not 8)
- Docker image includes all dependencies from requirements.txt
- The `qarray` package installation may need manual configuration in Dockerfile
