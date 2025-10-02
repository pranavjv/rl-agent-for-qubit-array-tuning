# SLURM Cluster Setup with Container Support

## Overview
This guide explains how to use the modified `cluster_with_auth.sh` script that now uses SLURM's native container support instead of Docker.

## Key Changes

### 1. Container Runtime
- **Before**: Used Docker directly with `docker run`
- **After**: Uses SLURM's container support with `--container-image` and `--container-mounts`

### 2. Script Structure
The new script follows the standard SLURM container job structure:

```bash
srun -A <account> \
     --container-image="<image>" \
     --container-mounts="<mounts>" \
     --ntasks-per-node=<gpus> \
     bash -c "<commands>"
```

### 3. Key Parameters

- **Account**: `nvr_asicvlsi_quantum` 
- **Job Name**: `nvr_asicvlsi_quantum:rl_qarray_train1`
- **Time Limit**: 3:58:00 (just under 4 hours)
- **Nodes**: 1 (can be increased as needed)
- **GPUs per Node**: 8
- **Container Image**: `gitlab-master.nvidia.com/pvaidhyanath/rlqarray/rl-qarray-training:latest`

## Usage

### Step 1: Set up GitLab Authentication

First, ensure you have a GitLab personal access token:

```bash
# Create token file
echo 'your-gitlab-token' > ~/.gitlab-token
chmod 600 ~/.gitlab-token

# Run authentication setup (optional, for additional config)
./setup_cluster_auth.sh
```

### Step 2: Submit the Job

```bash
sbatch cluster_with_auth.sh
```

### Step 3: Monitor the Job

```bash
# Check job status
squeue -u $USER

# View output logs
tail -f sbatch_logs/multi/*.out

# View error logs
tail -f sbatch_logs/multi/*.err
```

## Customization

### Changing the Number of Nodes
Edit line 6 in `cluster_with_auth.sh`:
```bash
#SBATCH -N 4  # for 4 nodes instead of 1
```

### Changing the Job Name
Edit line 3:
```bash
#SBATCH -J nvr_asicvlsi_quantum:your_job_name
```

### Modifying the Run Command
Edit the `RUN_CMD` variable (line 30):
```bash
RUN_CMD="python -u src/swarm/training/train.py --your-args"
```

### Adding Command Arguments
You can pass arguments to your training script:
```bash
RUN_CMD="python -u src/swarm/training/train.py exp_tag=6 model.model_type=combined"
```

## Filesystem Mounts

The script mounts:
- Current working directory to the same path in container
- `/tmp` directory for temporary files

To add more mounts, modify the `_cont_mounts` variable:
```bash
readonly _cont_mounts="${_code_root}:${_code_root},/tmp:/tmp,/data:/data"
```

## Environment Variables

The following environment variables are automatically set:
- `WORLD_SIZE`: Total number of GPUs across all nodes
- `WORLD_RANK`: Rank of current process
- `HDF5_USE_FILE_LOCKING`: Set to FALSE
- `CUDNN_V8_API_ENABLED`: Set to 1
- `OMP_NUM_THREADS`: Set to number of CPUs per node
- `PYTHONPATH`: Includes the code root directory

## Troubleshooting

### Authentication Issues
If you encounter authentication errors:
1. Verify your GitLab token is valid
2. Run `./setup_cluster_auth.sh` to set up authentication
3. Check if the cluster requires specific authentication methods

### Container Not Found
If the container image cannot be pulled:
1. Verify the image path is correct
2. Ensure you have access to the GitLab registry
3. Check if the cluster can reach the registry

### Path Issues
If files are not found in the container:
1. Verify the mount paths are correct
2. Use absolute paths when possible
3. Check that the working directory is set correctly

## Notes

- The script uses SLURM's Singularity/Apptainer backend for containers
- Authentication may be handled differently than with Docker
- Some Docker-specific features may not be available
- The `ldconfig` command updates the dynamic linker cache in the container
