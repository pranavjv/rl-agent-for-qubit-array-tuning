# Container Import Guide for SLURM Clusters

## Problem
When using SLURM with pyxis/enroot, you may encounter authentication issues when trying to pull private Docker images from GitLab. The error typically shows:
- Authenticating as `<anonymous>` user
- Trying to pull from Docker Hub instead of GitLab
- 401 Unauthorized errors

## Solution: Pre-import Containers

Instead of pulling images at runtime, pre-import them into squashfs (.sqsh) files.

## Steps

### 1. Set up GitLab Token
```bash
# Create token file with your GitLab personal access token
echo 'your-gitlab-token' > ~/.gitlab-token
chmod 600 ~/.gitlab-token
```

### 2. Import the Container
Run one of the import scripts:

```bash
# Basic import (uses $HOME/containers/)
./import_container.sh

# OR cluster-aware import (auto-detects cluster paths)
./import_container_cluster.sh
```

The script will:
- Authenticate with GitLab using your token
- Pull the Docker image
- Convert it to a .sqsh file
- Save it to an appropriate location

### 3. Submit Your Job
```bash
sbatch cluster_with_auth.sh
```

The updated script will automatically:
- Look for the .sqsh file in common locations
- Use the local file instead of pulling from registry
- Fall back to registry pull if no local file is found

## Customization

### Custom Container Path
You can specify a custom path:
```bash
export CONTAINER_PATH=/path/to/your/container.sqsh
sbatch cluster_with_auth.sh
```

### Different Container Versions
To use a different container version:
1. Edit the import script to change the image tag
2. Re-run the import
3. Update the container name in `cluster_with_auth.sh` if needed

## Troubleshooting

### Import Fails
- Verify your GitLab token has registry read access
- Try running on a compute node: `srun --pty bash`
- Check if enroot is available: `which enroot`

### Container Not Found
- Check the import completed successfully
- Verify the path in `cluster_with_auth.sh` matches where the file was saved
- Look for the .sqsh file: `find /lustre /scratch $HOME -name "*.sqsh" 2>/dev/null`

### Still Getting Authentication Errors
If using the .sqsh file approach doesn't work:
1. Contact your cluster admin about the correct authentication method
2. Ask about cluster-specific container registry configurations
3. Check if there are cluster-specific modules to load: `module avail`
