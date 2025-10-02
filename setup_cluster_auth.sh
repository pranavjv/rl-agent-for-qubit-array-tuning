#!/bin/bash
# Helper script to set up GitLab authentication for SLURM container jobs

echo "Setting up GitLab authentication for cluster containers..."

# Check if GitLab token exists
if [ ! -f "$HOME/.gitlab-token" ]; then
    echo "Error: GitLab token not found at $HOME/.gitlab-token"
    echo "Please create this file with your GitLab personal access token"
    echo "Example: echo 'your-token-here' > ~/.gitlab-token && chmod 600 ~/.gitlab-token"
    exit 1
fi

GITLAB_TOKEN=$(cat "$HOME/.gitlab-token")
GITLAB_REGISTRY="gitlab-master.nvidia.com"

# Option 1: Set up Singularity/Apptainer authentication
if command -v singularity &> /dev/null; then
    echo "Setting up Singularity authentication..."
    export SINGULARITY_DOCKER_USERNAME="gitlab-ci-token"
    export SINGULARITY_DOCKER_PASSWORD="${GITLAB_TOKEN}"
    echo "Singularity authentication variables exported"
elif command -v apptainer &> /dev/null; then
    echo "Setting up Apptainer authentication..."
    export APPTAINER_DOCKER_USERNAME="gitlab-ci-token"
    export APPTAINER_DOCKER_PASSWORD="${GITLAB_TOKEN}"
    echo "Apptainer authentication variables exported"
fi

# Option 2: Create Docker config if needed
DOCKER_CONFIG_DIR="$HOME/.docker"
DOCKER_CONFIG_FILE="$DOCKER_CONFIG_DIR/config.json"

if [ ! -d "$DOCKER_CONFIG_DIR" ]; then
    mkdir -p "$DOCKER_CONFIG_DIR"
    chmod 700 "$DOCKER_CONFIG_DIR"
fi

# Generate base64 encoded auth string
AUTH_STRING=$(echo -n "gitlab-ci-token:${GITLAB_TOKEN}" | base64)

# Create Docker config.json
cat > "$DOCKER_CONFIG_FILE" << EOF
{
    "auths": {
        "${GITLAB_REGISTRY}": {
            "auth": "${AUTH_STRING}"
        }
    }
}
EOF

chmod 600 "$DOCKER_CONFIG_FILE"
echo "Docker config created at $DOCKER_CONFIG_FILE"

echo ""
echo "Authentication setup complete!"
echo "You can now submit your SLURM job with: sbatch cluster_with_auth.sh"
echo ""
echo "Note: You may need to run this script before each session or add the exports to your ~/.bashrc"
