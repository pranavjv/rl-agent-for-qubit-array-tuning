#!/bin/bash

# Script to securely store GitLab token
# DO NOT commit this script after running it!

echo "GitLab Token Setup"
echo "=================="
echo ""
echo "This script will help you securely store your GitLab token."
echo "The token will be stored in ~/.gitlab-token with restricted permissions."
echo ""

# Check if token file already exists
if [ -f "$HOME/.gitlab-token" ]; then
    echo "WARNING: Token file already exists at ~/.gitlab-token"
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting without changes."
        exit 0
    fi
fi

# Prompt for token
echo "Please enter your GitLab personal access token:"
read -s GITLAB_TOKEN

# Validate token is not empty
if [ -z "$GITLAB_TOKEN" ]; then
    echo "ERROR: Token cannot be empty"
    exit 1
fi

# Store token securely
echo "$GITLAB_TOKEN" > "$HOME/.gitlab-token"
chmod 600 "$HOME/.gitlab-token"

echo ""
echo "✓ Token stored successfully in ~/.gitlab-token"
echo "✓ Permissions set to 600 (read/write for owner only)"
echo ""
echo "You can now use the cluster scripts that require authentication."
echo ""
echo "To test authentication, run:"
echo "  docker login gitlab-master.nvidia.com -u gitlab-ci-token -p \$(cat ~/.gitlab-token)"
echo ""
echo "IMPORTANT: Never share your token or commit it to version control!"
