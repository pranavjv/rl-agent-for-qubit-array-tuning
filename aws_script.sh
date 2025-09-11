#!/bin/bash
set -e

# Configuration
BUCKET="s3://rahul-ml-checkpoints/checkpoints"
CODE_REPO="https://github.com/edwindn/rl-agent-for-qubit-array-tuning.git"
CODE_DIR="/home/ec2-user/rl-agent"
CHECKPOINT_DIR="$CODE_DIR/src/swarm/training/checkpoints"

# Install prerequisites
yum update -y
yum install -y awscli git python3 python3-pip

# Setup working directory
mkdir -p $CODE_DIR
cd $CODE_DIR

# Clone or update code
if [ ! -d ".git" ]; then
  git clone $CODE_REPO .
else
  git pull origin main
fi

pip3 install --upgrade pip
if [ -f "requirements.txt" ]; then
    pip3 install -r requirements.txt
fi

# restores checkpoints from S3
mkdir -p $CHECKPOINT_DIR
aws s3 sync $BUCKET $CHECKPOINT_DIR

# starts training (auto-resume latest)
cd $CODE_DIR/src/swarm/training

nohup python3 train.py --resume-latest > training.log 2>&1 &

# syncs checkpoints
(
  while true; do
    sleep 600   # every 10 minutes
    echo "$(date): Syncing checkpoints to S3"
    aws s3 sync $CHECKPOINT_DIR $BUCKET
  done
) &

# checks for spot termination flag
(
  while true; do
    if curl -s http://169.254.169.254/latest/meta-data/spot/termination-time; then
      echo "$(date): Spot termination notice received"
      # Final checkpoint sync
      aws s3 sync $CHECKPOINT_DIR $BUCKET
      exit 0
    fi
    sleep 5
  done
) &
