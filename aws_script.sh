#!/bin/bash
# ==== User Data Script for Spot Instance Training ====

# Fail on error
set -e

# Update + basic tools
yum update -y
yum install -y awscli git

# Set variables
BUCKET="s3://my-ml-checkpoints"
CODE_BUCKET="s3://my-ml-code"
WORKDIR="/home/ec2-user/training"
CHECKPOINT_DIR="$WORKDIR/checkpoints"

mkdir -p $WORKDIR
cd $WORKDIR

# Pull code and checkpoints
aws s3 sync $CODE_BUCKET $WORKDIR/code
aws s3 sync $BUCKET $CHECKPOINT_DIR

# Find latest checkpoint if exists
LATEST_CKPT=$(ls -t $CHECKPOINT_DIR/*.ckpt 2>/dev/null | head -n1 || true)

# Start training (resume if checkpoint exists)
if [ -n "$LATEST_CKPT" ]; then
  echo "Resuming from checkpoint: $LATEST_CKPT"
  nohup bash $WORKDIR/code/run_training.sh --resume-from $LATEST_CKPT > train.log 2>&1 &
else
  echo "Starting fresh training run"
  nohup bash $WORKDIR/code/run_training.sh > train.log 2>&1 &
fi

# Background job to periodically sync checkpoints back to S3
(
  while true; do
    sleep 900  # every 15 minutes
    echo "Syncing checkpoints..."
    aws s3 sync $CHECKPOINT_DIR $BUCKET
  done
) &
