
from ray.rllib.algorithms.algorithm import Algorithm
import numpy as np
import os
import ray

# Initialize Ray first
ray.init()

# Load checkpoint with absolute path
checkpoint_path = "/home/rahul/Summer2025/rl-agent-for-qubit-array-tuning/src/swarm/training/rich-valley-141"
print(f"Loading checkpoint from: {checkpoint_path}")
algo = Algorithm.from_checkpoint(checkpoint_path)

# Run inference on environment
env = algo.get_policy().env  # Get environment
obs = env.reset()
action = algo.compute_single_action(obs)

print(action)