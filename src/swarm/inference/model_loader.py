#!/usr/bin/env python3
"""
Model loader for quantum device tuning RL agents.
"""
import sys
import numpy as np
import torch
from pathlib import Path
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
import ray

# Add src directory to path for clean imports
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def create_env(config=None):
    """Create multi-agent quantum environment."""
    from swarm.environment.multi_agent_wrapper import MultiAgentEnvWrapper
    return MultiAgentEnvWrapper(training=False)


def load_model(checkpoint_path=None):
    """
    Load trained RL model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint directory (defaults to latest)
    
    Returns:
        RLlib Algorithm instance
    """
    if checkpoint_path is None:
        checkpoints_dir = Path(__file__).parent.parent / "training" / "checkpoints"
        iteration_dirs = [d for d in checkpoints_dir.iterdir() 
                         if d.is_dir() and d.name.startswith("iteration_")]
        if not iteration_dirs:
            raise FileNotFoundError("No checkpoints found")
        iteration_dirs.sort(key=lambda x: int(x.name.split("_")[1]))
        checkpoint_path = iteration_dirs[-1]
    
    if not ray.is_initialized():
        # Use same runtime environment pattern as train.py
        ray_config = {
            "include_dashboard": False,
            "log_to_driver": False,
            "logging_level": 30,
            "runtime_env": {
                "working_dir": str(src_dir),
                "excludes": ["dataset",
                             "dataset_v1",
                             "wandb",
                             "outputs",
                             "test_outputs",
                             "checkpoints",
                             "weights*",
                             "*dataset*"],
                "env_vars": {
                    "JAX_PLATFORM_NAME": "cuda",
                    "JAX_PLATFORMS": "cuda",
                    "PYTHONWARNINGS": "ignore::DeprecationWarning",
                    "RAY_DEDUP_LOGS": "0",
                    "RAY_DISABLE_IMPORT_WARNING": "1",
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",

                }
            }
        }
        ray.init(**ray_config)
    
    # Register environment before loading checkpoint
    register_env("qarray_multiagent_env", create_env)
    
    return Algorithm.from_checkpoint(str(Path(checkpoint_path).absolute()))


def run_inference(algo, deterministic=True):
    """Run inference example with loaded model."""
    env = create_env()
    try:
        obs, _ = env.reset()
        
        # Multi-agent case using new RLModule API
        actions = {}
        for agent_id, agent_obs in obs.items():
            policy_id = f"{agent_id.split('_')[0]}_policy"
            
            # Get RLModule and compute action
            rl_module = algo.get_module(policy_id)
            obs_tensor = torch.from_numpy(agent_obs).unsqueeze(0).float()
            result = rl_module.forward_inference({"obs": obs_tensor})
            
            # Extract continuous action from distribution inputs
            action_dist_inputs = result["action_dist_inputs"][0]
            action_dim = action_dist_inputs.shape[0] // 2
            mean = action_dist_inputs[:action_dim]
            log_std = action_dist_inputs[action_dim:]
            
            # Sample from the distribution
            if deterministic:
                action = mean
            else:
                std = torch.exp(log_std)
                action = torch.normal(mean, std)
            action = torch.clamp(action, -1.0, 1.0)
            actions[agent_id] = action.item()
        
        return actions
    finally:
        env.close()


if __name__ == "__main__":
    try:
        algo = load_model()
        actions = run_inference(algo)
        print(f"Actions: {actions}")
    finally:
        if ray.is_initialized():
            ray.shutdown()