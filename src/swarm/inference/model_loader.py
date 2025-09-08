
#!/usr/bin/env python3
"""
Model loader for quantum device tuning RL agents.
"""
import sys
from pathlib import Path
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
import ray

# Add src directory to path for clean imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
sys.path.insert(0, str(src_dir))


def create_env():
    """Create multi-agent quantum environment."""
    from swarm.environment.multi_agent_wrapper import MultiAgentQuantumWrapper
    return MultiAgentQuantumWrapper(training=False)


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
        ray.init(include_dashboard=False, log_to_driver=False)
    
    # Register environment before loading checkpoint
    register_env("qarray_multiagent_env", create_env)
    
    return Algorithm.from_checkpoint(str(Path(checkpoint_path).absolute()))


def run_inference(algo):
    """Run inference example with loaded model."""
    env = create_env()
    try:
        obs, _ = env.reset()
        if isinstance(obs, dict):
            actions = {agent_id: algo.compute_single_action(agent_obs, 
                      policy_id=f"{agent_id.split('_')[0]}_policy") 
                      for agent_id, agent_obs in obs.items()}
        else:
            actions = algo.compute_single_action(obs)
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
