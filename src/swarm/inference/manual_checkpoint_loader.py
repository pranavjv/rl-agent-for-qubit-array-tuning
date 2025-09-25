#!/usr/bin/env python3
"""
Manual checkpoint loader that reconstructs the PPO config and loads model weights.
"""
import sys
import yaml
from pathlib import Path
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

# Add src directory to path for clean imports  
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
sys.path.insert(0, str(src_dir))

def create_env(config=None):
    """Create multi-agent quantum environment."""
    from swarm.environment.multi_agent_wrapper import MultiAgentQuantumWrapper
    return MultiAgentQuantumWrapper(training=False)

def load_training_config():
    """Load training configuration from YAML file."""
    config_path = Path(__file__).parent.parent / "training" / "training_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """Map agents to their respective policies."""
    if agent_id.startswith("plunger"):
        return "plunger_policy"
    elif agent_id.startswith("barrier"):
        return "barrier_policy"
    else:
        raise ValueError(f"Unknown agent ID: {agent_id}")

def create_algo_config():
    """Create PPO algorithm configuration matching training setup."""
    config = load_training_config()
    
    # Create environment instance to get RL module spec
    from swarm.voltage_model import create_rl_module_spec
    env_instance = create_env()
    rl_module_spec = create_rl_module_spec(env_instance)
    
    ppo_config = (
        PPOConfig()
        .environment(
            env="qarray_multiagent_env",
        )
        .multi_agent(
            policy_mapping_fn=policy_mapping_fn,
            policies=config['rl_config']['multi_agent']['policies'],
            policies_to_train=config['rl_config']['multi_agent']['policies_to_train'],
            count_steps_by=config['rl_config']['multi_agent']['count_steps_by'],
        )
        .rl_module(
            rl_module_spec=rl_module_spec,
        )
        .env_runners(
            num_env_runners=0,  # Set to 0 for inference
            rollout_fragment_length=config['rl_config']['env_runners']['rollout_fragment_length'],
            sample_timeout_s=config['rl_config']['env_runners']['sample_timeout_s'],
            num_gpus_per_env_runner=0,  # No GPUs for env runners in inference
        )
        .learners(
            num_learners=0,  # Set to 0 for inference
            num_gpus_per_learner=0  # No GPUs for learners in inference
        )
        .training(
            train_batch_size=config['rl_config']['training']['train_batch_size'],
            minibatch_size=config['rl_config']['training']['minibatch_size'],
            lr=config['rl_config']['training']['lr'],
            gamma=config['rl_config']['training']['gamma'],
            lambda_=config['rl_config']['training']['lambda_'],
            clip_param=config['rl_config']['training']['clip_param'],
            entropy_coeff=config['rl_config']['training']['entropy_coeff'],
            vf_loss_coeff=config['rl_config']['training']['vf_loss_coeff'],
            num_epochs=config['rl_config']['training']['num_epochs'],
        )
        .resources(num_gpus=0)  # No GPUs for inference
    )
    
    # Clean up the environment instance
    env_instance.close()
    del env_instance
    
    return ppo_config

def load_model_manual(checkpoint_path=None):
    """
    Manually load model by recreating config and loading weights.
    
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
    
    # Register environment
    register_env("qarray_multiagent_env", create_env)
    
    # Create algorithm configuration
    print("Creating algorithm configuration...")
    ppo_config = create_algo_config()
    
    # Build algorithm with the configuration
    print("Building algorithm...")
    algo = ppo_config.build()
    
    # Try to load weights from checkpoint
    print(f"Loading weights from checkpoint: {checkpoint_path}")
    try:
        algo.restore(str(Path(checkpoint_path).absolute()))
        print("Successfully loaded model weights!")
        return algo
    except Exception as e:
        print(f"Error loading weights: {e}")
        # Return the algorithm with random weights if loading fails
        print("Returning algorithm with random weights...")
        return algo

if __name__ == "__main__":
    try:
        algo = load_model_manual()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if ray.is_initialized():
            ray.shutdown()