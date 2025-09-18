#!/usr/bin/env python3
"""
Simplified multi-agent RL training for quantum device tuning using Ray RLlib 2.49.0.
Enhanced with comprehensive memory usage logging.
"""
import os
import sys

# Suppress Ray warnings and verbose output
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
os.environ["RAY_DEDUP_LOGS"] = "0"

import argparse
import glob
import logging
import re
import yaml
from functools import partial

# Memory monitoring imports
import time
from pathlib import Path

import ray
import wandb
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig
from ray.tune.registry import register_env

# Set logging level to reduce verbosity
logging.getLogger("ray").setLevel(logging.WARNING)
logging.getLogger("ray.tune").setLevel(logging.WARNING)
logging.getLogger("ray.rllib").setLevel(logging.WARNING)

# Add src directory to path for clean imports
current_dir = Path(__file__).parent
swarm_package_dir = current_dir.parent  # swarm package directory
src_dir = swarm_package_dir.parent  # src directory
project_root = src_dir.parent  # project root directory
sys.path.insert(0, str(src_dir))

from swarm.training.utils import (  # noqa: E402
    log_to_wandb,
    print_training_progress,
    setup_wandb_metrics,
    upload_checkpoint_artifact,
    log_scans_to_wandb,
    policy_mapping_fn,
    CustomCallbacks,
)

from swarm.voltage_model import create_rl_module_spec


def parse_config_overrides(unknown_args):
    """Parse config override arguments in the format --key.subkey value (allows dynamically overriding settings when calling train.py)"""
    overrides = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith('--') and i + 1 < len(unknown_args):
            key = arg[2:]  # Remove '--' prefix
            value = unknown_args[i + 1]
            
            try:
                # Try int first
                if '.' not in value:
                    try:
                        value = int(value)
                    except ValueError:
                        # Try float
                        try:
                            value = float(value)
                        except ValueError:
                            # Try boolean
                            if value.lower() in ('true', 'false'):
                                value = value.lower() == 'true'
                            # Otherwise keep as string
                else:
                    # Has decimal point, try float
                    try:
                        value = float(value)
                    except ValueError:
                        # Keep as string if not a number
                        pass
            except ValueError:
                pass  # Keep as string
                
            overrides[key] = value
            i += 2
        else:
            i += 1
    return overrides


def apply_config_overrides(config, overrides):
    """Apply config overrides using dot notation to nested dictionary."""
    for key, value in overrides.items():
        keys = key.split('.')
        current = config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the final value
        current[keys[-1]] = value
        print(f"Config override: {key} = {value}")
    
    return config


def find_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint in the given directory.
    
    Args:
        checkpoint_dir (str or Path): Directory containing checkpoint folders
        
    Returns:
        tuple: (checkpoint_path, iteration_number) or (None, None) if no checkpoints found
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None, None
    
    # Find all iteration directories
    iteration_pattern = checkpoint_dir / "iteration_*"
    iteration_dirs = glob.glob(str(iteration_pattern))
    
    if not iteration_dirs:
        return None, None
    
    # Extract iteration numbers and find the maximum
    max_iteration = 0
    latest_checkpoint = None
    
    for iteration_dir in iteration_dirs:
        # Extract iteration number from directory name
        match = re.search(r'iteration_(\d+)', iteration_dir)
        if match:
            iteration_num = int(match.group(1))
            if iteration_num > max_iteration:
                max_iteration = iteration_num
                latest_checkpoint = iteration_dir
    
    return latest_checkpoint, max_iteration


def parse_arguments():
    """Parse command line arguments for checkpoint loading and config overrides."""
    parser = argparse.ArgumentParser(description='Multi-agent RL training for quantum device tuning')
    
    parser.add_argument(
        '--load-checkpoint', 
        type=str, 
        help='Path to specific checkpoint directory to load'
    )
    
    parser.add_argument(
        '--resume-latest', 
        action='store_true',
        help='Resume training from the most recent checkpoint in the default checkpoints directory'
    )

    parser.add_argument(
        '--disable-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    
    # Parse known args to allow for dynamic config overrides
    args, unknown = parser.parse_known_args()
    
    # Parse config overrides from remaining arguments
    config_overrides = parse_config_overrides(unknown)
    args.config_overrides = config_overrides
    
    return args



def create_env(config=None):
    """Create multi-agent quantum environment."""
    from swarm.environment.multi_agent_wrapper import MultiAgentEnvWrapper

    # Wrap in multi-agent wrapper (config unused but required by RLlib)
    return MultiAgentEnvWrapper()


def load_config():
    """Load training configuration from YAML file."""
    config_path = Path(__file__).parent / "training_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    """Main training function using Ray RLlib 2.49.0 API with wandb logging."""

    # Parse command line arguments
    args = parse_arguments()

    use_wandb = not args.disable_wandb
    
    config = load_config()
    
    # Apply command line overrides to config
    if hasattr(args, 'config_overrides') and args.config_overrides:
        config = apply_config_overrides(config, args.config_overrides)

    # Initialize Weights & Biases
    if use_wandb:
        wandb.init(
            entity=config['wandb']['entity'], 
            project=config['wandb']['project']
        )
        # Log the final config (with command line overrides) to wandb
        wandb.config.update(config)
        setup_wandb_metrics()

    # Initialize Ray with runtime environment from config
    ray_config = {
        "include_dashboard": config['ray']['include_dashboard'],
        "log_to_driver": config['ray']['log_to_driver'],
        "logging_level": config['ray']['logging_level'],
        "runtime_env": {
            "working_dir": str(src_dir),
            "excludes": config['ray']['runtime_env']['excludes'],
            "env_vars": {
                **config['ray']['runtime_env']['env_vars'],
                "SWARM_PROJECT_ROOT": str(project_root),
            },
        },
    }

    print("\nInitialising ray...\n")
    ray.init(**ray_config)

    try:
        register_env("qarray_multiagent_env", create_env)
        env_instance = create_env()

        # Optionally update the rl module config to allow log_std clamping, shared log_std vector etc.
        rl_module_config = {
            "plunger_policy": {
                "free_log_std": config['rl_config']['multi_agent']['free_log_std'],
                "log_std_bounds": config['rl_config']['multi_agent']['log_std_bounds'],
            },
            "barrier_policy": {
                "free_log_std": config['rl_config']['multi_agent']['free_log_std'],
                "log_std_bounds": config['rl_config']['multi_agent']['log_std_bounds'],
            }
        }
        
        algo = config['rl_config']['algorithm'].lower()

        rl_module_spec = create_rl_module_spec(env_instance, algo=algo, config=rl_module_config)

        # Configure custom callbacks for logging to Wandb
        # log_images = config['wandb']['log_images']
        # custom_callbacks = partial(CustomCallbacks, log_images=log_images)

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
                num_env_runners=config['rl_config']['env_runners']['num_env_runners'],
                rollout_fragment_length=config['rl_config']['env_runners']['rollout_fragment_length'],
                sample_timeout_s=config['rl_config']['env_runners']['sample_timeout_s'],
                num_gpus_per_env_runner=config['rl_config']['env_runners']['num_gpus_per_env_runner'],
            )
            .learners(
                num_learners=config['rl_config']['learners']['num_learners'], 
                num_gpus_per_learner=config['rl_config']['learners']['num_gpus_per_learner']
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
                grad_clip=config['rl_config']['training']['grad_clip'],
                grad_clip_by=config['rl_config']['training']['grad_clip_by'],
            )
            .resources(num_gpus=config['resources']['num_gpus'])
            # .callbacks([custom_callbacks] if use_wandb else [])
        )

        # Build the algorithm
        print("\nBuilding PPO algorithm...\n")

        algo = ppo_config.build()  # creates a PPO object

        # Clean up the environment instance used for spec creation
        env_instance.close()
        del env_instance

        # Handle checkpoint loading if requested
        start_iteration = 0
        checkpoint_loaded = False
        
        if args.load_checkpoint:
            # Load specific checkpoint
            checkpoint_path = Path(args.load_checkpoint)
            if checkpoint_path.exists():
                print(f"\nLoading checkpoint from: {checkpoint_path}")
                try:
                    algo.restore_from_path(str(checkpoint_path))
                    
                    # Extract iteration number from path
                    match = re.search(r'iteration_(\d+)', str(checkpoint_path))
                    if match:
                        start_iteration = int(match.group(1))
                        checkpoint_loaded = True
                        print(f"Checkpoint loaded successfully. Resuming from iteration {start_iteration + 1}")
                    else:
                        print("Warning: Could not determine iteration number from checkpoint path")
                        
                except Exception as e:
                    print(f"Error loading checkpoint: {e}")
                    print("Continuing with fresh training...")
            else:
                print(f"Error: Checkpoint path does not exist: {checkpoint_path}")
                print("Continuing with fresh training...")
                
        elif args.resume_latest:
            # Find and load most recent checkpoint
            checkpoint_dir = Path(config['checkpoints']['save_dir'])
            latest_checkpoint, latest_iteration = find_latest_checkpoint(checkpoint_dir)
            
            if latest_checkpoint:
                print(f"\nFound latest checkpoint: {latest_checkpoint} (iteration {latest_iteration})")
                try:
                    algo.restore_from_path(str(latest_checkpoint))
                    start_iteration = latest_iteration
                    checkpoint_loaded = True
                    print(f"Latest checkpoint loaded successfully. Resuming from iteration {start_iteration + 1}")
                except Exception as e:
                    print(f"Error loading latest checkpoint: {e}")
                    print("Continuing with fresh training...")
            else:
                print("\nNo checkpoints found in checkpoint directory.")
                print("Starting fresh training...")
        
        if not checkpoint_loaded:
            print("\nStarting fresh training from iteration 1...")
        else:
            print(f"Training will continue from iteration {start_iteration + 1} to {config['defaults']['num_iterations']}")
            # Validate that we haven't already completed training
            if start_iteration >= config['defaults']['num_iterations']:
                print(f"Training already completed! Loaded checkpoint is at iteration {start_iteration}, "
                      f"but max iterations is {config['defaults']['num_iterations']}.")
                return

        # Save training config to checkpoint directory for easy reference
        checkpoint_base_dir = Path(config['checkpoints']['save_dir'])
        checkpoint_base_dir.mkdir(parents=True, exist_ok=True)
        config_save_path = checkpoint_base_dir / "training_config.yaml"
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        remaining_iterations = config['defaults']['num_iterations'] - start_iteration
        print(f"\nStarting training for {remaining_iterations} iterations (from iteration {start_iteration + 1} to {config['defaults']['num_iterations']})...\n")
        print(f"Training config saved to: {config_save_path}\n")

        training_start_time = time.time()
        best_reward = float("-inf")  # Track best performance for artifact upload

        for i in range(start_iteration, config['defaults']['num_iterations']):
            result = algo.train()

            # Clean console output and wandb logging
            print_training_progress(result, i, training_start_time)

            # Log metrics to wandb
            log_to_wandb(result, i)

            # Save checkpoint using modern RLlib API
            local_checkpoint_dir = Path(config['checkpoints']['save_dir']) / f"iteration_{i+1}"
            local_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = algo.save_to_path(str(local_checkpoint_dir.absolute()))

            # Upload checkpoint as wandb artifact if performance improved
            if config['checkpoints']['upload_best_only']:
                current_reward = result.get("env_runners", {}).get(
                    "episode_return_mean", float("-inf")
                )
                if current_reward is not None and current_reward > best_reward:
                    best_reward = current_reward
                    upload_checkpoint_artifact(checkpoint_path, i + 1, current_reward)
            else:
                upload_checkpoint_artifact(checkpoint_path, i + 1, 0.0)

            print(f"\nIteration {i+1} completed. Checkpoint saved to: {checkpoint_path}\n")

    finally:
        if ray.is_initialized():
            ray.shutdown()

        if use_wandb:
            wandb.finish()
            print("Wandb session finished")


if __name__ == "__main__":
    main()
