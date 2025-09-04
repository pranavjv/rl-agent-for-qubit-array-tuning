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

import logging
import yaml

# Memory monitoring imports
import time
from pathlib import Path

import ray
import wandb
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

# Set logging level to reduce verbosity
logging.getLogger("ray").setLevel(logging.WARNING)
logging.getLogger("ray.tune").setLevel(logging.WARNING)
logging.getLogger("ray.rllib").setLevel(logging.WARNING)

# Add current directory to path for imports
current_dir = Path(__file__).parent
swarm_package_dir = current_dir.parent  # Get swarm package directory (/path/to/swarm)
swarm_src_dir = swarm_package_dir.parent  # Get src directory (/path/to/src)
swarm_root_dir = swarm_src_dir.parent  # Get Swarm root directory
sys.path.append(str(swarm_src_dir))

from swarm.training.utils.metrics_logger import (  # noqa: E402
    log_to_wandb,
    print_training_progress,
    setup_wandb_metrics,
    upload_checkpoint_artifact,
)
from swarm.training.utils.policy_mapping import policy_mapping_fn  # noqa: E402


def create_env(config=None):
    """Create multi-agent quantum environment."""
    from swarm.environment.multi_agent_wrapper import MultiAgentQuantumWrapper

    # Wrap in multi-agent wrapper (config unused but required by RLlib)
    return MultiAgentQuantumWrapper(training=True)


def load_config():
    """Load training configuration from YAML file."""
    config_path = Path(__file__).parent / "training_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    """Main training function using Ray RLlib 2.49.0 modern API with wandb logging."""

    config = load_config()

    # Initialize Weights & Biases
    wandb.init(
        entity=config['wandb']['entity'], 
        project=config['wandb']['project'], 
        config=config
    )
    setup_wandb_metrics()

    # Initialize Ray with runtime environment from config
    ray_config = {
        "include_dashboard": config['ray']['include_dashboard'],
        "log_to_driver": config['ray']['log_to_driver'],
        "logging_level": config['ray']['logging_level'],
        "runtime_env": {
            "working_dir": str(swarm_src_dir),
            "excludes": config['ray']['runtime_env']['excludes'],
            "env_vars": {
                **config['ray']['runtime_env']['env_vars'],
                "SWARM_PROJECT_ROOT": str(swarm_root_dir),
            },
        },
    }

    print("\nInitialising ray...\n")
    ray.init(**ray_config)

    try:
        register_env("qarray_multiagent_env", create_env)
        env_instance = create_env()

        from swarm.voltage_model import create_rl_module_spec
        
        rl_module_spec = create_rl_module_spec(env_instance)

        ppo_config = (
            PPOConfig()
            .environment(
                env="qarray_multiagent_env",
            )
            .multi_agent(
                policy_mapping_fn=policy_mapping_fn,
                policies=config['ppo']['multi_agent']['policies'],
                policies_to_train=config['ppo']['multi_agent']['policies_to_train'],
                count_steps_by=config['ppo']['multi_agent']['count_steps_by'],
            )
            .rl_module(
                rl_module_spec=rl_module_spec,
            )
            .env_runners(
                num_env_runners=config['ppo']['env_runners']['num_env_runners'],
                rollout_fragment_length=config['ppo']['env_runners']['rollout_fragment_length'],
                sample_timeout_s=config['ppo']['env_runners']['sample_timeout_s'],
                num_gpus_per_env_runner=config['ppo']['env_runners']['num_gpus_per_env_runner'],
            )
            .learners(
                num_learners=config['ppo']['learners']['num_learners'], 
                num_gpus_per_learner=config['ppo']['learners']['num_gpus_per_learner']
            )
            .training(
                train_batch_size=config['ppo']['training']['train_batch_size'],
                minibatch_size=config['ppo']['training']['minibatch_size'],
                lr=config['ppo']['training']['lr'],
                gamma=config['ppo']['training']['gamma'],
                lambda_=config['ppo']['training']['lambda_'],
                clip_param=config['ppo']['training']['clip_param'],
                entropy_coeff=config['ppo']['training']['entropy_coeff'],
                vf_loss_coeff=config['ppo']['training']['vf_loss_coeff'],
                num_epochs=config['ppo']['training']['num_epochs'],
            )
            .resources(num_gpus=config['resources']['num_gpus'])
        )

        # Build the algorithm
        print("\nBuilding PPO algorithm...\n")

        algo = ppo_config.build()  # creates a PPO object

        # Clean up the environment instance used for spec creation
        env_instance.close()
        del env_instance

        print(f"\nStarting training for {config['defaults']['num_iterations']} iterations...\n")

        training_start_time = time.time()
        best_reward = float("-inf")  # Track best performance for artifact upload

        for i in range(config['defaults']['num_iterations']):
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

        wandb.finish()
        print("Wandb session finished")


if __name__ == "__main__":
    main()
