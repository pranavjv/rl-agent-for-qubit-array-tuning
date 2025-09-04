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
import logging

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
from swarm.training.utils.policy_mapping import create_rl_module_spec, policy_mapping_fn  # noqa: E402


def create_env(config=None):
    """Create multi-agent quantum environment."""
    from swarm.environment.multi_agent_wrapper import MultiAgentQuantumWrapper

    # Wrap in multi-agent wrapper (config unused but required by RLlib)
    return MultiAgentQuantumWrapper(training=True)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train multi-agent RL for quantum device tuning")
    parser.add_argument("--num-quantum-dots", type=int, default=8, help="Number of quantum dots")
    parser.add_argument(
        "--num-iterations", type=int, default=100, help="Number of training iterations"
    )
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable wandb logging")

    return parser.parse_args()


def main():
    """Main training function using Ray RLlib 2.49.0 modern API with wandb logging."""

    args = parse_arguments()

    # Initialize Weights & Biases if not disabled
    if not args.disable_wandb:
        wandb.init(entity="rl_agents_for_tuning", project="RLModel", config={})
        setup_wandb_metrics()
    else:
        print("Wandb logging disabled")

    # Initialize Ray with runtime environment
    ray_config = {
        "include_dashboard": False,
        "log_to_driver": False,  # Reduce driver logs
        "logging_level": logging.WARNING,  # Set Ray logging level
        "runtime_env": {
            "working_dir": str(swarm_src_dir),
            "excludes": ["dataset", "dataset_v1", "wandb", "outputs", "test_outputs", "checkpoints"],
            "env_vars": {
                "JAX_PLATFORM_NAME": "cuda", 
                "JAX_PLATFORMS": "cuda",
                "SWARM_PROJECT_ROOT": str(swarm_root_dir),
                "PYTHONWARNINGS": "ignore::DeprecationWarning",
                "RAY_DEDUP_LOGS": "0",
                "RAY_DISABLE_IMPORT_WARNING": "1",
            },
        },
    }

    print("\nInitialising ray...\n")
    ray.init(**ray_config)

    try:
        register_env("qarray_multiagent_env", create_env)
        env_instance = create_env()

        rl_module_spec = create_rl_module_spec(env_instance)

        config = (
            PPOConfig()
            .environment(
                env="qarray_multiagent_env",
            )
            .multi_agent(
                policy_mapping_fn=policy_mapping_fn,
                policies=["plunger_policy", "barrier_policy"],
                policies_to_train=["plunger_policy", "barrier_policy"],
                count_steps_by="agent_steps",
            )
            .rl_module(
                rl_module_spec=rl_module_spec,
            )
            .env_runners(
                num_env_runners=5,
                rollout_fragment_length=50,
                sample_timeout_s=600.0,
                num_gpus_per_env_runner=0.6,
            )
            .learners(num_learners=1, num_gpus_per_learner=0.75)
            .training(
                train_batch_size=4096,
                minibatch_size=64,
                lr=3e-4,
                gamma=0.99,
                lambda_=0.95,
                clip_param=0.2,
                entropy_coeff=0.01,
                vf_loss_coeff=0.5,
                num_epochs=10,
            )
            .resources(num_gpus=8)
        )

        # Build the algorithm
        print("\nBuilding PPO algorithm...\n")

        algo = config.build_algo()  # creates a PPO object

        # Clean up the environment instance used for spec creation
        env_instance.close()
        del env_instance

        print(f"\nStarting training for {args.num_iterations} iterations...\n")

        training_start_time = time.time()
        best_reward = float("-inf")  # Track best performance for artifact upload

        for i in range(args.num_iterations):
            result = algo.train()

            # Clean console output and wandb logging
            print_training_progress(result, i, training_start_time)

            # Log metrics to wandb
            if not args.disable_wandb:
                log_to_wandb(result, i)

            # Save checkpoint using modern RLlib API
            local_checkpoint_dir = Path("./checkpoints") / f"iteration_{i+1}"
            local_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = algo.save_to_path(str(local_checkpoint_dir.absolute()))

            # Upload checkpoint as wandb artifact if performance improved
            if not args.disable_wandb:
                current_reward = result.get("env_runners", {}).get(
                    "episode_return_mean", float("-inf")
                )
                if current_reward is not None and current_reward > best_reward:
                    best_reward = current_reward
                    upload_checkpoint_artifact(checkpoint_path, i + 1, current_reward)

            print(f"\nIteration {i+1} completed. Checkpoint saved to: {checkpoint_path}\n")

    finally:
        if ray.is_initialized():
            ray.shutdown()

        if not args.disable_wandb:
            wandb.finish()
            print("Wandb session finished")


if __name__ == "__main__":
    main()
