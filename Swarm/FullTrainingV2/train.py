#!/usr/bin/env python3
"""
Simplified multi-agent RL training for quantum device tuning using Ray RLlib 2.49.0.
Enhanced with comprehensive memory usage logging.
"""
import os
import sys
from typing import Any, Optional

# Configure JAX to use CPU-only before any other imports
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.25'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import argparse
from pathlib import Path
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.tune.registry import register_env

# Memory monitoring imports
import psutil
import gc
import time
import logging
from datetime import datetime
import wandb

# Add current directory to path for imports
current_dir = Path(__file__).parent
swarm_dir = current_dir.parent  # Get Swarm directory
sys.path.append(str(swarm_dir))

from utils.policy_mapping import create_rl_module_spec
from utils.logging_utils import (
    log_memory_usage_wandb, memory_checkpoint_wandb, log_training_metrics_wandb,
    setup_memory_logging, log_memory_usage, memory_checkpoint
)
from metrics_utils import extract_training_metrics


def create_env(config=None):
    """Create multi-agent quantum environment."""
    from Environment.multi_agent_wrapper import MultiAgentQuantumWrapper

    num_quantum_dots = config["num_quantum_dots"]    
    # Wrap in multi-agent wrapper
    return MultiAgentQuantumWrapper(num_quantum_dots=num_quantum_dots, training=True)


def policy_mapping_fn(agent_id: str, episode=None, **kwargs) -> str:
    """Map agent IDs to policy IDs. Ray 2.49.0 passes agent_id and episode."""
    if agent_id.startswith("plunger") or "plunger" in agent_id.lower():
        return "plunger_policy"
    elif agent_id.startswith("barrier") or "barrier" in agent_id.lower():
        return "barrier_policy"
    else:
        raise ValueError(
            f"Agent ID '{agent_id}' must contain 'plunger' or 'barrier' to determine policy type. "
            f"Expected format: 'plunger_X' or 'barrier_X' where X is the agent number."
        )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train multi-agent RL for quantum device tuning")
    parser.add_argument("--num-quantum-dots", type=int, default=8, help="Number of quantum dots")
    parser.add_argument("--num-iterations", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--disable-wandb", action="store_true", help="Disable wandb logging")

    return parser.parse_args()


def main():
    """Main training function using Ray RLlib 2.49.0 modern API with wandb logging."""
    args = parse_arguments()
    
    # Initialize Weights & Biases if not disabled
    if not args.disable_wandb:
        run_name = f"qarray-{args.num_quantum_dots}dots-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb.init(
            entity="rl_agents_for_tuning",
            project="RLModel",
            name=run_name,
            config={
                "num_quantum_dots": args.num_quantum_dots,
                "num_iterations": args.num_iterations,
                "num_gpus": args.num_gpus,
            }
        )
        memory_checkpoint_wandb("STARTUP", "Training script started")
    else:
        print("Wandb logging disabled")
        # Still setup basic memory logging for console output
        memory_logger = setup_memory_logging()
        memory_checkpoint(memory_logger, "STARTUP", "Training script started")
    
    # Set environment variables for Ray
    os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'
    
    # Add VoltageAgent to path for custom RLModule
    sys.path.append(str(swarm_dir / "VoltageAgent"))
    
    # Initialize Ray with runtime environment
    ray_config = {
        "include_dashboard": False,
        "runtime_env": {
            "working_dir": str(swarm_dir),
            "py_modules": [
                str(swarm_dir / "Environment"),
                str(swarm_dir / "VoltageAgent")
            ],
            "env_vars": {
                "JAX_PLATFORM_NAME": "cpu",
                "JAX_PLATFORMS": "cpu",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.25",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false"
            }
        }
    }
    
    start_time = time.time()
    ray.init(**ray_config)
    ray_init_time = time.time() - start_time
    
    try:
        register_env("qarray_multiagent_env", create_env)
        env_instance = create_env({"num_quantum_dots": args.num_quantum_dots})

        rl_module_spec = create_rl_module_spec(env_instance)

        config = (
            PPOConfig()
            .environment(
                env="qarray_multiagent_env", 
                env_config={"num_quantum_dots": args.num_quantum_dots}
            )
            .multi_agent(
                policy_mapping_fn=policy_mapping_fn,
                policies=["plunger_policy", "barrier_policy"],
                policies_to_train=["plunger_policy", "barrier_policy"]
            )
            .rl_module(
                rl_module_spec=rl_module_spec,
            )
            .env_runners(
                num_env_runners=40,
                rollout_fragment_length='auto',
                sample_timeout_s=60.0,
            )
            .training(
                train_batch_size=64,
                minibatch_size=8,
                lr=3e-4,
                gamma=0.99,
                lambda_=0.95,
                clip_param=0.2,
                entropy_coeff=0.01,
                vf_loss_coeff=0.5,
                num_sgd_iter=4  # Fewer SGD iterations to speed up training
            )
            .resources(
                num_gpus=4
            )
        )

        # Build the algorithm
        print("\nBuilding PPO algorithm...\n")
        
        start_time = time.time()
        algo = config.build_algo() # creates a PPO object
        build_time = time.time() - start_time
        
        # Clean up the environment instance used for spec creation
        env_instance.close()
        del env_instance
        
        print(f"\nStarting training for {args.num_iterations} iterations...\n")
        
        for i in range(args.num_iterations):
            iteration_start_time = time.time()
            
            try:
                result = algo.train()
                iteration_time = time.time() - iteration_start_time
                print(result)
                
                # Extract focused training metrics
                metrics = extract_training_metrics(result, iteration_time)
                
                # Log metrics to wandb or console
                if not args.disable_wandb:
                    log_training_metrics_wandb(metrics, i)
                    # Log memory usage every iteration for the first 5, then every 10
                    if i < 5 or i % 10 == 0:
                        memory_checkpoint_wandb(f"ITERATION_{i}_COMPLETE", 
                                              f"Iteration {i} completed in {iteration_time:.2f} seconds", step=i)
                else:
                    # Fallback logging to console and file
                    if i < 5 or i % 10 == 0:
                        memory_checkpoint(memory_logger, f"ITERATION_{i}_COMPLETE", 
                                        f"Iteration {i} completed in {iteration_time:.2f} seconds")
                
                # Console output for all modes
                print(f"Iteration {i:3d}: {metrics['summary']}")
                
            except Exception as e:
                error_msg = f"Error in iteration {i}: {str(e)}"
                raise
        
        # Save final checkpoint
        checkpoint_path = algo.save()

        print(f"\nTraining completed. Checkpoint saved to: {checkpoint_path}\n")
        
    finally:
        if ray.is_initialized():
            ray.shutdown()
            
        if not args.disable_wandb:
            wandb.finish()
            print("Wandb session finished")
        else:
            memory_logger.info("=" * 100)
            memory_logger.info("TRAINING SESSION COMPLETED")
            memory_logger.info("=" * 100)


if __name__ == "__main__":
    main()