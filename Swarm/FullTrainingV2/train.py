#!/usr/bin/env python3
"""
Simplified multi-agent RL training for quantum device tuning using Ray RLlib 2.49.0.
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

# Add current directory to path for imports
current_dir = Path(__file__).parent
swarm_dir = current_dir.parent  # Get Swarm directory
sys.path.append(str(swarm_dir))

from utils.policy_mapping import create_rl_module_spec


def create_env(config=None):
    """Create multi-agent quantum environment."""
    from Environment.multi_agent_wrapper import MultiAgentQuantumWrapper
    from Environment.env import QuantumDeviceEnv

    num_quantum_dots = config["num_quantum_dots"]
    
    # Create base environment
    base_env = QuantumDeviceEnv(num_dots=num_quantum_dots, training=True)
    
    # Wrap in multi-agent wrapper
    return MultiAgentQuantumWrapper(base_env, num_quantum_dots=num_quantum_dots)


def policy_mapping_fn(agent_id: str, episode: Optional[Any] = None, worker: Optional[Any] = None, **kwargs) -> str:
    if agent_id.startswith("plunger") or "plunger" in agent_id.lower():
        return "plunger_policy"
    elif agent_id.startswith("barrier") or "barrier" in agent_id.lower():
        return "barrier_policy"
    else:
        raise ValueError(
            f"Agent ID '{agent_id}' must contain 'plunger' or 'barrier' to determine policy type. "
            f"Expected format: 'plunger_X' or 'barrier_X' where X is the agent number."
        )


def test_environment(num_quantum_dots):
    """Test the multi-agent environment setup."""
    print("Testing multi-agent environment...")
    
    env = create_env({"num_quantum_dots": num_quantum_dots})
    obs, info = env.reset()
    
    print(f"✓ Environment created with {len(env.get_agent_ids())} agents")
    print(f"  Agent IDs: {env.get_agent_ids()}")
    
    # Test step with random actions
    actions = {}
    for agent_id in env.get_agent_ids():
        actions[agent_id] = env.action_space[agent_id].sample()
    
    obs, rewards, terminated, truncated, info = env.step(actions)
    print(f"✓ Environment step successful")
    print(f"  Sample rewards: {dict(list(rewards.items())[:2])}")
    
    env.close()
    return True


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train multi-agent RL for quantum device tuning")
    
    parser.add_argument("--num-quantum-dots", type=int, default=8, help="Number of quantum dots")
    parser.add_argument("--num-iterations", type=int, default=100, help="Number of training iterations")
    parser.add_argument("--test-env", action="store_true", help="Test environment and exit")

    parser.add_argument("--num-gpus", type=int, default=1, help="Number of GPUs to use")

    return parser.parse_args()


def main():
    """Main training function using Ray RLlib 2.49.0 modern API."""
    args = parse_arguments()
    
    # Set environment variables for Ray
    os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'
    
    # Test environment if requested
    if args.test_env:
        success = test_environment(args.num_quantum_dots)
        sys.exit(0 if success else 1)
    
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
    ray.init(**ray_config)
    
    try:
        # Register environment
        register_env("quantum_multiagent", create_env)

        # Register RL module spec
        env_instance = create_env({"num_quantum_dots": args.num_quantum_dots})
        rl_module_spec = create_rl_module_spec(env_instance)
        
        # Create PPO configuration using modern API
        config = (
            PPOConfig()
            .environment(
                env="quantum_multiagent", 
                env_config={"num_quantum_dots": args.num_quantum_dots}
            )
            .multi_agent(
                policy_mapping_fn=policy_mapping_fn,
                policies=["plunger_policy", "barrier_policy"],
                policies_to_train=["plunger_policy", "barrier_policy"]
            )
            .rl_module(
                rl_module_spec=rl_module_spec
            )
            .env_runners(
                num_env_runners=2,  # Use 2 parallel workers
                rollout_fragment_length=200
            )
            .training(
                train_batch_size=4000,
                lr=3e-4,
                gamma=0.99,
                lambda_=0.95,
                clip_param=0.2,
                entropy_coeff=0.01,
                vf_loss_coeff=0.5,
                num_sgd_iter=10
            )
            .resources(
                num_gpus=0  # Use CPU only for simplicity
            )
        )
        
        # Build the algorithm
        print("\nBuilding PPO algorithm...\n")
        algo = config.build_algo()
        
        # Training loop
        print(f"\nStarting training for {args.num_iterations} iterations...\n")
        for i in range(args.num_iterations):
            result = algo.train()
            
            if i % 10 == 0:
                print(f"Iteration {i:3d}: "
                      f"reward_mean={result.get('env_runners', {}).get('episode_reward_mean', 'N/A'):.3f}, "
                      f"len_mean={result.get('env_runners', {}).get('episode_len_mean', 'N/A'):.1f}")
        
        # Save final checkpoint
        checkpoint_path = algo.save()
        print(f"\nTraining completed. Checkpoint saved to: {checkpoint_path}\n")
        
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()