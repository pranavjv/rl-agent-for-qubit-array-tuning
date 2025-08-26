#!/usr/bin/env python3
"""
Main training script for multi-agent quantum device RL.
Orchestrates the entire training process with Ray RLlib and W&B logging.
"""

import os
import sys
import argparse
from pathlib import Path
import ray
from typing import Optional

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from utils.config_loader import ConfigLoader, load_config_from_file
from utils.wandb_logger import setup_wandb_logging
from utils.policy_mapping import get_policy_mapping_fn, get_policies_to_train, create_policy_specs

# Add VoltageAgent to path for imports
sys.path.append(str(current_dir.parent))
from VoltageAgent import get_trainer_class


def setup_environment():
    """Setup environment variables and paths."""
    # Set matplotlib backend to avoid GUI issues
    os.environ['MPLBACKEND'] = 'Agg'
    
    # Disable Ray dashboard completely to avoid pydantic compatibility issues
    os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'
    os.environ['RAY_DISABLE_RUNTIME_ENV_LOG_TO_DRIVER'] = '1'
    os.environ['RAY_DEDUP_LOGS'] = '0'
    
    # Add Swarm directory to Python path to import environment
    swarm_path = current_dir.parent
    if swarm_path.exists():
        sys.path.append(str(swarm_path))
    else:
        print(f"Warning: Swarm directory not found at {swarm_path}")


def import_environment():
    """Import the multi-agent quantum device environment wrapper."""
    try:
        # Add Swarm directory to path for package imports
        swarm_dir = current_dir.parent
        if str(swarm_dir) not in sys.path:
            sys.path.insert(0, str(swarm_dir))
        
        # Import multi-agent wrapper and base environment
        from Environment.multi_agent_wrapper import MultiAgentQuantumWrapper
        from Environment.env import QuantumDeviceEnv
        
        # Return factory function that creates wrapped environment
        def create_wrapped_env(num_quantum_dots=8, **kwargs):
            base_env = QuantumDeviceEnv(training=True, **kwargs)
            return MultiAgentQuantumWrapper(base_env, num_quantum_dots=num_quantum_dots)
        
        return create_wrapped_env
    except Exception as e:
        print(f"Failed to import MultiAgentQuantumWrapper: {e}")
        print("Please ensure the environment is properly set up in Swarm/Environment/")
        sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train multi-agent RL for quantum device tuning")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default=str(current_dir / "configs" / "config.yaml"),
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--num-quantum-dots", 
        type=int, 
        default=8,
        help="Number of quantum dots (N)"
    )
    
    parser.add_argument(
        "--num-iterations", 
        type=int, 
        default=None,
        help="Number of training iterations (overrides config)"
    )
    
    parser.add_argument(
        "--resume-from", 
        type=str, 
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--experiment-name", 
        type=str, 
        default=None,
        help="Override experiment name from config"
    )
    
    parser.add_argument(
        "--disable-wandb", 
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    
    parser.add_argument(
        "--ray-address", 
        type=str, 
        default=None,
        help="Ray cluster address (for distributed training)"
    )
    
    parser.add_argument(
        "--test-env", 
        action="store_true",
        help="Test environment setup and exit"
    )
    
    return parser.parse_args()


def test_environment_setup(env_instance, num_quantum_dots: int = 8):
    """Test multi-agent environment setup and print information."""
    print("Testing multi-agent environment setup...")
    
    try:
        print(f"✓ Environment created successfully")
        print(f"  Agent IDs: {env_instance.get_agent_ids}")
        print(f"  Number of agents: {len(env_instance.get_agent_ids)}")
        
        # Test reset
        obs, info = env_instance.reset()
        print(f"✓ Environment reset successful")
        print(f"  Got observations for {len(obs)} agents")
        
        # Check observation structure for a few agents
        sample_agents = env_instance.get_agent_ids[:3]  # Check first 3 agents
        for agent_id in sample_agents:
            agent_obs = obs[agent_id]
            print(f"  {agent_id}:")
            print(f"    Image shape: {agent_obs['image'].shape}")
            print(f"    Voltage shape: {agent_obs['voltage'].shape}")
        
        # Test step with sample actions
        actions = {}
        for agent_id in env_instance.get_agent_ids:
            actions[agent_id] = env_instance.action_space[agent_id].sample()
        
        obs, rewards, terminated, truncated, info = env_instance.step(actions)
        print(f"✓ Environment step successful")
        print(f"  Got rewards for {len(rewards)} agents")
        print(f"  Sample rewards: {dict(list(rewards.items())[:3])}")
        print(f"  Termination status: {terminated[env_instance.get_agent_ids[0]]}")
        
        env_instance.close()
        print("✓ Multi-agent environment test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Setup environment
    setup_environment()
    
    # Import environment factory function
    env_factory = import_environment()
    
    # Test environment if requested
    if args.test_env:
        # Create test environment instance
        test_env = env_factory(num_quantum_dots=args.num_quantum_dots)
        success = test_environment_setup(test_env, args.num_quantum_dots)
        sys.exit(0 if success else 1)
    
    # Load configuration
    try:
        config = load_config_from_file(args.config)
        print(f"Loaded configuration from: {args.config}")
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Apply command line overrides
    if args.experiment_name:
        config["experiment"]["name"] = args.experiment_name
    
    if args.disable_wandb:
        config["logging"]["wandb"]["enabled"] = False
    
    # Initialize Ray
    # Setup Ray runtime environment
    swarm_dir = current_dir.parent
    ray_config = {
        "num_gpus": config["ray"]["num_gpus"],
        "object_store_memory": config["ray"]["object_store_memory"],
        "include_dashboard": False,  # Disable dashboard to avoid pydantic compatibility issues
        "_node_ip_address": "127.0.0.1",  # Force local IP to avoid network issues
        "dashboard_host": "127.0.0.1",
        "dashboard_port": None,  # Disable dashboard port
        "_temp_dir": "/tmp/ray_temp",  # Set explicit temp dir
        "runtime_env": {
            "working_dir": str(swarm_dir),
            "py_modules": [
                str(swarm_dir / "Environment"),
                str(swarm_dir / "CapacitanceModel")
            ]
        }
    }
    
    if args.ray_address:
        ray.init(address=args.ray_address)
        print(f"Connected to Ray cluster at: {args.ray_address}")
    else:
        ray.init(**ray_config)
        print("Initialized local Ray cluster")
    
    try:
        # Setup W&B logging
        wandb_logger, callback = setup_wandb_logging(config)
        print("W&B logging initialized")
        
        # Get trainer class and create instance
        trainer_type = config.get("trainer_type", "recurrent_ppo") # use recurrent by default
        trainer_class = get_trainer_class(trainer_type)
        trainer = trainer_class(config, env_factory)
        print(f"{trainer_type.upper()} trainer created")
        
        # Create environment instance for policy setup
        env_instance = env_factory(num_quantum_dots=args.num_quantum_dots)

        # Setup policies and mapping functions
        policies = create_policy_specs(env_instance)
        policy_mapping_fn = get_policy_mapping_fn(args.num_quantum_dots)
        policies_to_train = get_policies_to_train()
        
        # Setup training configuration
        ppo_config = trainer.setup_training(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=policies_to_train,
            callback_class=callback,
            num_quantum_dots=args.num_quantum_dots
        )
        print(f"Training configuration setup for {args.num_quantum_dots} quantum dots")
        
        # Resume from checkpoint if specified
        if args.resume_from:
            print(f"Resuming from checkpoint: {args.resume_from}")
            # Note: Checkpoint resuming would be implemented based on specific needs
        
        # Start training
        print("Starting training...")
        print(f"Configuration: {config['experiment']['name']}")
        print(f"Total workers: {config['ray']['num_workers']}")
        print(f"GPUs: {config['ray']['num_gpus']}")
        print(f"Stopping criteria: {config['stopping_criteria']}")
        
        algorithm = trainer.train(args.num_iterations)
        
        # Save final checkpoint
        final_checkpoint = algorithm.save()
        print(f"Training completed. Final checkpoint saved to: {final_checkpoint}")
        
        # Cleanup
        wandb_logger.finish()
        trainer.cleanup()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        if 'wandb_logger' in locals():
            wandb_logger.finish()
        if 'trainer' in locals():
            trainer.cleanup()
        sys.exit(1)
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        if 'wandb_logger' in locals():
            wandb_logger.finish()
        if 'trainer' in locals():
            trainer.cleanup()
        sys.exit(1)
    
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main() 