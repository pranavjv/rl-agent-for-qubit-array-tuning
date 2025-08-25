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
from agents.ppo_trainer import PPOTrainer


def setup_environment():
    """Setup environment variables and paths."""
    # Set matplotlib backend to avoid GUI issues
    os.environ['MPLBACKEND'] = 'Agg'
    
    # Add Swarm directory to Python path to import environment
    swarm_path = current_dir.parent / "Swarm"
    if swarm_path.exists():
        sys.path.append(str(swarm_path))
    else:
        print(f"Warning: Swarm directory not found at {swarm_path}")


def import_environment():
    """Import the quantum device environment."""
    try:
        raise NotImplementedError
    except NotImplementedError as e:
        print(f"Failed to import QuantumDeviceEnv: {e}")
        print("Please ensure the environment is properly set up in Swarm/Qarray/env.py")
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


def test_environment_setup(env_class, num_quantum_dots: int = 8):
    """Test environment setup and print information."""
    print("Testing environment setup...")
    
    try:
        env = env_class()
        print(f"✓ Environment created successfully")
        
        # Test reset
        obs, info = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  Observation keys: {list(obs.keys()) if isinstance(obs, dict) else 'Single observation'}")
        
        # Test step
        if hasattr(env, 'action_space'):
            if isinstance(env.action_space, dict):
                # Multi-agent action space
                actions = {agent_id: space.sample() for agent_id, space in env.action_space.items()}
            else:
                # Single action space
                actions = env.action_space.sample()
            
            obs, rewards, terminated, truncated, info = env.step(actions)
            print(f"✓ Environment step successful")
            print(f"  Reward keys: {list(rewards.keys()) if isinstance(rewards, dict) else 'Single reward'}")
        
        env.close()
        print("✓ Environment test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Setup environment
    setup_environment()
    
    # Import environment class
    env_class = import_environment()
    
    # Test environment if requested
    if args.test_env:
        success = test_environment_setup(env_class, args.num_quantum_dots)
        sys.exit(0 if success else 1)
    
    # Load configuration
    try:
        config = load_config_from_file(args.config)
        print(f"Loaded configuration from: {args.config}")
    except FileNotFoundError:
        print(f"Configuration file not found: {args.config}")
        print("Creating default configuration...")
        
        # Create default config and save it
        from utils.config_loader import create_default_config
        config = create_default_config()
        
        config_path = Path(args.config)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        loader = ConfigLoader()
        loader.config = config
        loader.save_config(args.config)
        print(f"Default configuration saved to: {args.config}")
    
    # Apply command line overrides
    if args.experiment_name:
        config["experiment"]["name"] = args.experiment_name
    
    if args.disable_wandb:
        config["logging"]["wandb"]["enabled"] = False
    
    # Initialize Ray
    ray_config = {
        "num_gpus": config["ray"]["num_gpus"],
        "object_store_memory": config["ray"]["object_store_memory"]
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
        
        # Create trainer
        trainer = PPOTrainer(config, env_class)
        print("PPO trainer created")
        
        # Setup training configuration
        ppo_config = trainer.setup_training(args.num_quantum_dots)
        print(f"Training configuration setup for {args.num_quantum_dots} quantum dots")
        
        # Update callback in configuration
        ppo_config = ppo_config.callbacks(callback)
        trainer.ppo_config = ppo_config
        
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