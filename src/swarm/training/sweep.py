#!/usr/bin/env python3
"""
Hyperparameter sweep for Ray PPO using wandb sweeps.
This script integrates with the existing train.py and allows wandb to handle sweep orchestration.
TODO allow sweep params to override train script
"""

import os
import sys
import yaml
import wandb
from pathlib import Path

# Add src directory to path for clean imports
current_dir = Path(__file__).parent
swarm_package_dir = current_dir.parent
src_dir = swarm_package_dir.parent
sys.path.insert(0, str(src_dir))

from swarm.training.train import main as train_main


def create_sweep_config():
    """Define the wandb sweep configuration with key hyperparameters to optimize."""
    sweep_config = {
        'method': 'random',  # Can be 'grid', 'random', or 'bayes'
        'metric': {
            'name': 'episode_return_mean',
            'goal': 'maximize'
        },
        'parameters': {
            # PPO training hyperparameters
            'lr': {
                'values': [1e-5, 3e-4, 1e-3]
            },
            'gamma': {
                'values': [0.95, 0.99, 0.995]
            },
            'clip_param': {
                'values': [0.1, 0.2, 0.3]
            },
            'entropy_coeff': {
                'values': [0.0, 0.01, 0.05]
            },
            'vf_loss_coeff': {
                'values': [0.5, 1.0]
            },
            'num_epochs': {
                'values': [5, 10, 15]
            },
            
            # Batch size parameters
            'train_batch_size': {
                'values': [2048, 4096, 8192]
            },
            'minibatch_size': {
                'values': [32, 64, 128]
            },
            
            # Environment runner parameters
            'num_env_runners': {
                'values': [2, 4, 6]
            },
            'rollout_fragment_length': {
                'values': [25, 50, 100]
            },
            
            # Neural network architecture
            'plunger_feature_size': {
                'values': [128, 256, 512]
            },
            'plunger_lstm_cell_size': {
                'values': [256, 512, 1024]
            },
            'barrier_feature_size': {
                'values': [64, 128, 256]
            }
        }
    }
    return sweep_config


def train_with_sweep():
    """Training function that accepts wandb config and modifies the training config accordingly."""
    # Initialize wandb run (this is called by wandb agent)
    wandb.init()
    
    # Load base training config
    config_path = Path(__file__).parent / "training_config.yaml"
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Override config with wandb sweep parameters
    sweep_params = wandb.config
    
    # Update PPO training parameters
    if 'lr' in sweep_params:
        base_config['rl_config']['training']['lr'] = sweep_params['lr']
    if 'gamma' in sweep_params:
        base_config['rl_config']['training']['gamma'] = sweep_params['gamma']
    if 'clip_param' in sweep_params:
        base_config['rl_config']['training']['clip_param'] = sweep_params['clip_param']
    if 'entropy_coeff' in sweep_params:
        base_config['rl_config']['training']['entropy_coeff'] = sweep_params['entropy_coeff']
    if 'vf_loss_coeff' in sweep_params:
        base_config['rl_config']['training']['vf_loss_coeff'] = sweep_params['vf_loss_coeff']
    if 'num_epochs' in sweep_params:
        base_config['rl_config']['training']['num_epochs'] = sweep_params['num_epochs']
    
    # Update batch size parameters
    if 'train_batch_size' in sweep_params:
        base_config['rl_config']['training']['train_batch_size'] = sweep_params['train_batch_size']
    if 'minibatch_size' in sweep_params:
        base_config['rl_config']['training']['minibatch_size'] = sweep_params['minibatch_size']
    
    # Update environment runner parameters
    if 'num_env_runners' in sweep_params:
        base_config['rl_config']['env_runners']['num_env_runners'] = sweep_params['num_env_runners']
    if 'rollout_fragment_length' in sweep_params:
        base_config['rl_config']['env_runners']['rollout_fragment_length'] = sweep_params['rollout_fragment_length']
    
    # Update neural network architecture
    if 'plunger_feature_size' in sweep_params:
        base_config['neural_networks']['plunger_policy']['backbone']['feature_size'] = sweep_params['plunger_feature_size']
    if 'plunger_lstm_cell_size' in sweep_params:
        base_config['neural_networks']['plunger_policy']['backbone']['lstm']['cell_size'] = sweep_params['plunger_lstm_cell_size']
    if 'barrier_feature_size' in sweep_params:
        base_config['neural_networks']['barrier_policy']['backbone']['feature_size'] = sweep_params['barrier_feature_size']
    
    # Save modified config temporarily
    temp_config_path = Path(__file__).parent / f"temp_sweep_config_{wandb.run.id}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(base_config, f, default_flow_style=False, sort_keys=False)
    
    # Set environment variable to use the modified config
    os.environ['SWEEP_CONFIG_PATH'] = str(temp_config_path)
    
    try:
        # Run training with modified config
        train_main()
    finally:
        # Clean up temporary config file
        if temp_config_path.exists():
            temp_config_path.unlink()


def start_sweep():
    """Initialize and start a wandb sweep."""
    # Load base config for wandb project info
    config_path = Path(__file__).parent / "training_config.yaml"
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create sweep
    sweep_config = create_sweep_config()
    sweep_id = wandb.sweep(
        sweep_config, 
        project=base_config['wandb']['project'],
        entity=base_config['wandb']['entity']
    )
    
    print(f"Created sweep with ID: {sweep_id}")
    print(f"Run sweep with: wandb agent {base_config['wandb']['entity']}/{base_config['wandb']['project']}/{sweep_id}")
    
    return sweep_id


def run_agent(sweep_id=None):
    """Run a single sweep agent."""
    if sweep_id is None:
        print("Error: sweep_id is required")
        return
    
    # Load base config for project info
    config_path = Path(__file__).parent / "training_config.yaml"
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    full_sweep_id = f"{base_config['wandb']['entity']}/{base_config['wandb']['project']}/{sweep_id}"
    
    print(f"Starting sweep agent for: {full_sweep_id}")
    wandb.agent(full_sweep_id, train_with_sweep, count=1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Hyperparameter sweep for Ray PPO')
    parser.add_argument('--create-sweep', action='store_true', help='Create a new sweep')
    parser.add_argument('--run-agent', type=str, help='Run sweep agent with given sweep ID')
    
    args = parser.parse_args()
    
    if args.create_sweep:
        sweep_id = start_sweep()
    elif args.run_agent:
        run_agent(args.run_agent)
    else:
        print("Usage:")
        print("  Create sweep: python hyperparameter_sweep.py --create-sweep")
        print("  Run agent:    python hyperparameter_sweep.py --run-agent <sweep_id>")
        print("  Or use wandb: wandb agent <entity>/<project>/<sweep_id>")