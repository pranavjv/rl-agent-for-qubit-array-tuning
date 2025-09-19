#!/usr/bin/env python3
"""
Comprehensive hyperparameter sweep for RL training using wandb sweeps.
This script provides configurable sweep parameters for all rl_config.training parameters
and integrates with the existing train.py for full wandb visualization.
"""

import os
import sys
import yaml
import wandb
from pathlib import Path
from typing import Dict, Any, List, Union

# Add src directory to path for clean imports
current_dir = Path(__file__).parent
swarm_package_dir = current_dir.parent
src_dir = swarm_package_dir.parent
sys.path.insert(0, str(src_dir))

from swarm.training.train import main as train_main


# =============================================================================
# SWEEP CONFIGURATION - MODIFY THESE VALUES TO CUSTOMIZE THE SWEEP
# =============================================================================
"""
This section contains all configurable parameters for the hyperparameter sweep.
To customize your sweep:

1. Modify the values in SWEEP_PARAMETERS below
2. Change SWEEP_METHOD ('random', 'grid', 'bayes')
3. Update SWEEP_METRIC if needed
4. Adjust MAX_RUNS for random/bayes methods

Examples of parameter modifications:
- To test only specific learning rates: 'lr': [3e-4, 1e-3]
- To disable a parameter: 'gamma': [0.99]  # Single value = disabled
- To add more values: 'clip_param': [0.1, 0.15, 0.2, 0.25, 0.3]

The sweep will automatically include only parameters with multiple values.
"""

# train_batch_size: [8192, 16384, 32768, 65536]
# sgd_minibatch_size: [1024, 2048, 4096]
# lr: [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]

SWEEP_PARAMETERS = {
    # Core training hyperparameters from rl_config.training
    'minibatch_size': [64, 256],
    'num_epochs': [1, 2, 5],
    'lr': [1e-5, 3e-4, 1e-3],

    # 'gamma': [0.95, 0.99, 0.995, 0.999],
    # 'lambda_': [0.9, 0.95, 0.98, 1.0],

    'clip_param': [0.1, 0.2, 0.3, 0.4],
    'entropy_coeff': [0.0, 0.001, 0.01, 0.05, 0.1],
    'vf_loss_coeff': [0.25, 0.5, 1.0, 2.0],

    'kl_target': [0.005, 0.01, 0.02, 0.05],
    'grad_clip': [5.0, 10.0, 20.0, None],
    'grad_clip_by': ['norm'],
    
    # Algorithm choice
    'algorithm': ['PPO'],
    
    # Control training duration for sweeps
    'num_iterations': [50],  # Shorter runs for hyperparameter search
}

# Sweep method configuration
SWEEP_METHOD = 'bayes'  # Options: 'random', 'grid', 'bayes'
SWEEP_METRIC = {
    'name': 'plunger_return_ema',
    'goal': 'maximize'
}

# Number of runs per sweep (only applies to random and bayes methods)
MAX_RUNS = 50


def create_sweep_config() -> Dict[str, Any]:
    """
    Create wandb sweep configuration based on the parameters defined above.
    
    Returns:
        Dict containing the complete wandb sweep configuration
    """
    sweep_config = {
        'method': SWEEP_METHOD,
        'metric': SWEEP_METRIC,
        'parameters': {},
        'program': 'train.py',  # Specify the training script
        'command': [
            '${env}',
            'python3',
            '${program}',
            '${args}'
        ]
    }
    
    # Add early termination for bayes method
    if SWEEP_METHOD == 'bayes':
        sweep_config['early_terminate'] = {
            'type': 'hyperband',
            'min_iter': 10,
            'eta': 3
        }
    
    # Convert parameter lists to wandb format
    for param_name, values in SWEEP_PARAMETERS.items():
        if isinstance(values, list) and len(values) > 0:
            sweep_config['parameters'][param_name] = {'values': values}
    
    return sweep_config


def get_enabled_parameters() -> List[str]:
    """
    Get list of parameters that are enabled for sweeping (have multiple values).
    
    Returns:
        List of parameter names that will be swept
    """
    enabled = []
    for param_name, values in SWEEP_PARAMETERS.items():
        if isinstance(values, list) and len(values) > 1:
            enabled.append(param_name)
    return enabled


def print_sweep_info():
    """Print information about the current sweep configuration."""
    enabled_params = get_enabled_parameters()
    
    print("=" * 80)
    print("WANDB SWEEP CONFIGURATION")
    print("=" * 80)
    print(f"Method: {SWEEP_METHOD}")
    print(f"Metric: {SWEEP_METRIC['name']} ({SWEEP_METRIC['goal']})")
    if SWEEP_METHOD in ['random', 'bayes']:
        print(f"Max runs: {MAX_RUNS}")
    print(f"\nParameters to sweep ({len(enabled_params)}):")
    
    for param in enabled_params:
        values = SWEEP_PARAMETERS[param]
        print(f"  {param}: {len(values)} values - {values}")
    
    print("=" * 80)


def train_with_sweep():
    """
    Training function that accepts wandb config and modifies the training config accordingly.
    This function handles all parameters defined in SWEEP_PARAMETERS.
    """
    # Initialize wandb run (this is called by wandb agent)
    wandb.init()
    
    # Load base training config
    config_path = Path(__file__).parent / "training_config.yaml"
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Override config with wandb sweep parameters
    sweep_params = wandb.config
    
    # Update core rl_config.training parameters
    training_params = [
        'train_batch_size', 'minibatch_size', 'lr', 'gamma', 'lambda_', 
        'clip_param', 'entropy_coeff', 'vf_loss_coeff', 'kl_target', 
        'num_epochs', 'grad_clip', 'grad_clip_by'
    ]
    
    for param in training_params:
        if param in sweep_params:
            base_config['rl_config']['training'][param] = sweep_params[param]
    
    # Update algorithm choice
    if 'algorithm' in sweep_params:
        base_config['rl_config']['algorithm'] = sweep_params['algorithm']
    
    # Update environment runner parameters
    env_runner_params = [
        'num_env_runners', 'rollout_fragment_length', 'sample_timeout_s', 
        'num_gpus_per_env_runner'
    ]
    
    for param in env_runner_params:
        if param in sweep_params:
            base_config['rl_config']['env_runners'][param] = sweep_params[param]
    
    # Update learner parameters
    learner_params = ['num_learners', 'num_gpus_per_learner']
    
    for param in learner_params:
        if param in sweep_params:
            base_config['rl_config']['learners'][param] = sweep_params[param]
    
    # Update multi-agent parameters
    if 'free_log_std' in sweep_params:
        base_config['rl_config']['multi_agent']['free_log_std'] = sweep_params['free_log_std']
    if 'log_std_bounds' in sweep_params:
        base_config['rl_config']['multi_agent']['log_std_bounds'] = sweep_params['log_std_bounds']
    
    # Update neural network architecture - plunger policy
    if 'plunger_feature_size' in sweep_params:
        base_config['neural_networks']['plunger_policy']['backbone']['feature_size'] = sweep_params['plunger_feature_size']
    
    if 'plunger_lstm_cell_size' in sweep_params:
        if 'lstm' in base_config['neural_networks']['plunger_policy']['backbone']:
            base_config['neural_networks']['plunger_policy']['backbone']['lstm']['cell_size'] = sweep_params['plunger_lstm_cell_size']
    
    if 'plunger_lstm_max_seq_len' in sweep_params:
        if 'lstm' in base_config['neural_networks']['plunger_policy']['backbone']:
            base_config['neural_networks']['plunger_policy']['backbone']['lstm']['max_seq_len'] = sweep_params['plunger_lstm_max_seq_len']
    
    if 'plunger_policy_hidden_layers' in sweep_params:
        base_config['neural_networks']['plunger_policy']['policy_head']['hidden_layers'] = sweep_params['plunger_policy_hidden_layers']
    
    if 'plunger_value_hidden_layers' in sweep_params:
        base_config['neural_networks']['plunger_policy']['value_head']['hidden_layers'] = sweep_params['plunger_value_hidden_layers']
    
    # Update neural network architecture - barrier policy
    if 'barrier_feature_size' in sweep_params:
        base_config['neural_networks']['barrier_policy']['backbone']['feature_size'] = sweep_params['barrier_feature_size']
    
    if 'barrier_policy_hidden_layers' in sweep_params:
        base_config['neural_networks']['barrier_policy']['policy_head']['hidden_layers'] = sweep_params['barrier_policy_hidden_layers']
    
    if 'barrier_value_hidden_layers' in sweep_params:
        base_config['neural_networks']['barrier_policy']['value_head']['hidden_layers'] = sweep_params['barrier_value_hidden_layers']
    
    # Log the modified parameters to wandb for tracking
    wandb.config.update({
        'modified_config': {k: v for k, v in sweep_params.items() if k in SWEEP_PARAMETERS}
    })
    
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
    """Initialize and start a wandb sweep with comprehensive parameter configuration."""
    # Load base config for wandb project info
    config_path = Path(__file__).parent / "training_config.yaml"
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Print sweep configuration info
    print_sweep_info()
    
    # Create sweep configuration
    sweep_config = create_sweep_config()
    
    # Add max runs for random/bayes methods
    if SWEEP_METHOD in ['random', 'bayes']:
        sweep_config['run_cap'] = MAX_RUNS
    
    # Create the sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        project=base_config['wandb']['project'],
        entity=base_config['wandb']['entity']
    )
    
    print(f"\n✅ Created sweep with ID: {sweep_id}")
    print(f"🚀 Run sweep with: wandb agent {base_config['wandb']['entity']}/{base_config['wandb']['project']}/{sweep_id}")
    print(f"📊 Or view at: https://wandb.ai/{base_config['wandb']['entity']}/{base_config['wandb']['project']}/sweeps/{sweep_id}")
    
    return sweep_id


def run_agent(sweep_id=None, count=None):
    """
    Run sweep agent(s) for the specified sweep.
    
    Args:
        sweep_id: The sweep ID to run agents for
        count: Number of runs for this agent (default: 1, use None for unlimited)
    """
    if sweep_id is None:
        print("❌ Error: sweep_id is required")
        return
    
    # Load base config for project info
    config_path = Path(__file__).parent / "training_config.yaml"
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    full_sweep_id = f"{base_config['wandb']['entity']}/{base_config['wandb']['project']}/{sweep_id}"
    
    print(f"🤖 Starting sweep agent for: {full_sweep_id}")
    if count is not None:
        print(f"📊 Running {count} experiments")
        wandb.agent(full_sweep_id, train_with_sweep, count=count)
    else:
        print(f"🔄 Running unlimited experiments (until stopped)")
        wandb.agent(full_sweep_id, train_with_sweep)


def list_parameters():
    """List all available parameters and their current values for sweeping."""
    print("=" * 80)
    print("AVAILABLE SWEEP PARAMETERS")
    print("=" * 80)
    print("Modify the SWEEP_PARAMETERS dictionary in this script to customize values.\n")
    
    categories = {
        'Core Training Parameters': [
            'train_batch_size', 'minibatch_size', 'lr', 'gamma', 'lambda_', 
            'clip_param', 'entropy_coeff', 'vf_loss_coeff', 'kl_target', 
            'num_epochs', 'grad_clip', 'grad_clip_by'
        ],
        'Environment & System': [
            'num_env_runners', 'rollout_fragment_length', 'sample_timeout_s', 
            'num_gpus_per_env_runner', 'num_learners', 'num_gpus_per_learner'
        ],
        'Multi-agent': [
            'free_log_std', 'log_std_bounds'
        ],
        'Neural Networks - Plunger': [
            'plunger_feature_size', 'plunger_lstm_cell_size', 'plunger_lstm_max_seq_len',
            'plunger_policy_hidden_layers', 'plunger_value_hidden_layers'
        ],
        'Neural Networks - Barrier': [
            'barrier_feature_size', 'barrier_policy_hidden_layers', 'barrier_value_hidden_layers'
        ],
        'Algorithm': [
            'algorithm'
        ]
    }
    
    for category, params in categories.items():
        print(f"\n{category}:")
        for param in params:
            if param in SWEEP_PARAMETERS:
                values = SWEEP_PARAMETERS[param]
                status = "✅ ENABLED" if len(values) > 1 else "❌ DISABLED (single value)"
                print(f"  {param:30} {status:15} Values: {values}")
            else:
                print(f"  {param:30} ❌ NOT FOUND")
    
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Comprehensive hyperparameter sweep for RL training using wandb',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sweep.py --create-sweep                    # Create a new sweep
  python sweep.py --run-agent <sweep_id>            # Run single agent
  python sweep.py --run-agent <sweep_id> --count 5  # Run 5 experiments
  python sweep.py --list-params                     # Show all parameters
  python sweep.py --info                            # Show current config
        """
    )
    
    parser.add_argument('--create-sweep', action='store_true', 
                       help='Create a new wandb sweep')
    parser.add_argument('--run-agent', type=str, metavar='SWEEP_ID',
                       help='Run sweep agent with given sweep ID')
    parser.add_argument('--count', type=int, metavar='N',
                       help='Number of experiments to run (default: 1, use 0 for unlimited)')
    parser.add_argument('--list-params', action='store_true',
                       help='List all available parameters for sweeping')
    parser.add_argument('--info', action='store_true',
                       help='Show current sweep configuration')
    
    args = parser.parse_args()
    
    if args.list_params:
        list_parameters()
    elif args.info:
        print_sweep_info()
    elif args.create_sweep:
        sweep_id = start_sweep()
    elif args.run_agent:
        count = args.count if args.count != 0 else None
        if count is None and args.count == 0:
            count = None  # Unlimited
        elif count is None:
            count = 1  # Default
        run_agent(args.run_agent, count)
    else:
        parser.print_help()
        print(f"\n🔧 Current sweep method: {SWEEP_METHOD}")
        print(f"📊 Enabled parameters: {len(get_enabled_parameters())}")
        print(f"🎯 Target metric: {SWEEP_METRIC['name']} ({SWEEP_METRIC['goal']})")
        print(f"\nUse --info to see detailed configuration or --list-params to see all parameters.")