"""
PPO trainer setup for multi-agent quantum device environment.
Handles distributed training, policy configuration, and model setup.
"""

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.utils.framework import try_import_torch
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Type
import numpy as np

torch, nn = try_import_torch()


class MultiModalPPOModel(TorchModelV2, nn.Module):
    """
    Custom PPO model for handling multi-modal observations (scans + voltage).
    """
    
    def __init__(self, obs_space: gym.Space, action_space: gym.Space, num_outputs: int,
                 model_config: Dict[str, Any], name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.model_config = model_config
        
        # Parse observation space
        if isinstance(obs_space, spaces.Dict):
            self.is_multimodal = True
            self.scan_space = obs_space['scans']
            self.voltage_space = obs_space['voltage']
        else:
            self.is_multimodal = False
            # Fallback for non-dict observation spaces
        
        if self.is_multimodal:
            # Handle scan observations (images)
            scan_shape = self.scan_space.shape
            if len(scan_shape) == 3:  # (height, width, channels) or (channels, height, width)
                # Assume (height, width, channels) format, convert to (channels, height, width)
                if scan_shape[2] <= 4:  # channels last
                    self.scan_input_shape = (scan_shape[2], scan_shape[0], scan_shape[1])
                else:  # channels first
                    self.scan_input_shape = scan_shape
            elif len(scan_shape) == 4:  # Multiple scans: (num_scans, height, width, channels)
                if scan_shape[3] <= 4:  # channels last
                    self.scan_input_shape = (scan_shape[0] * scan_shape[3], scan_shape[1], scan_shape[2])
                else:  # channels first
                    self.scan_input_shape = scan_shape
            else:
                raise ValueError(f"Unexpected scan shape: {scan_shape}")
            
            # Convolutional layers for scan processing
            self.conv_layers = nn.Sequential(
                nn.Conv2d(self.scan_input_shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten()
            )
            
            # Calculate conv output size
            with torch.no_grad():
                dummy_input = torch.zeros(1, *self.scan_input_shape)
                conv_output = self.conv_layers(dummy_input)
                conv_output_size = conv_output.shape[1]
            
            # Voltage processing (simple linear layer)
            voltage_size = np.prod(self.voltage_space.shape)
            self.voltage_fc = nn.Linear(voltage_size, 64)
            
            # Combined feature size
            combined_size = conv_output_size + 64
        else:
            # Simple fully connected network for non-multimodal observations
            combined_size = np.prod(obs_space.shape)
        
        # Shared layers
        hidden_dims = model_config.get("fcnet_hiddens", [256, 256])
        layers = []
        prev_size = combined_size
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_size, hidden_dim),
                nn.ReLU()
            ])
            prev_size = hidden_dim
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Value function head
        self.value_head = nn.Linear(prev_size, 1)
        
        # Policy head
        self.policy_head = nn.Linear(prev_size, num_outputs)
        
        self._value = None
    
    def forward(self, input_dict: Dict[str, torch.Tensor], state: list, seq_lens: torch.Tensor):
        """Forward pass through the model."""
        obs = input_dict["obs"]
        
        if self.is_multimodal:
            # Process scans
            scans = obs["scans"]
            voltage = obs["voltage"]
            
            # Reshape scans if necessary
            if len(scans.shape) == 5:  # (batch, num_scans, height, width, channels)
                batch_size = scans.shape[0]
                scans = scans.view(batch_size, -1, *scans.shape[3:])  # Flatten num_scans into channels
            
            if scans.shape[-1] <= 4:  # channels last -> channels first
                scans = scans.permute(0, -1, 1, 2)
            
            scan_features = self.conv_layers(scans)
            
            # Process voltage
            voltage_features = torch.relu(self.voltage_fc(voltage))
            
            # Combine features
            combined = torch.cat([scan_features, voltage_features], dim=1)
        else:
            combined = obs.float()
        
        # Shared processing
        shared_out = self.shared_layers(combined)
        
        # Policy output
        policy_out = self.policy_head(shared_out)
        
        # Store value for value_function() call
        self._value = self.value_head(shared_out).squeeze(-1)
        
        return policy_out, state
    
    def value_function(self) -> torch.Tensor:
        """Return the value function estimate."""
        return self._value


def create_ppo_config(config: Dict[str, Any], env_class: Type, policies: Dict[str, Any], 
                     policy_mapping_fn, policies_to_train: list, callback_class, 
                     num_quantum_dots: int = 8) -> PPOConfig:
    """
    Create PPO configuration for multi-agent training.
    
    Args:
        config: Training configuration dictionary
        env_class: Environment class
        policies: Policy specifications from FullTrainingInfra
        policy_mapping_fn: Policy mapping function from FullTrainingInfra
        policies_to_train: List of policies to train from FullTrainingInfra
        callback_class: Callback class from FullTrainingInfra
        num_quantum_dots: Number of quantum dots (N)
        
    Returns:
        Configured PPOConfig instance
    """
    from .ppo_config import get_ppo_config
    
    # Register custom model
    ModelCatalog.register_custom_model("multimodal_ppo", MultiModalPPOModel)
    
    # Get PPO configuration with any overrides from training config
    ppo_overrides = config.get("ppo_overrides", {})
    ppo_config_dict = get_ppo_config(ppo_overrides)
    
    # Configure model for each policy
    model_config = ppo_config_dict["model"].copy()
    model_config["custom_model"] = "multimodal_ppo"
    
    # Update policies with model config
    for policy_id in policies:
        policies[policy_id]["config"] = {
            "model": model_config
        }
    
    # Create PPO configuration
    ppo_config = (
        PPOConfig()
        .environment(env=env_class)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=policies_to_train,
        )
        .rollouts(
            num_rollout_workers=config["ray"]["num_workers"],
            rollout_fragment_length=config["env"]["rollout_fragment_length"],
            batch_mode=config["ray"]["batch_mode"],
            remote_worker_envs=config["ray"]["remote_worker_envs"],
        )
        .training(
            train_batch_size=config["env"]["train_batch_size"],
            sgd_minibatch_size=config["env"]["sgd_minibatch_size"],
            num_sgd_iter=ppo_config_dict["num_sgd_iter"],
            lr=ppo_config_dict["lr"],
            lr_schedule=ppo_config_dict["lr_schedule"],
            clip_param=ppo_config_dict["clip_param"],
            vf_clip_param=ppo_config_dict["vf_clip_param"],
            entropy_coeff=ppo_config_dict["entropy_coeff"],
            vf_loss_coeff=ppo_config_dict["vf_loss_coeff"],
            kl_coeff=ppo_config_dict["kl_coeff"],
            kl_target=ppo_config_dict["kl_target"],
            gamma=ppo_config_dict["gamma"],
            lambda_=ppo_config_dict["lambda"],
            model=model_config,
        )
        .resources(
            num_gpus=config["ray"]["num_gpus"],
            num_cpus_per_worker=config["ray"]["num_cpus_per_worker"],
            num_gpus_per_worker=config["ray"]["num_gpus_per_worker"],
        )
        .callbacks(callback_class)
        .debugging(seed=config["experiment"]["seed"])
    )
    
    # Add evaluation configuration if specified
    eval_config = config.get("evaluation", {})
    if eval_config:
        ppo_config = ppo_config.evaluation(
            evaluation_interval=eval_config.get("evaluation_interval"),
            evaluation_duration=eval_config.get("evaluation_duration"),
            evaluation_parallel_to_training=eval_config.get("evaluation_parallel_to_training", True),
        )
    
    return ppo_config


def create_tune_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create Ray Tune configuration for hyperparameter optimization.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Ray Tune configuration dictionary
    """
    tune_config = {
        "stop": config["stopping_criteria"],
        "checkpoint_config": {
            "checkpoint_frequency": config["checkpointing"]["frequency"],
            "num_to_keep": config["checkpointing"]["keep_checkpoints_num"],
            "checkpoint_score_attribute": config["checkpointing"]["checkpoint_score_attr"],
        },
        "verbose": 1,
    }
    
    return tune_config


class PPOTrainer:
    """Manages PPO training for quantum device environment."""
    
    def __init__(self, config: Dict[str, Any], env_class: Type):
        """
        Initialize PPO trainer.
        
        Args:
            config: Training configuration dictionary
            env_class: Environment class
        """
        self.config = config
        self.env_class = env_class
        self.algorithm = None
        self.ppo_config = None
        
        # Setup Ray if not already initialized
        if not ray.is_initialized():
            ray.init(
                num_gpus=config["ray"]["num_gpus"],
                object_store_memory=config["ray"]["object_store_memory"]
            )
    
    def setup_training(self, policies: Dict[str, Any], policy_mapping_fn, 
                      policies_to_train: list, callback_class, num_quantum_dots: int = 8):
        """
        Setup training configuration.
        
        Args:
            policies: Policy specifications from FullTrainingInfra
            policy_mapping_fn: Policy mapping function from FullTrainingInfra
            policies_to_train: List of policies to train from FullTrainingInfra
            callback_class: Callback class from FullTrainingInfra
            num_quantum_dots: Number of quantum dots
        """
        self.ppo_config = create_ppo_config(
            self.config, self.env_class, policies, policy_mapping_fn, 
            policies_to_train, callback_class, num_quantum_dots
        )
        return self.ppo_config
    
    def train(self, num_iterations: int = None):
        """
        Run training.
        
        Args:
            num_iterations: Number of training iterations (overrides config)
        """
        if self.ppo_config is None:
            raise RuntimeError("Must call setup_training() before train()")
        
        # Use config iterations if not specified
        if num_iterations is None:
            num_iterations = self.config["stopping_criteria"]["training_iteration"]
        
        # Build algorithm
        self.algorithm = self.ppo_config.build()
        
        # Training loop
        for i in range(num_iterations):
            result = self.algorithm.train()
            
            # Log progress
            if i % self.config["logging"]["log_frequency"] == 0:
                print(f"Iteration {i}: "
                      f"reward_mean={result.get('episode_reward_mean', 'N/A')}, "
                      f"len_mean={result.get('episode_len_mean', 'N/A')}")
            
            # Save checkpoint
            if i % self.config["checkpointing"]["frequency"] == 0:
                checkpoint_path = self.algorithm.save()
                print(f"Saved checkpoint: {checkpoint_path}")
            
            # Check stopping criteria
            if self._should_stop(result):
                print(f"Stopping criteria met at iteration {i}")
                break
        
        return self.algorithm
    
    def _should_stop(self, result: Dict[str, Any]) -> bool:
        """Check if training should stop based on criteria."""
        criteria = self.config["stopping_criteria"]
        
        # Check reward threshold
        if "episode_reward_mean" in criteria:
            if result.get("episode_reward_mean", -float('inf')) >= criteria["episode_reward_mean"]:
                return True
        
        # Check timesteps
        if "timesteps_total" in criteria:
            if result.get("timesteps_total", 0) >= criteria["timesteps_total"]:
                return True
        
        return False
    
    def cleanup(self):
        """Cleanup resources."""
        if self.algorithm:
            self.algorithm.stop()
        ray.shutdown() 