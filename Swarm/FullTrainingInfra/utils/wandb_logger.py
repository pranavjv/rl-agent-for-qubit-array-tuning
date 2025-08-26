"""
Weights & Biases logging utilities for multi-agent quantum device RL training.
Handles experiment tracking, custom metrics, and visualization.
"""

import wandb
import os
import yaml
from typing import Dict, Any, List, Optional
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.policy import Policy
from ray.rllib.evaluation.worker_set import WorkerSet


class WandbLogger:
    """Manages Weights & Biases experiment logging."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize W&B logger.
        
        Args:
            config: Configuration dictionary with logging settings
        """
        self.config = config
        self.wandb_config = config.get('logging', {}).get('wandb', {})
        self.enabled = self.wandb_config.get('enabled', True)
        self.project = config.get('experiment', {}).get('project', 'qubit-array-tuning')
        self.experiment_name = config.get('experiment', {}).get('name', 'quantum_device_training')
        self.tags = config.get('experiment', {}).get('tags', [])
        
        if self.enabled:
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize Weights & Biases run."""
        # Set API key from environment variable if specified
        api_key_env = self.wandb_config.get('api_key_env')
        if api_key_env and api_key_env in os.environ:
            wandb.login(key=os.environ[api_key_env])
        
        # Initialize run
        wandb.init(
            project=self.project,
            name=self.experiment_name,
            tags=self.tags,
            config=self._flatten_config(self.config),
            reinit=True
        )
        
        print(f"Initialized W&B logging for project: {self.project}")
    
    def _flatten_config(self, config: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """Flatten nested configuration dictionary for W&B."""
        items = []
        for k, v in config.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_config(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to W&B."""
        if self.enabled:
            wandb.log(metrics, step=step)
    
    def log_config_update(self, new_config: Dict[str, Any]):
        """Update configuration in W&B."""
        if self.enabled:
            wandb.config.update(self._flatten_config(new_config))
    
    def finish(self):
        """Finish W&B run."""
        if self.enabled:
            wandb.finish()


class MultiAgentMetricsCallback(DefaultCallbacks):
    """Custom callback for tracking multi-agent specific metrics."""
    
    def __init__(self):
        super().__init__()
        self.wandb_logger = None
    
    def set_wandb_logger(self, logger: WandbLogger):
        """Set the W&B logger instance."""
        self.wandb_logger = logger
    
    def on_episode_end(self, *, worker, base_env: BaseEnv, policies: Dict[str, Policy], 
                      episode: MultiAgentEpisode, env_index: int, **kwargs):
        """Called when an episode ends."""
        # Collect per-agent rewards
        plunger_rewards = []
        barrier_rewards = []
        individual_rewards = {}
        
        for agent_id, reward in episode.agent_rewards.items():
            individual_rewards[f"agent_{agent_id}_reward"] = reward
            
            if "plunger" in agent_id.lower():
                plunger_rewards.append(reward)
            elif "barrier" in agent_id.lower():
                barrier_rewards.append(reward)
        
        # Calculate aggregate metrics
        metrics = {
            "episode_length": episode.length,
            "total_episode_reward": sum(episode.agent_rewards.values()),
        }
        
        if plunger_rewards:
            metrics.update({
                "plunger_reward_mean": np.mean(plunger_rewards),
                "plunger_reward_std": np.std(plunger_rewards),
                "plunger_reward_min": np.min(plunger_rewards),
                "plunger_reward_max": np.max(plunger_rewards),
            })
        
        if barrier_rewards:
            metrics.update({
                "barrier_reward_mean": np.mean(barrier_rewards),
                "barrier_reward_std": np.std(barrier_rewards),
                "barrier_reward_min": np.min(barrier_rewards),
                "barrier_reward_max": np.max(barrier_rewards),
            })
        
        # Add individual agent rewards (limited to avoid cluttering)
        metrics.update(individual_rewards)
        
        # Store custom metrics in episode
        for key, value in metrics.items():
            episode.custom_metrics[key] = value
    
    def on_train_result(self, *, algorithm, result: Dict[str, Any], **kwargs):
        """Called after each training iteration."""
        if self.wandb_logger and self.wandb_logger.enabled:
            # Extract relevant metrics for W&B
            wandb_metrics = {}
            
            # Basic training metrics
            if "episode_reward_mean" in result:
                wandb_metrics["episode_reward_mean"] = result["episode_reward_mean"]
            if "episode_len_mean" in result:
                wandb_metrics["episode_len_mean"] = result["episode_len_mean"]
            if "episodes_this_iter" in result:
                wandb_metrics["episodes_this_iter"] = result["episodes_this_iter"]
            if "timesteps_this_iter" in result:
                wandb_metrics["timesteps_this_iter"] = result["timesteps_this_iter"]
            
            # Policy-specific metrics
            for policy_id in ["plunger_policy", "barrier_policy"]:
                policy_key = f"info/learner/{policy_id}"
                if policy_key in result:
                    policy_info = result[policy_key]
                    for metric_name in ["policy_loss", "vf_loss", "entropy", "kl"]:
                        if metric_name in policy_info:
                            wandb_metrics[f"{policy_id}_{metric_name}"] = policy_info[metric_name]
            
            # Custom multi-agent metrics
            custom_metrics = result.get("custom_metrics", {})
            for metric_name in ["plunger_reward_mean", "barrier_reward_mean", 
                              "total_episode_reward", "episode_length"]:
                if f"{metric_name}_mean" in custom_metrics:
                    wandb_metrics[metric_name] = custom_metrics[f"{metric_name}_mean"]
                elif metric_name in custom_metrics:
                    wandb_metrics[metric_name] = custom_metrics[metric_name]
            
            # Training iteration
            if "training_iteration" in result:
                wandb_metrics["training_iteration"] = result["training_iteration"]
            
            # Log to W&B
            self.wandb_logger.log_metrics(
                wandb_metrics, 
                step=result.get("training_iteration")
            )


def setup_wandb_logging(config: Dict[str, Any]):
    """
    Set up W&B logging with custom callback.
    
    Args:
        config: Training configuration dictionary
        
    Returns:
        Tuple of (wandb_logger, callback_class)
    """
    # Create W&B logger
    wandb_logger = WandbLogger(config)
    
    # Create callback class factory that captures the wandb_logger
    class ConfiguredMultiAgentMetricsCallback(MultiAgentMetricsCallback):
        def __init__(self):
            super().__init__()
            self.set_wandb_logger(wandb_logger)
    
    return wandb_logger, ConfiguredMultiAgentMetricsCallback


def create_custom_metrics_dict() -> Dict[str, Any]:
    """
    Create a dictionary of custom metrics for RLlib configuration.
    
    Returns:
        Dictionary with custom metrics configuration
    """
    return {
        "custom_metrics": [
            "plunger_reward_mean",
            "barrier_reward_mean", 
            "total_episode_reward",
            "episode_length"
        ]
    } 