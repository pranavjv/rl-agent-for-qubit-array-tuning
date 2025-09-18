#!/usr/bin/env python3
"""
Random policy rollout script for quantum device tuning.
Mimics the exact same setup as PPO train.py but uses completely random actions
to provide a baseline comparison for PPO performance.
"""

import os
import sys
import argparse
import yaml
import time
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import ray
import wandb
from ray.tune.registry import register_env

# Suppress warnings
os.environ["PYTHONWARNINGS"] = "ignore::DeprecationWarning"
os.environ["RAY_DEDUP_LOGS"] = "0"

# Set logging level to reduce verbosity
logging.getLogger("ray").setLevel(logging.WARNING)
logging.getLogger("ray.tune").setLevel(logging.WARNING)

# Add src directory to path for clean imports
current_dir = Path(__file__).parent
swarm_package_dir = current_dir.parent  # swarm package directory
src_dir = swarm_package_dir.parent  # src directory
project_root = src_dir.parent  # project root directory
sys.path.insert(0, str(src_dir))

from swarm.environment.multi_agent_wrapper import MultiAgentEnvWrapper


def create_env(config=None):
    """Create multi-agent quantum environment."""
    return MultiAgentEnvWrapper()


def load_config():
    """Load training configuration from YAML file."""
    config_path = Path(__file__).parent / "training_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


class RandomPolicy:
    """Random policy that samples actions from the action space."""
    
    def __init__(self, action_space):
        self.action_space = action_space
    
    def get_action(self, observation):
        """Sample a random action."""
        return self.action_space.sample()


class EpisodeLogger:
    """Logger for tracking episode metrics."""
    
    def __init__(self, use_wandb: bool = True):
        self.use_wandb = use_wandb
        self.episode_rewards = []
        self.episode_lengths = []
        self.agent_rewards = {
            'plunger_rewards': [],
            'barrier_rewards': []
        }
    
    def log_episode(self, episode_reward: float, episode_length: int, agent_rewards: Dict[str, float]):
        """Log metrics for a completed episode."""
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        
        # Separate plunger and barrier rewards
        plunger_rewards = [reward for agent_id, reward in agent_rewards.items() if 'plunger' in agent_id]
        barrier_rewards = [reward for agent_id, reward in agent_rewards.items() if 'barrier' in agent_id]
        
        self.agent_rewards['plunger_rewards'].extend(plunger_rewards)
        self.agent_rewards['barrier_rewards'].extend(barrier_rewards)
        
        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log({
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'mean_plunger_reward': np.mean(plunger_rewards) if plunger_rewards else 0,
                'mean_barrier_reward': np.mean(barrier_rewards) if barrier_rewards else 0,
                'total_plunger_reward': np.sum(plunger_rewards) if plunger_rewards else 0,
                'total_barrier_reward': np.sum(barrier_rewards) if barrier_rewards else 0,
            })
    
    def get_stats(self) -> Dict[str, float]:
        """Get summary statistics."""
        if not self.episode_rewards:
            return {}
        
        return {
            'mean_episode_reward': np.mean(self.episode_rewards),
            'std_episode_reward': np.std(self.episode_rewards),
            'min_episode_reward': np.min(self.episode_rewards),
            'max_episode_reward': np.max(self.episode_rewards),
            'mean_episode_length': np.mean(self.episode_lengths),
            'total_episodes': len(self.episode_rewards),
            'mean_plunger_reward': np.mean(self.agent_rewards['plunger_rewards']) if self.agent_rewards['plunger_rewards'] else 0,
            'mean_barrier_reward': np.mean(self.agent_rewards['barrier_rewards']) if self.agent_rewards['barrier_rewards'] else 0,
        }


def run_random_episodes(num_episodes: int, config: Dict, use_wandb: bool = True):
    """
    Run episodes with random policy.
    
    Args:
        num_episodes: Number of episodes to run
        config: Training configuration (same as PPO)
        use_wandb: Whether to log to wandb
    """
    
    # Initialize Ray with the same config as PPO training
    ray_config = {
        "include_dashboard": config['ray']['include_dashboard'],
        "log_to_driver": config['ray']['log_to_driver'],
        "logging_level": config['ray']['logging_level'],
        "runtime_env": {
            "working_dir": str(src_dir),
            "excludes": config['ray']['runtime_env']['excludes'],
            "env_vars": {
                **config['ray']['runtime_env']['env_vars'],
                "SWARM_PROJECT_ROOT": str(project_root),
            },
        },
    }
    
    print("\nInitializing Ray...\n")
    ray.init(**ray_config)
    
    try:
        # Register environment (same as PPO)
        register_env("qarray_multiagent_env", create_env)
        
        # Create environment instance
        env = create_env()
        agent_ids = env.get_agent_ids()
        
        # Create random policies for each agent
        policies = {}
        for agent_id in agent_ids:
            policies[agent_id] = RandomPolicy(env.action_spaces[agent_id])
        
        # Initialize logger
        logger = EpisodeLogger(use_wandb=use_wandb)
        
        print(f"Running {num_episodes} episodes with random policy...")
        print(f"Environment has {len(agent_ids)} agents: {agent_ids}")
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Reset environment
            observations, infos = env.reset()
            
            episode_rewards = {agent_id: 0.0 for agent_id in agent_ids}
            episode_length = 0
            terminated = False
            truncated = False
            
            while not terminated and not truncated:
                # Get random actions for all agents
                actions = {}
                for agent_id in agent_ids:
                    actions[agent_id] = policies[agent_id].get_action(observations[agent_id])
                
                # Step environment
                observations, rewards, term_dict, trunc_dict, infos = env.step(actions)
                
                # Accumulate rewards
                for agent_id in agent_ids:
                    episode_rewards[agent_id] += rewards[agent_id]
                
                episode_length += 1
                terminated = term_dict.get("__all__", False)
                truncated = trunc_dict.get("__all__", False)
            
            # Calculate total episode reward
            total_episode_reward = sum(episode_rewards.values())
            
            # Log episode
            logger.log_episode(total_episode_reward, episode_length, episode_rewards)
            
            # Print progress
            if (episode + 1) % 10 == 0 or episode == 0:
                elapsed_time = time.time() - start_time
                episodes_per_sec = (episode + 1) / elapsed_time
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Reward: {total_episode_reward:.2f} | "
                      f"Length: {episode_length} | "
                      f"Rate: {episodes_per_sec:.2f} eps/s")
        
        # Final statistics
        stats = logger.get_stats()
        
        print("\n" + "="*60)
        print("RANDOM POLICY ROLLOUT RESULTS")
        print("="*60)
        print(f"Total Episodes: {stats['total_episodes']}")
        print(f"Mean Episode Reward: {stats['mean_episode_reward']:.3f} ± {stats['std_episode_reward']:.3f}")
        print(f"Min/Max Episode Reward: {stats['min_episode_reward']:.3f} / {stats['max_episode_reward']:.3f}")
        print(f"Mean Episode Length: {stats['mean_episode_length']:.1f}")
        print(f"Mean Plunger Reward: {stats['mean_plunger_reward']:.3f}")
        print(f"Mean Barrier Reward: {stats['mean_barrier_reward']:.3f}")
        
        total_time = time.time() - start_time
        print(f"Total Runtime: {total_time:.1f}s ({stats['total_episodes']/total_time:.2f} eps/s)")
        print("="*60)
        
        # Log final stats to wandb
        if use_wandb:
            wandb.log({
                'final_mean_episode_reward': stats['mean_episode_reward'],
                'final_std_episode_reward': stats['std_episode_reward'],
                'final_mean_episode_length': stats['mean_episode_length'],
                'final_mean_plunger_reward': stats['mean_plunger_reward'],
                'final_mean_barrier_reward': stats['mean_barrier_reward'],
                'total_runtime': total_time,
                'episodes_per_second': stats['total_episodes']/total_time,
            })
        
        env.close()
        return stats
        
    finally:
        if ray.is_initialized():
            ray.shutdown()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Random policy rollout for quantum device tuning')
    
    parser.add_argument(
        '--num-episodes', 
        type=int, 
        default=100,
        help='Number of episodes to run with random policy (default: 100)'
    )
    
    parser.add_argument(
        '--disable-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    
    parser.add_argument(
        '--project-name',
        type=str,
        default='RLModel-Random-Baseline',
        help='Wandb project name (default: RLModel-Random-Baseline)'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Load same config as PPO training
    config = load_config()
    
    use_wandb = not args.disable_wandb
    
    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            entity=config['wandb']['entity'],
            project=args.project_name,
            config={
                **config,
                'policy_type': 'random',
                'num_episodes': args.num_episodes,
            },
            tags=['random_policy', 'baseline']
        )
    
    try:
        # Run random episodes
        stats = run_random_episodes(
            num_episodes=args.num_episodes,
            config=config,
            use_wandb=use_wandb
        )
        
        print(f"\nRandom policy baseline completed successfully!")
        print(f"Results can be compared with PPO training runs in wandb.")
        
    finally:
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()