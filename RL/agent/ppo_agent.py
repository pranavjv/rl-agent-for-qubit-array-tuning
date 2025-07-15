import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
import yaml
import os
from typing import Dict, List, Tuple, Optional
import time
from tqdm import tqdm
import psutil

try:
    from .networks.multi_modal_net import ActorCriticNetwork
    from .utils.reward_utils import (
        compute_advantages, compute_returns, normalize_advantages,
        compute_policy_loss, compute_value_loss, compute_entropy_bonus,
        compute_kl_divergence
    )
except ImportError:
    # Fallback for direct script execution, was getting an error when running scripts directly
    from networks.multi_modal_net import ActorCriticNetwork
    from utils.reward_utils import (
        compute_advantages, compute_returns, normalize_advantages,
        compute_policy_loss, compute_value_loss, compute_entropy_bonus,
        compute_kl_divergence
    )


class PPOAgent:
    """
    PPO agent for the qarray env
    """
    
    def __init__(self, env, config_path: str = "config/ppo_config.yaml"):
        """
        Initialize PPO agent.
        
        Args:
            env: Gymnasium environment
            config_path: Path to configuration file
        """
        self.env = env
        self.config = self._load_config(config_path)
        
        # Set device
        self.device = self._setup_device()
        
        # Initialize network
        self.network = self._create_network()
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(
            self.network.parameters(),
            lr=float(self.config['ppo']['actor_lr']),
            betas=(float(self.config['optimization']['beta1']), float(self.config['optimization']['beta2'])),
            eps=float(self.config['optimization']['eps'])
        )
        
        # Initialize action distribution parameters
        self.log_std = nn.Parameter(torch.zeros(self.config['network']['action_dim']))
        self.actor_optimizer.add_param_group({
            'params': [self.log_std],
            'lr': float(self.config['ppo']['actor_lr'])
        })
        
        # Training state
        self.total_timesteps = 0
        self.iteration = 0
        self.best_reward = float('-inf')
        self.save_final_model = self.config['logging'].get('save_final_model', True)
        
        # Debug info
        if self.config['debug']['print_network_info']:
            self._print_network_info()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _setup_device(self) -> torch.device:
        """Setup device (CPU/GPU)."""
        if self.config['device']['use_cuda'] and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.config['device']['device_id']}")
            print(f"Using CUDA device: {device}")
        else:
            device = torch.device("cpu")
            print("Using CPU device")
        
        return device
    
    def _create_network(self) -> ActorCriticNetwork:
        """Create the actor-critic network."""
        network_config = self.config['network']
        
        network = ActorCriticNetwork(
            image_shape=tuple(network_config['image_shape']),
            voltage_dim=int(network_config['voltage_dim']),
            hidden_dim=int(network_config['fusion_hidden_dim']),
            action_dim=int(network_config['action_dim'])
        ).to(self.device)
        
        return network
    
    def _print_network_info(self):
        """Print network architecture information. Useful for debugging."""
        print("Network Architecture:")
        print(f"  Image shape: {self.config['network']['image_shape']}")
        print(f"  Voltage dim: {self.config['network']['voltage_dim']}")
        print(f"  Action dim: {self.config['network']['action_dim']}")
        print(f"  Hidden dim: {self.config['network']['fusion_hidden_dim']}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
    
    def _prepare_observation(self, observation: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare observation for network input by moving to device and adding batch dimension.
        
        Args:
            observation: Multi-modal observation as torch tensors
            
        Returns:
            dict: Observation ready for network input
        """
        prepared = {}
        
        for key, tensor in observation.items():
            # Move to device
            tensor = tensor.to(self.device)
            
            # Add batch dimension if not present
            if tensor.ndim == 3:  # (H, W, C) for image
                tensor = tensor.unsqueeze(0)  # (1, H, W, C)
            elif tensor.ndim == 1:  # (N,) for voltages
                tensor = tensor.unsqueeze(0)  # (1, N)
            
            prepared[key] = tensor
        
        return prepared
    
    def _batch_observations(self, observations: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Batch multiple observations into a single tensor batch.
        
        Args:
            observations: List of observation dictionaries with torch tensors
            
        Returns:
            dict: Batched observations
        """
        if not observations:
            raise ValueError("Empty observation list")
        
        batched = {}
        
        # Get keys from first observation
        keys = observations[0].keys()
        
        for key in keys:
            # Collect tensors for this key
            tensors = []
            for obs in observations:
                tensor = obs[key].to(self.device)
                # Add batch dimension if not present
                if tensor.ndim == 3:  # (H, W, C) for image
                    tensor = tensor.unsqueeze(0)  # (1, H, W, C)
                elif tensor.ndim == 1:  # (N,) for voltages
                    tensor = tensor.unsqueeze(0)  # (1, N)
                tensors.append(tensor)
            
            # Concatenate along batch dimension
            batched[key] = torch.cat(tensors, dim=0)
        
        return batched
    
    def get_action(self, observation: Dict[str, torch.Tensor]) -> Tuple[np.ndarray, float, float]:
        """
        Get action from current policy.
        
        Args:
            observation: Multi-modal observation as torch tensors
            
        Returns:
            action: Action to take
            log_prob: Log probability of the action
            value: State value estimate
        """
        # Move observation to device and add batch dimension if needed
        obs_tensor = self._prepare_observation(observation)
        
        # Get network outputs
        with torch.no_grad():
            actor_output, critic_output = self.network(obs_tensor)
            value = critic_output.squeeze().detach().cpu().numpy()
        
        # Create action distribution
        std = torch.exp(self.log_std)
        dist = Normal(actor_output, std)
        
        # Sample action
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Clip action to valid range
        action_low = torch.tensor(self.env.action_space.low, device=self.device)
        action_high = torch.tensor(self.env.action_space.high, device=self.device)
        action = torch.clamp(action, action_low, action_high)
        
        # Always return a 1D action
        action_np = action.cpu().numpy().squeeze()
        if action_np.ndim > 1:
            action_np = action_np.flatten()
        return action_np, log_prob.detach().cpu().numpy(), value
    
    def collect_rollout(self) -> Tuple[List[Dict[str, torch.Tensor]], List[torch.Tensor], List, List[torch.Tensor], List]:
        """
        Collect a batch of trajectories.
        
        Returns:
            observations: List of observation dictionaries with torch tensors
            actions: List of action tensors
            rewards: List of rewards
            log_probs: List of log probability tensors
            values: List of value estimates
        """
        observations, actions, rewards, log_probs, values = [], [], [], [], []
        
        timesteps_collected = 0
        timesteps_per_batch = self.config['ppo']['timesteps_per_batch']
        max_timesteps_per_episode = self.config['ppo']['max_timesteps_per_episode']
        
        # Progress bar for rollout collection
        rollout_pbar = tqdm(total=timesteps_per_batch, desc="Collecting rollout", unit="timesteps", leave=False)
        
        episode_count = 0
        while timesteps_collected < timesteps_per_batch:
            episode_count += 1
            obs, _ = self.env.reset()
            episode_rewards = []
            episode_observations = []
            episode_actions = []
            episode_log_probs = []
            episode_values = []
            
            episode_timesteps = 0
            for step in range(max_timesteps_per_episode):
                # Get action from policy
                action, log_prob, value = self.get_action(obs)
                
                # Take action in environment
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                
                # Store transition
                episode_observations.append(obs)
                episode_actions.append(torch.tensor(action, dtype=torch.float32))
                episode_rewards.append(reward)
                episode_log_probs.append(torch.tensor(log_prob, dtype=torch.float32))
                episode_values.append(value)
                
                obs = next_obs
                timesteps_collected += 1
                episode_timesteps += 1
                
                # Check if episode is done
                if terminated or truncated:
                    break
                
                # Check if we've collected enough timesteps
                if timesteps_collected >= timesteps_per_batch:
                    break
            
            # Update progress bar
            rollout_pbar.update(episode_timesteps)
            rollout_pbar.set_postfix({
                'episodes': episode_count,
                'collected': timesteps_collected,
                'target': timesteps_per_batch
            })
            
            # Extend batch with episode data
            observations.extend(episode_observations)
            actions.extend(episode_actions)
            rewards.extend(episode_rewards)
            log_probs.extend(episode_log_probs)
            values.extend(episode_values)
        
        rollout_pbar.close()
        return observations, actions, rewards, log_probs, values
    
    def update_policy(self, observations: List[Dict[str, torch.Tensor]], actions: List[torch.Tensor], 
                     old_log_probs: List[torch.Tensor], returns: List, advantages: List):
        """
        Update policy using PPO.
        
        Args:
            observations: Batch of observation dictionaries with torch tensors
            actions: Batch of action tensors
            old_log_probs: Batch of old log probability tensors
            returns: Batch of returns
            advantages: Batch of advantages
            
        Returns:
            dict: Training statistics including KL divergence
        """
        # Convert to tensors
        obs_batch = self._batch_observations(observations)
        actions_tensor = torch.stack(actions).to(self.device)
        old_log_probs_tensor = torch.stack(old_log_probs).to(self.device)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        
        # Normalize advantages if configured
        if self.config['ppo']['normalize_advantages']:
            advantages_tensor = torch.tensor(normalize_advantages(advantages_tensor.cpu().numpy()), device=self.device)
        
        # Multiple update iterations
        updates_per_iteration = self.config['ppo']['updates_per_iteration']
        max_kl_div = self.config['ppo']['max_kl_div']
        
        # Training statistics
        training_stats = {
            'policy_losses': [],
            'value_losses': [],
            'entropy_bonuses': [],
            'kl_divergences': [],
            'early_stop_update': None
        }
        
        # Progress bar for policy updates
        update_pbar = tqdm(range(updates_per_iteration), desc="Policy updates", leave=False)
        
        for update in update_pbar:
            # Forward pass
            actor_output, critic_output = self.network(obs_batch)
            
            # Create action distribution
            std = torch.exp(self.log_std)
            dist = Normal(actor_output, std)
            
            # Compute new log probabilities
            new_log_probs = dist.log_prob(actions_tensor).sum(dim=-1)
            
            # Compute KL divergence for monitoring
            kl_div = compute_kl_divergence(old_log_probs_tensor, new_log_probs)
            training_stats['kl_divergences'].append(kl_div.item())
            
            # Early stopping if KL divergence is too high
            if kl_div > max_kl_div:
                if self.config['debug']['verbose']:
                    print(f"  Early stopping at update {update + 1}/{updates_per_iteration} "
                          f"due to high KL divergence: {kl_div.item():.4f} > {max_kl_div}")
                training_stats['early_stop_update'] = update + 1
                update_pbar.close()
                break
            
            # Compute losses
            policy_loss = compute_policy_loss(
                advantages_tensor, old_log_probs_tensor, new_log_probs,
                self.config['ppo']['epsilon']
            )
            
            value_loss = compute_value_loss(
                critic_output.squeeze(), returns_tensor,
                self.config['ppo']['value_clip_ratio']
            )
            
            entropy_bonus = compute_entropy_bonus(
                new_log_probs, self.config['ppo']['entropy_coef']
            )
            
            # Store losses for logging
            training_stats['policy_losses'].append(policy_loss.item())
            training_stats['value_losses'].append(value_loss.item())
            training_stats['entropy_bonuses'].append(entropy_bonus.item())
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss - entropy_bonus
            
            # Backward pass
            self.actor_optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), 
                self.config['optimization']['max_grad_norm']
            )
            
            self.actor_optimizer.step()
        
        update_pbar.close()
        return training_stats
    
    def train(self):
        """Main training loop."""
        total_timesteps = self.config['ppo']['total_timesteps']
        log_interval = self.config['logging']['log_interval']
        
        print(f"Starting training for {total_timesteps} timesteps...")
        print(f"Configuration:")
        print(f"  - Timesteps per batch: {self.config['ppo']['timesteps_per_batch']}")
        print(f"  - Max timesteps per episode: {self.config['ppo']['max_timesteps_per_episode']}")
        print(f"  - Updates per iteration: {self.config['ppo']['updates_per_iteration']}")
        print()
        
        # Memory monitoring function
        def get_memory_usage():
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # MB
        
        # Create progress bar for overall training
        
        while self.total_timesteps < total_timesteps:
            # Collect rollout with progress indication
            start_time = time.time()
            observations, actions, rewards, log_probs, values = self.collect_rollout()
            rollout_time = time.time() - start_time
            
            # Calculate rollout statistics
            timesteps_collected = len(rewards)
            num_episodes = 0 #not yet implemented
            mean_reward = np.mean(rewards)
            
            print(f"\n=== Iteration {self.iteration + 1} ===")
            print(f"Rollout Collection:")
            print(f"  - Timesteps collected: {timesteps_collected}")
            print(f"  - Episodes run: {num_episodes}")
            print(f"  - Mean reward: {mean_reward:.3f}")
            print(f"  - Time taken: {rollout_time:.2f}s")
            
            # Compute returns and advantages
            print("Computing returns and advantages...")
            start_time = time.time()
            returns = compute_returns(rewards, self.config['ppo']['gamma'])
            advantages = compute_advantages(
                rewards, values, 
                self.config['ppo']['gamma'], 
                self.config['ppo']['gae_lambda']
            )
            compute_time = time.time() - start_time
            print(f"  - Time taken: {compute_time:.2f}s")
            
            # Update policy with progress indication
            print("Updating policy...")
            start_time = time.time()
            training_stats = self.update_policy(observations, actions, log_probs, list(returns), list(advantages))
            update_time = time.time() - start_time
            
            # Calculate policy update statistics
            num_updates = len(training_stats['kl_divergences'])
            mean_kl_div = np.mean(training_stats['kl_divergences']) if training_stats['kl_divergences'] else 0.0
            mean_policy_loss = np.mean(training_stats['policy_losses']) if training_stats['policy_losses'] else 0.0
            mean_value_loss = np.mean(training_stats['value_losses']) if training_stats['value_losses'] else 0.0
            
            print(f"  - Policy updates performed: {num_updates}/{self.config['ppo']['updates_per_iteration']}")
            print(f"  - Mean KL divergence: {mean_kl_div:.4f}")
            print(f"  - Mean policy loss: {mean_policy_loss:.4f}")
            print(f"  - Mean value loss: {mean_value_loss:.4f}")
            print(f"  - Time taken: {update_time:.2f}s")
        
            # Update counters
            timesteps_this_iteration = len(rewards)
            self.total_timesteps += timesteps_this_iteration
            self.iteration += 1
            
            
            
            # Logging
            if self.iteration % log_interval == 0:
                mean_return = np.mean(returns)
                mean_advantage = np.mean(advantages)
                
                # KL divergence statistics
                max_kl_div = np.max(training_stats['kl_divergences']) if training_stats['kl_divergences'] else 0.0
                
                print(f"\n--- Logging Interval {self.iteration // log_interval} ---")
                print(f"  Timesteps: {self.total_timesteps:,}/{total_timesteps:,}")
                print(f"  Mean return: {mean_return:.3f}")
                print(f"  Mean advantage: {mean_advantage:.3f}")
                print(f"  Max KL divergence: {max_kl_div:.4f}")
                
                if training_stats['early_stop_update']:
                    print(f"  Early stopped at update {training_stats['early_stop_update']}")
                
                print()
                
                # Update best reward
                if mean_reward > self.best_reward:
                    self.best_reward = mean_reward
                    if self.config['logging']['save_best_only']:
                        self.save_model("best_model.pth")
        
        print("\nTraining completed!")
        print(f"Final statistics:")
        print(f"  - Total iterations: {self.iteration}")
        print(f"  - Total timesteps: {self.total_timesteps}")
        print(f"  - Best mean reward: {self.best_reward:.3f}")
        
        # Save final model
        if hasattr(self, 'save_final_model') and self.save_final_model:
            self.save_model("final_model.pth")
    
    def save_model(self, filename: str):
        """Save model to file."""
        model_dir = self.config['logging']['model_dir']
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, filename)
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'log_std': self.log_std.data,
            'config': self.config,
            'iteration': self.iteration,
            'total_timesteps': self.total_timesteps,
            'best_reward': self.best_reward
        }, model_path)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, filename: str):
        """Load model from file."""
        model_path = filename
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.log_std.data = checkpoint['log_std']
        
        self.iteration = checkpoint.get('iteration', 0)
        self.total_timesteps = checkpoint.get('total_timesteps', 0)
        self.best_reward = checkpoint.get('best_reward', float('-inf'))
        
        print(f"Model loaded from {model_path}")
    
    def evaluate(self, num_episodes: int = 10) -> Dict:
        """Evaluate the current policy."""
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action, _, _ = self.get_action(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'episode_rewards': episode_rewards
        } 