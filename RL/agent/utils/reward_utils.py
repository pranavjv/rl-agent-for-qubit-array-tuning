import numpy as np
import torch
from typing import List, Tuple, Optional

#actual reward calc is in the environment.py file (should probably port it over to here at some point)

def compute_advantages(rewards: List[float], 
                       values: List[float], 
                       gamma: float = 0.95,
                       gae_lambda: float = 0.95) -> np.ndarray:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards (list): List of rewards for each timestep
        values (list): List of value estimates for each timestep
        gamma (float): Discount factor
        gae_lambda (float): GAE lambda parameter
        
    Returns:
        np.ndarray: Computed advantages
    """
    rewards_array = np.array(rewards)
    values_array = np.array(values)
    
    advantages = np.zeros_like(rewards_array)
    last_advantage = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            # Last timestep
            next_value = 0
        else:
            next_value = values[t + 1]
        
        # Compute TD error
        delta = rewards[t] + gamma * next_value - values[t]
        
        # Compute advantage using GAE
        advantages[t] = delta + gamma * gae_lambda * last_advantage
        last_advantage = advantages[t]
    
    return advantages


def compute_returns(rewards: List[float], gamma: float = 0.95) -> np.ndarray:
    """
    Compute discounted returns.
    
    Args:
        rewards (list): List of rewards
        gamma (float): Discount factor
        
    Returns:
        np.ndarray: Discounted returns
    """
    returns = np.zeros_like(rewards)
    running_return = 0
    
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    
    return returns


def normalize_advantages(advantages: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Normalize advantages to zero mean and unit variance.
    
    Args:
        advantages (np.ndarray): Raw advantages
        epsilon (float): Small value to prevent division by zero
        
    Returns:
        np.ndarray: Normalized advantages
    """
    mean = np.mean(advantages)
    std = np.std(advantages)
    return (advantages - mean) / (std + epsilon)


def compute_entropy_bonus(log_probs: torch.Tensor, entropy_coef: float = 0.01) -> torch.Tensor:
    """
    Compute entropy bonus for exploration.
    
    Args:
        log_probs (torch.Tensor): Log probabilities of actions
        entropy_coef (float): Entropy coefficient
        
    Returns:
        torch.Tensor: Entropy bonus
    """
    # Compute entropy: -sum(p * log(p))
    # Since we have log_probs, entropy = -sum(exp(log_probs) * log_probs)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy_coef * entropy


def compute_kl_divergence(old_log_probs: torch.Tensor, 
                         new_log_probs: torch.Tensor) -> torch.Tensor:
    """
    Compute KL divergence between old and new policy distributions.
    
    Args:
        old_log_probs (torch.Tensor): Log probabilities from old policy
        new_log_probs (torch.Tensor): Log probabilities from new policy
        
    Returns:
        torch.Tensor: KL divergence
    """
    # KL divergence = E[log(old_probs) - log(new_probs)]
    # Since we have log_probs, KL = E[old_log_probs - new_log_probs]
    kl_div = torch.mean(old_log_probs - new_log_probs)
    return kl_div


def compute_policy_loss(advantages: torch.Tensor,
                       old_log_probs: torch.Tensor,
                       new_log_probs: torch.Tensor,
                       epsilon: float = 0.2) -> torch.Tensor:
    """
    Compute PPO policy loss with clipping.
    
    Args:
        advantages (torch.Tensor): Advantage estimates
        old_log_probs (torch.Tensor): Log probabilities from old policy
        new_log_probs (torch.Tensor): Log probabilities from new policy
        epsilon (float): Clipping parameter
        
    Returns:
        torch.Tensor: Policy loss
    """
    # Compute probability ratios
    ratio = torch.exp(new_log_probs - old_log_probs)
    
    # Compute clipped surrogate loss
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
    
    # Take minimum of clipped and unclipped surrogate
    policy_loss = -torch.min(surr1, surr2).mean()
    
    return policy_loss


def compute_value_loss(predicted_values: torch.Tensor,
                      target_values: torch.Tensor,
                      clip_ratio: float = 0.2) -> torch.Tensor:
    """
    Compute PPO value loss with clipping.
    
    Args:
        predicted_values (torch.Tensor): Predicted values from critic
        target_values (torch.Tensor): Target values (returns)
        clip_ratio (float): Clipping parameter for value loss
        
    Returns:
        torch.Tensor: Value loss
    """
    # Compute unclipped value loss
    value_loss_unclipped = torch.mean((predicted_values - target_values) ** 2)
    
    # Compute clipped value loss
    value_pred_clipped = torch.clamp(
        predicted_values, 
        target_values - clip_ratio, 
        target_values + clip_ratio
    )
    value_loss_clipped = torch.mean((value_pred_clipped - target_values) ** 2)
    
    # Take maximum of clipped and unclipped loss
    value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
    
    return value_loss 