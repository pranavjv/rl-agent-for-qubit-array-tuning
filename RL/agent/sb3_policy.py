import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import DiagGaussianDistribution
from typing import Dict, Tuple, Type, Union


class CNNEncoder(nn.Module):
    """
    CNN encoder for processing charge sensor image data.
    """
    def __init__(self, input_channels=1, output_dim=64):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        
        # Pre-calculate output size for 128x128 input with 3 conv layers + pooling
        # 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        # After global avg pool: 64 features
        self.projection_layer = nn.Linear(64, output_dim)
        
    def forward(self, x):
        # x shape: (batch, height, width, channels) -> (batch, channels, height, width)
        if x.shape[-1] == 1:  # If channels is last dimension
            x = x.permute(0, 3, 1, 2)
        
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        
        # Project to desired output dimension
        x = self.projection_layer(x)
        
        return x


class QuantumMultiModalFeaturesExtractor(BaseFeaturesExtractor):
    """
    Multi-modal feature extractor for quantum device observations.
    """
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        
        # Extract dimensions from observation space
        image_space = observation_space.spaces['image']
        voltage_space = observation_space.spaces['voltages']
        
        image_shape = image_space.shape  # (128, 128, 1)
        voltage_dim = voltage_space.shape[0]  # 2
        
        # CNN encoder for image data
        self.cnn_encoder = CNNEncoder(
            input_channels=image_shape[2], 
            output_dim=features_dim // 2
        )
        
        # MLP for voltage data
        self.voltage_encoder = nn.Sequential(
            nn.Linear(voltage_dim, features_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim // 4, features_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with multi-modal observation.
        
        Args:
            observations: Dictionary with 'image' and 'voltages' keys
            
        Returns:
            torch.Tensor: Extracted features
        """
        # Process image through CNN
        image_features = self.cnn_encoder(observations['image'])
        
        # Process voltages through MLP
        voltage_features = self.voltage_encoder(observations['voltages'])
        
        # Concatenate features
        combined_features = torch.cat([image_features, voltage_features], dim=-1)
        
        # Fusion layer
        fused_features = self.fusion_layer(combined_features)
        
        return fused_features


class QuantumMultiModalPolicy(ActorCriticPolicy):
    """
    Custom policy for quantum device environment with multi-modal observations.
    """
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)
        
        # Create feature extractor
        self.features_extractor = QuantumMultiModalFeaturesExtractor(
            observation_space, 
            features_dim=128
        )
        
        # Actor head (policy) - outputs action means
        self.actor_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, action_space.shape[0])  # action_dim
        )
        
        # Critic head (value function)
        self.critic_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        # Learnable log standard deviation for continuous actions
        self.log_std = nn.Parameter(torch.zeros(action_space.shape[0]))
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass in policy.
        
        Args:
            obs: Observation tensor
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (actions, values, log_probs)
        """
        # Extract features
        features = self.features_extractor(obs)
        
        # Get action means and values
        latent_pi = self.actor_head(features)
        latent_vf = self.critic_head(features)
        
        # Create action distribution
        mean_actions = latent_pi
        std_actions = torch.exp(self.log_std)
        
        # Sample actions
        if deterministic:
            actions = mean_actions
        else:
            actions = torch.normal(mean_actions, std_actions)
        
        # Compute log probabilities
        log_probs = -0.5 * ((actions - mean_actions) / std_actions) ** 2 - torch.log(std_actions) - 0.5 * np.log(2 * np.pi)
        log_probs = log_probs.sum(dim=-1)
        
        return actions, latent_vf, log_probs
    
    def forward_actor(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass for actor only."""
        features = self.features_extractor(obs)
        return self.actor_head(features)
    
    def forward_critic(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass for critic only."""
        features = self.features_extractor(obs)
        return self.critic_head(features)
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given observations.
        
        Args:
            obs: Observation tensor
            actions: Action tensor
            
        Returns:
            Tuple of (values, log_probs, entropy)
        """
        features = self.features_extractor(obs)
        
        # Get action means and values
        latent_pi = self.actor_head(features)
        mean_actions = latent_pi
        std_actions = torch.exp(self.log_std)
        
        # Compute log probabilities
        log_probs = -0.5 * ((actions - mean_actions) / std_actions) ** 2 - torch.log(std_actions) - 0.5 * np.log(2 * np.pi)
        log_probs = log_probs.sum(dim=-1)
        
        # Compute entropy
        entropy = 0.5 * torch.log(2 * np.pi * std_actions ** 2) + 0.5
        entropy = entropy.sum(dim=-1)
        
        return self.critic_head(features), log_probs, entropy 