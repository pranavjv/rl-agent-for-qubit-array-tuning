import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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
        
        # Calculate the size after convolutions and pooling
        # Assuming input is (batch, channels, height, width)
        # This will be computed dynamically in forward pass
        self.output_dim = output_dim
        
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
        if x.size(1) != self.output_dim:
            x = nn.Linear(x.size(1), self.output_dim).to(x.device)(x)
        
        return x


class MultiModalNetwork(nn.Module):
    """
    Multi-modal neural network that processes both image and voltage observations.
    """
    def __init__(self, 
                 image_shape=(64, 64, 1), 
                 voltage_dim=2, 
                 hidden_dim=128, 
                 output_dim=2,
                 is_actor=True):
        super().__init__()
        
        self.image_shape = image_shape
        self.voltage_dim = voltage_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.is_actor = is_actor
        
        # CNN encoder for image data
        self.cnn_encoder = CNNEncoder(
            input_channels=image_shape[2], 
            output_dim=hidden_dim // 2
        )
        
        # MLP for voltage data
        self.voltage_encoder = nn.Sequential(
            nn.Linear(voltage_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Feature fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Output layer
        if is_actor:
            # Actor: output action means
            self.output_layer = nn.Linear(hidden_dim, output_dim)
        else:
            # Critic: output state value
            self.output_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, observation):
        """
        Forward pass with multi-modal observation.
        
        Args:
            observation (dict): Dictionary with 'image' and 'voltages' keys
                - image: torch.Tensor of shape (batch, height, width, channels)
                - voltages: torch.Tensor of shape (batch, voltage_dim)
        
        Returns:
            torch.Tensor: Network output (action means for actor, state value for critic)
        """
        # Process image through CNN
        image_features = self.cnn_encoder(observation['image'])
        
        # Process voltages through MLP
        voltage_features = self.voltage_encoder(observation['voltages'])
        
        # Concatenate features
        combined_features = torch.cat([image_features, voltage_features], dim=-1)
        
        # Fusion layer
        fused_features = self.fusion_layer(combined_features)
        
        # Output layer
        output = self.output_layer(fused_features)
        
        return output


class ActorCriticNetwork(nn.Module):
    """
    Combined actor-critic network for PPO.
    """
    def __init__(self, 
                 image_shape=(64, 64, 1), 
                 voltage_dim=2, 
                 hidden_dim=128, 
                 action_dim=2):
        super().__init__()
        
        # Shared feature extractors
        self.cnn_encoder = CNNEncoder(
            input_channels=image_shape[2], 
            output_dim=hidden_dim // 2
        )
        
        self.voltage_encoder = nn.Sequential(
            nn.Linear(voltage_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Shared fusion layer
        self.shared_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, observation):
        """
        Forward pass for both actor and critic.
        
        Args:
            observation (dict): Multi-modal observation
            
        Returns:
            tuple: (actor_output, critic_output)
        """
        # Shared feature extraction
        image_features = self.cnn_encoder(observation['image'])
        voltage_features = self.voltage_encoder(observation['voltages'])
        combined_features = torch.cat([image_features, voltage_features], dim=-1)
        shared_features = self.shared_fusion(combined_features)
        
        # Actor and critic outputs
        actor_output = self.actor_head(shared_features)
        critic_output = self.critic_head(shared_features)
        
        return actor_output, critic_output
    
    def get_actor_output(self, observation):
        """Get only actor output."""
        actor_output, _ = self.forward(observation)
        return actor_output
    
    def get_critic_output(self, observation):
        """Get only critic output."""
        _, critic_output = self.forward(observation)
        return critic_output 