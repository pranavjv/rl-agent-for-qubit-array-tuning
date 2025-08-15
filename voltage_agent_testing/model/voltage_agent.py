import torch
import torch.nn as nn
import numpy as np

from typing import Optional, Sequence, Type, Dict, Tuple
ModuleType = Optional[Type[nn.Module]]

try:
    from model.base_classes import MLP, CNN
    from model.rssm import RecurrentModel
except ModuleNotFoundError:
    from base_classes import MLP, CNN
    from rssm import RecurrentModel


class Actor(nn.Module):
    """
    actor class for generating actions conditioned on the latent space and the recurrent state
    """
    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        recurrent_dim: int,
        hidden_dims: Sequence[int],
        latent_dim: int,
        min_log_std: float = -1.0,
        max_log_std: float = 1.0,
    ) -> None:
        super().__init__()

        self.mlp = MLP(
            input_dim=feature_dim + recurrent_dim,
            hidden_dims=hidden_dims,
            output_dim=latent_dim
        )

        self.mean_head = nn.Linear(latent_dim, action_dim)
        self.log_std_head = nn.Linear(latent_dim, action_dim)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, latent: torch.Tensor, recurrent: torch.Tensor, greedy: bool = False) -> torch.Tensor:
        x = torch.cat((latent, recurrent), dim=-1)
        x = self.mlp(x)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        if greedy:
            action = mean
        else:
            action = dist.rsample()
        logprob = dist.log_prob(action).sum(dim=-1, keepdim=True) 
        return action, dist, logprob


class Agent(nn.Module):
    """
    main agent class with feature extractor, quality predictor and actor
    """
    def __init__(
        self,
        device: torch.device,
        input_channels: int,
        action_dim: int = 1,
        image_size: int = 128,
        image_feature_dim: int = 512,
        num_input_voltages: int = 1,
        voltage_feature_dim: int = 32,
        actor_latent_dim: int = 512,
        recurrent_dim: int = 1024,
        cnn_hidden_dims: Sequence[int] = [16, 32, 64, 64],
        actor_hidden_dims: Sequence[int] = [128, 128, 64, 32],
        print_size: bool = False,
    ) -> None:
        super().__init__()
        if print_size:
            print('-'*40)

        feature_dim = image_feature_dim + voltage_feature_dim
        self.feature_dim = feature_dim

        self.actor = Actor(
            feature_dim=feature_dim,
            action_dim=action_dim,
            recurrent_dim=recurrent_dim,
            hidden_dims=actor_hidden_dims,
            latent_dim=actor_latent_dim,
        )
        if print_size:
            print(f'Initialised actor with {sum(p.numel() for p in self.actor.parameters())} parameters')

        self.image_feature_extractor = CNN(
            img_size=image_size,
            feature_dim=image_feature_dim,
            input_channels=input_channels,
            hidden_channels=cnn_hidden_dims,
            kernel_sizes=[4, 4, 4, 4],
            strides=[4, 2, 2, 2],
            cnn_layer=nn.Conv2d,
            dropout_layer=nn.Dropout,
            dropout_p=0.1,
            activation=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            norm_args=None
        )

        self.voltage_feature_extractor = MLP(
            input_dim=num_input_voltages,
            hidden_dims=[32, 64],
            output_dim=voltage_feature_dim
        )
        if print_size:
            print(f'Initialised feature extractor with {sum(p.numel() for p in self.image_feature_extractor.parameters()) + sum(p.numel() for p in self.voltage_feature_extractor.parameters())} parameters')

        self.quality_head = MLP(
            input_dim=feature_dim,
            hidden_dims=[128, 64, 64],
            output_dim=1
        ) # for predicting when the agent has reached the goal
        self.critic = MLP(
            input_dim=feature_dim,
            hidden_dims=[128, 64, 64],
            output_dim=1
        ) # for predicting expected rewards (or whatever our value function is)
        if print_size:
            print(f'Initialised critic/quality head with {sum(p.numel() for p in self.quality_head.parameters())} parameters')

        self.rssm = RecurrentModel(
            input_dim=feature_dim,
            hidden_dim=recurrent_dim,
        )
        if print_size:
            print(f'Initialised RSSM with {sum(p.numel() for p in self.rssm.parameters())} parameters')

        self.recurrent_state = torch.zeros(recurrent_dim, dtype=torch.float32).unsqueeze(0).to(device)

        if print_size:
            print(f'Agent initialised with {sum(p.numel() for p in self.parameters())} parameters')
            print('-'*40)
        
        self.to(device)


    def encode_obs(self, image: torch.Tensor, voltages: torch.Tensor) -> torch.Tensor:
        img_feats = self.image_feature_extractor(image)
        voltage_feats = self.voltage_feature_extractor(voltages)
        return torch.cat([img_feats, voltage_feats], dim=-1)

    def predict_action(self, feats: torch.Tensor, update_recurrent: bool = True) -> torch.Tensor:
        if update_recurrent:
            self.recurrent_state = self.rssm(self.recurrent_state, feats)
        action, _, _ = self.actor(feats, self.recurrent_state)
        return action

    def get_quality(self, feats: torch.Tensor) -> torch.Tensor:
        logits = self.quality_head(feats)
        return torch.softmax(logits)

    def forward_step(self, feats: torch.Tensor, update_recurrent: bool = True) -> Tuple[torch.Tensor]:
        if update_recurrent:
            self.recurrent_state = self.rssm(self.recurrent_state, feats)
        quality_logits = self.quality_head(feats)
        value = self.critic(feats)
        action, dist, logprob = self.actor(feats, self.recurrent_state)
        return action, dist, logprob, quality_logits, value

    def forward(self, image: torch.Tensor, voltages: torch.Tensor, update_recurrent: bool = True) -> Tuple[torch.Tensor]:
        feats = self.encode_obs(image, voltages)
        if update_recurrent:
            self.recurrent_state = self.rssm(self.recurrent_state, feats)
        quality_logits = self.quality_head(feats)
        action, dist, logprob = self.actor(feats, self.recurrent_state)
        return action, logprob, quality_logits


if __name__ == '__main__':
    agent = Agent(input_channels=2, print_size=True)
    x = torch.randn(1, 2, 128, 128)
    action, quality_logits = agent(x)
    print(action.shape)
    print(quality_logits.shape)
