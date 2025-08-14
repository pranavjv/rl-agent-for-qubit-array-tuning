import torch
import torch.nn as nn
import numpy as np

from typing import Optional, Sequence, Type, Dict
ModuleType = Optional[Type[nn.Module]]

from model.base_classes import MLP, CNN
from model.rssm import RecurrentModel


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
        return action, dist


class Agent(nn.Module):
    """
    main agent class with feature extractor, quality predictor and actor
    """
    def __init__(
        self,
        input_channels: int,
        image_size: int = 128,
        latent_dim: int = 256,
        action_dim: int = 1,
        recurrent_dim: int = 128,
        feature_dim: int = 256,
        cnn_hidden_dims: Sequence[int] = [16, 32, 64, 64],
        actor_hidden_dims: Sequence[int] = [128, 128, 64, 32],
    ) -> None:
        super().__init__()
        print('-'*40)

        self.actor = Actor(
            feature_dim=feature_dim,
            action_dim=action_dim,
            recurrent_dim=recurrent_dim,
            hidden_dims=actor_hidden_dims,
            latent_dim=latent_dim,
        )
        print(f'Initialised actor with {sum(p.numel() for p in self.actor.parameters())} parameters')

        self.feature_extractor = CNN(
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
        print(f'Initialised feature extractor with {sum(p.numel() for p in self.feature_extractor.parameters())} parameters')

        self.quality_head = MLP(
            input_dim=feature_dim,
            hidden_dims=[128, 64, 64],
            output_dim=1
        )
        print(f'Initialised quality head with {sum(p.numel() for p in self.quality_head.parameters())} parameters')

        self.rssm = RecurrentModel(
            input_dim=latent_dim,
            hidden_dim=recurrent_dim,
        )
        print(f'Initialised RSSM with {sum(p.numel() for p in self.rssm.parameters())} parameters')

        self.recurrent_state = torch.zeros(recurrent_dim, dtype=torch.float32).unsqueeze(0)

        print(f'Agent initialised with {sum(p.numel() for p in self.parameters())} parameters')
        print('-'*40)


    def predict_action(self, obs: torch.Tensor, update_recurrent: bool = True) -> torch.Tensor:
        feats = self.encode_image(obs)
        if update_recurrent:
            self.recurrent_state = self.rssm(self.recurrent_state, feats)
        action, _ = self.actor(feats, self.recurrent_state)
        return action

    def encode_image(self, obs: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(obs)

    def get_quality(self, obs: torch.Tensor) -> torch.Tensor:
        feats = self.encode_image(obs)
        logits = self.quality_head(feats)
        return torch.sigmoid(logits)

    def forward(self, obs: torch.Tensor, update_recurrent: bool = True) -> torch.Tensor:
        feats = self.encode_image(obs)
        if update_recurrent:
            self.recurrent_state = self.rssm(self.recurrent_state, feats)
        quality_logits = self.quality_head(feats)
        action, dist = self.actor(feats, self.recurrent_state)
        return action, quality_logits


if __name__ == '__main__':
    agent = Agent(input_channels=2)
    x = torch.randn(1, 2, 128, 128)
    action, quality_logits = agent(x)
    print(action.shape)
    print(quality_logits.shape)