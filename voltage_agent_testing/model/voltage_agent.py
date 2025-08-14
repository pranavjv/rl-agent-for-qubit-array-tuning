import torch
import torch.nn as nn
import numpy as np

from typing import Optional, Sequence, Type, Dict
ModuleType = Optional[Type[nn.Module]]

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
        return action, dist


class Agent(nn.Module):
    """
    main agent class with feature extractor, quality predictor and actor
    """
    def __init__(
        self,
        image_size: int = 128,
        latent_dim: int = 256,
        action_dim: int = 1,
        recurrent_dim: int = 128,
        feature_dim: int = 256,
        cnn_hidden_dims: Sequence[int] = [16, 32, 64, 64],
        actor_hidden_dims: Sequence[int] = [32, 64, 128],
    ) -> None:
        super().__init__()

        self.actor = Actor(
            feature_dim=feature_dim,
            action_dim=action_dim,
            recurrent_dim=recurrent_dim,
            hidden_dims=actor_hidden_dims,
            latent_dim=latent_dim,
        )

        self.feature_extractor = CNN(
            input_channels=2,
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

        self.quality_head = MLP(
            input_dim=feature_dim,
            hidden_dims=[128, 64, 64],
            output_dim=1
        )

        self.rssm = RecurrentModel(
            input_dim=latent_dim,
            hidden_dim=recurrent_dim,
        )

        self.recurrent_state = torch.zeros(recurrent_dim, dtype=torch.float32)

    def predict_action(self, obs: torch.Tensor) -> torch.Tensor:
        feats = self.encode_image(obs)

    def encode_image(self, obs: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(obs)

    def get_quality(self, obs: torch.Tensor) -> torch.Tensor:
        feats = self.encode_image(obs)
        return self.quality_head(feats)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        feats = self.encode_image(obs)
        self.recurrent_state = self.rssm(self.recurrent_state, feats)
        quality, quality_logits = self.quality_head(feats)
        action = self.actor(feats, self.recurrent_state)
        return action, quality, quality_logits


if __name__ == '__main__':
    agent = Agent()