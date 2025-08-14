import torch
import torch.nn as nn
import numpy as np

from typing import Optional, Sequence, Type, Dict
ModuleType = Optional[Type[nn.Module]]


class QualityPredictor(nn.Module):
    """
    Quality score predictor from latent output of the feature extractor
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int = 1,
    ) -> None:
        super().__init__()

        self.net = self._build_net(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim
        )
        self.sigmoid = nn.Sigmoid()
    
    def _build_net(
        self,
        input_dim,
        hidden_dims,
        output_dim,
    ) -> ModuleType:
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        logit = self.net(latent)
        return logit, self.sigmoid(logit)


class Actor(nn.Module):
    """
    actor class for generating actions conditioned on the latent space
    """
    def __init__(
        self,
    ) -> None:
        super().__init__()


    def forward(self, latent: torch.Tensor, recurrent: torch.Tensor) -> torch.Tensor:
        pass


class Agent(nn.Module):
    """
    main agent class with feature extractor, quality predictor and actor
    """
    def __init__(
        self,
    ) -> None:
        super().__init__()


if __name__ == '__main__':
    from feature_extractor import CNNFeatureExtractor
    cnn_feature_extractor = CNNFeatureExtractor(
        input_channels=2,
        hidden_channels=[16, 32, 64, 64],
        kernel_sizes=[4, 4, 4, 4],
        strides=[4, 2, 2, 2],
        cnn_layer=nn.Conv2d,
        dropout_layer=nn.Dropout,
        dropout_p=0.1,
        activation=nn.ReLU,
        norm_layer=nn.BatchNorm2d,
        norm_args=None
    )

    quality_head = QualityPredictor(
        input_dim=256,
        hidden_dims=[128, 64, 64],
        output_dim=1
    )

    x = torch.randn(1, 2, 128, 128)
    latent = cnn_feature_extractor(x)
    print(latent.shape)
    _, quality = quality_head(latent)
    print(quality)