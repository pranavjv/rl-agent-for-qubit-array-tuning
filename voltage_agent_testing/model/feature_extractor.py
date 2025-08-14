
import torch
import torch.nn as nn
import numpy as np

from typing import Optional, Sequence, Type, Dict
ModuleType = Optional[Type[nn.Module]]

from utils import create_layers, miniblock


class CNNFeatureExtractor(nn.Module):
    """
    CNN feature extractor from an image to a latent vector
    on 2x128x128 size it returns a 256-d vector
    """
    def __init__(
        self,
        input_channels: int,
        hidden_channels: Sequence[int],
        kernel_sizes: Sequence[int],
        paddings: Optional[Sequence[int]] = None,
        strides: Optional[Sequence[int]] = None,
        cnn_layer: ModuleType = nn.Conv2d,
        dropout_layer: Optional[ModuleType] = None,
        dropout_p: Optional[float] = None,
        activation: ModuleType = nn.ReLU,
        norm_layer: Optional[ModuleType] = None,
        norm_args: Optional[dict] = None
    ) -> None:
        super().__init__()

        self.cnn = self._build_cnn(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_sizes=kernel_sizes,
            paddings=paddings,
            strides=strides,
            cnn_layer=cnn_layer,
            dropout_layer=dropout_layer,
            dropout_p=dropout_p,
            activation=activation,
            norm_layer=norm_layer,
            norm_args=norm_args,
        )

    def _build_cnn(
        self,
        input_channels,
        hidden_channels,
        kernel_sizes,
        paddings,
        strides,
        cnn_layer,
        dropout_layer,
        dropout_p,
        activation,
        norm_layer,
        norm_args,
    ) -> ModuleType:
        num_layers = len(hidden_channels)
        assert len(kernel_sizes) == num_layers

        dropout_layers, dropout_args = create_layers(dropout_layer, dropout_p, num_layers)
        norm_layers, norm_args = create_layers(norm_layer, norm_args, num_layers)
        activation_layers, act_args = create_layers(activation, None, num_layers)

        layer_args_list = [{}] * num_layers

        hidden_sizes = [input_channels] + list(hidden_channels)
        model = []
        for in_dim, out_dim, k_dim, pad, stride, l_args, drop, drop_args, norm, norm_args, activ, act_args in zip(
            hidden_sizes[:-1],
            hidden_sizes[1:],
            kernel_sizes,
            paddings if paddings is not None else [0]*num_layers,
            strides if strides is not None else [1]*num_layers,
            layer_args_list,
            dropout_layers,
            dropout_args,
            norm_layers,
            norm_args,
            activation_layers,
            act_args,
        ):

            l_args.update({"kernel_size": k_dim, "padding": pad, "stride": stride})
            if isinstance(norm_args, Dict):
                norm_args.update({"num_features": out_dim})
            else:
                norm_args = {"num_features": out_dim}
            mb = miniblock(in_dim, out_dim, cnn_layer, l_args, drop, drop_args, norm, norm_args, activ, act_args)

            model += mb

        self._output_dim = hidden_sizes[-1]
        return nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x).reshape(x.size(0), -1)


if __name__ == '__main__':
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

    x = torch.randn(1, 2, 128, 128)
    out = cnn_feature_extractor(x)
    print(out.shape)