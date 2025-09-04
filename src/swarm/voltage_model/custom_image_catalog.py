"""Custom CNN catalog for quantum device image processing."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import gymnasium as gym
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.base import ENCODER_OUT, Encoder
from ray.rllib.core.models.configs import CNNEncoderConfig, ModelConfig
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


@dataclass
class CustomCNNConfig(CNNEncoderConfig):
    """CNN configuration for quantum charge stability diagrams."""

    quantum_filters: Optional[List] = None
    adaptive_pooling: bool = True
    final_feature_size: int = 256

    def __post_init__(self):
        if self.quantum_filters is None:
            self.cnn_filter_specifiers = [
                [16, [4, 4], 2],
                [32, [3, 3], 2],
                [64, [3, 3], 1],
            ]
        else:
            self.cnn_filter_specifiers = self.quantum_filters

    @property
    def output_dims(self):
        return (self.final_feature_size,)

    def build(self, framework: str = "torch") -> "CustomCNNEncoder":
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        return CustomCNNEncoder(self)


class CustomCNNEncoder(TorchModel, Encoder):
    """CNN encoder for quantum charge stability diagrams."""

    def __init__(self, config: CustomCNNConfig):
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        self.config = config

        cnn_layers = []
        in_channels = config.input_dims[-1]

        for out_channels, kernel_size, stride in config.cnn_filter_specifiers:
            cnn_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1),
                    nn.ReLU() if config.cnn_activation == "relu" else nn.Tanh(),
                ]
            )
            in_channels = out_channels

        if config.adaptive_pooling:
            cnn_layers.append(nn.AdaptiveAvgPool2d((4, 4)))

        cnn_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*cnn_layers)

        self._calculate_cnn_output_size()

        self.final_mlp = nn.Sequential(
            nn.Linear(self._cnn_output_size, config.final_feature_size),
            nn.ReLU() if config.cnn_activation == "relu" else nn.Tanh(),
        )

        self._output_dims = (config.final_feature_size,)

    def _calculate_cnn_output_size(self):
        h, w, c = self.config.input_dims
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            cnn_output = self.cnn(dummy_input)
            self._cnn_output_size = cnn_output.shape[1]

    @property
    def output_dims(self) -> Tuple[int, ...]:
        return self._output_dims

    def _forward(self, inputs, **kwargs):
        if isinstance(inputs, dict):
            if "obs" in inputs:
                x = inputs["obs"]
            elif len(inputs) == 1:
                x = next(iter(inputs.values()))
            else:
                raise ValueError(f"Unexpected input dict structure: {list(inputs.keys())}")
        else:
            x = inputs

        if x.dim() == 3:
            x = x.unsqueeze(0)

        if x.shape[-1] <= 8:
            x = x.permute(0, 3, 1, 2)

        cnn_features = self.cnn(x)
        output_features = self.final_mlp(cnn_features)

        return {ENCODER_OUT: output_features}


class CustomImageCatalog(PPOCatalog):
    """Custom catalog for quantum device image processing."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: dict,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict=model_config_dict,
        )

    @classmethod
    @override(PPOCatalog)
    def _get_encoder_config(
        cls,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
    ) -> ModelConfig:
        from gymnasium.spaces import Box

        use_lstm = model_config_dict.get("use_lstm", False)

        if use_lstm:
            from ray.rllib.core.models.configs import RecurrentEncoderConfig

            tokenizer_config = cls._get_encoder_config(
                observation_space=observation_space,
                model_config_dict={**model_config_dict, "use_lstm": False},
                action_space=action_space,
            )

            return RecurrentEncoderConfig(
                input_dims=tokenizer_config.output_dims,
                recurrent_layer_type="lstm",
                hidden_dim=model_config_dict.get("lstm_cell_size", 128),
                num_layers=1,
                batch_major=True,
                tokenizer_config=tokenizer_config,
                use_bias=True,
            )

        if isinstance(observation_space, Box) and len(observation_space.shape) == 3:
            return CustomCNNConfig(
                input_dims=observation_space.shape,
                cnn_activation=model_config_dict.get("conv_activation", "relu"),
                cnn_kernel_initializer=model_config_dict.get(
                    "conv_kernel_initializer", "xavier_uniform"
                ),
                cnn_kernel_initializer_config=model_config_dict.get(
                    "conv_kernel_initializer_kwargs", {}
                ),
                cnn_bias_initializer=model_config_dict.get("conv_bias_initializer", "zeros"),
                cnn_bias_initializer_config=model_config_dict.get(
                    "conv_bias_initializer_kwargs", {}
                ),
                adaptive_pooling=True,
                final_feature_size=model_config_dict.get("quantum_feature_size", 256),
                quantum_filters=model_config_dict.get("quantum_conv_filters", None),
            )
        else:
            return super()._get_encoder_config(
                observation_space=observation_space,
                model_config_dict=model_config_dict,
                action_space=action_space,
            )
