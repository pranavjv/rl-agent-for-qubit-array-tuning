"""Custom neural network components for quantum device RL agents."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ray.rllib.core.models.base import ENCODER_OUT, Encoder
from ray.rllib.core.models.configs import CNNEncoderConfig, MLPHeadConfig, ModelConfig
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


@dataclass
class QuantumCNNConfig(CNNEncoderConfig):
    """CNN configuration for quantum charge stability diagrams with clean YAML interface."""
    
    conv_layers: Optional[List[Dict]] = None
    feature_size: int = 256
    adaptive_pooling: bool = True
    
    def __post_init__(self):
        if self.conv_layers:
            self.cnn_filter_specifiers = [
                [layer["channels"], [layer["kernel"], layer["kernel"]], layer["stride"]]
                for layer in self.conv_layers
            ]
        else:
            self.cnn_filter_specifiers = [
                [16, [4, 4], 2],
                [32, [3, 3], 2], 
                [64, [3, 3], 1],
            ]
    
    @property
    def output_dims(self):
        return (self.feature_size,)
    
    def build(self, framework: str = "torch") -> "QuantumCNNEncoder":
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        return QuantumCNNEncoder(self)


class QuantumCNNEncoder(TorchModel, Encoder):
    """CNN encoder for quantum charge stability diagrams."""
    
    def __init__(self, config: QuantumCNNConfig):
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)
        
        self.config = config
        
        cnn_layers = []
        in_channels = config.input_dims[-1]
        
        for out_channels, kernel_size, stride in config.cnn_filter_specifiers:
            cnn_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=1),
                nn.ReLU() if config.cnn_activation == "relu" else nn.Tanh(),
            ])
            in_channels = out_channels
        
        if config.adaptive_pooling:
            cnn_layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        
        cnn_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*cnn_layers)
        
        self._calculate_cnn_output_size()
        
        self.final_mlp = nn.Sequential(
            nn.Linear(self._cnn_output_size, config.feature_size),
            nn.ReLU() if config.cnn_activation == "relu" else nn.Tanh(),
        )
        
        self._output_dims = (config.feature_size,)
    
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


@dataclass
class QuantumPolicyHeadConfig(MLPHeadConfig):
    """Policy head configuration for quantum device control."""
    
    hidden_layers: Optional[List[int]] = None
    activation: str = "relu"
    use_attention: bool = False
    
    def __post_init__(self):
        if self.hidden_layers:
            self.hidden_layer_dims = self.hidden_layers
        else:
            self.hidden_layer_dims = [128, 128]
        
        self.hidden_layer_activation = self.activation
        self.output_layer_activation = "linear"
    
    def build(self, framework: str = "torch") -> "QuantumPolicyHead":
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        return QuantumPolicyHead(self)


class QuantumPolicyHead(TorchModel):
    """Policy head for quantum device control with optional attention."""
    
    def __init__(self, config: QuantumPolicyHeadConfig):
        super().__init__(config)
        
        self.config = config
        
        layers = []
        in_dim = config.input_dims[0] if isinstance(config.input_dims, (list, tuple)) else config.input_dims
        
        for hidden_dim in config.hidden_layer_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU() if config.activation == "relu" else nn.Tanh(),
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, config.output_layer_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        if config.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=config.input_dims[0] if isinstance(config.input_dims, (list, tuple)) else config.input_dims,
                num_heads=4,
                batch_first=True
            )
        
        self._output_dims = (config.output_layer_dim,)
    
    @property
    def output_dims(self) -> Tuple[int, ...]:
        return self._output_dims
    
    def _forward(self, inputs, **kwargs):
        if self.config.use_attention and inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
            attended, _ = self.attention(inputs, inputs, inputs)
            inputs = attended.squeeze(1)
        
        return self.mlp(inputs)


@dataclass
class QuantumValueHeadConfig(MLPHeadConfig):
    """Value head configuration for quantum device RL."""
    
    hidden_layers: Optional[List[int]] = None
    activation: str = "relu"
    use_attention: bool = False
    
    def __post_init__(self):
        if self.hidden_layers:
            self.hidden_layer_dims = self.hidden_layers
        else:
            self.hidden_layer_dims = [128, 64]
        
        self.hidden_layer_activation = self.activation
        self.output_layer_activation = "linear"
        self.output_layer_dim = 1
    
    def build(self, framework: str = "torch") -> "QuantumValueHead":
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        return QuantumValueHead(self)


class QuantumValueHead(TorchModel):
    """Value head for quantum device RL with optional attention mechanism."""
    
    def __init__(self, config: QuantumValueHeadConfig):
        super().__init__(config)
        
        self.config = config
        
        layers = []
        in_dim = config.input_dims[0] if isinstance(config.input_dims, (list, tuple)) else config.input_dims
        
        for hidden_dim in config.hidden_layer_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU() if config.activation == "relu" else nn.Tanh(),
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        if config.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=config.input_dims[0] if isinstance(config.input_dims, (list, tuple)) else config.input_dims,
                num_heads=4,
                batch_first=True
            )
        
        self._output_dims = (1,)
    
    @property
    def output_dims(self) -> Tuple[int, ...]:
        return self._output_dims
    
    def _forward(self, inputs, **kwargs):
        if self.config.use_attention and inputs.dim() == 2:
            inputs = inputs.unsqueeze(1)
            attended, _ = self.attention(inputs, inputs, inputs)
            inputs = attended.squeeze(1)
        
        return self.mlp(inputs)