"""
Custom Image Catalog for Quantum Device RL Training

This module provides a custom Ray RLlib catalog specifically designed for processing
quantum charge stability diagram images. It extends Ray's default catalog functionality
with a CNN architecture optimized for quantum device image patterns.

Key Features:
- Custom CNN encoder config for quantum device images  
- Compatible with Ray RLlib 2.49.0 API
- Maintains all default PPO functionality
- Supports both LSTM and non-LSTM configurations
- Handles various image dimensions automatically
"""

import gymnasium as gym
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

from ray.rllib.core.models.catalog import Catalog
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.configs import CNNEncoderConfig, ModelConfig
from ray.rllib.core.models.base import Encoder, ENCODER_OUT
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.models.utils import get_filter_config

torch, nn = try_import_torch()


@dataclass
class CustomCNNConfig(CNNEncoderConfig):
    """
    CNN Encoder configuration optimized for quantum charge stability diagrams.
    
    This extends Ray's CNNEncoderConfig with quantum-specific architectural choices:
    - Smaller kernels for fine-grained quantum features
    - Appropriate pooling for charge stability patterns  
    - Adaptive sizing for various image dimensions
    """
    
    # Override default conv filters for quantum images
    quantum_filters: Optional[List] = None
    adaptive_pooling: bool = True
    final_feature_size: int = 256
    
    def __post_init__(self):
        """Set quantum-optimized CNN architecture if not specified."""
        if self.quantum_filters is None:
            # Custom filter configuration optimized for quantum charge stability diagrams
            # These filters are designed to capture:
            # 1. Fine-grained charge transitions (small kernels)
            # 2. Stability regions (medium kernels) 
            # 3. Global charge patterns (larger receptive fields)
            self.cnn_filter_specifiers = [
                # Layer 1: Detect fine charge transitions  
                [16, [4, 4], 2],  # 16 filters, 4x4 kernel, stride 2
                # Layer 2: Capture stability regions
                [32, [3, 3], 2],  # 32 filters, 3x3 kernel, stride 2  
                # Layer 3: Global pattern recognition
                [64, [3, 3], 1],  # 64 filters, 3x3 kernel, stride 1
            ]
        else:
            self.cnn_filter_specifiers = self.quantum_filters
    
    @property
    def output_dims(self):
        """Override parent to return our final MLP output dimensions."""
        return (self.final_feature_size,)
    
    def build(self, framework: str = "torch") -> "CustomCNNEncoder":
        """Build the quantum-optimized CNN encoder."""
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
            
        return CustomCNNEncoder(self)


class CustomCNNEncoder(TorchModel, Encoder):
    """
    CNN encoder specifically designed for quantum charge stability diagram images.
    
    This encoder processes quantum device images with an architecture optimized for:
    - Charge transition detection
    - Stability region identification  
    - Multi-dot correlation patterns
    """
    
    def __init__(self, config: CustomCNNConfig):
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)
        
        self.config = config
        
        # Build CNN layers from filter specifications
        cnn_layers = []
        in_channels = config.input_dims[-1]  # Channels are last in input_dims
        
        for i, (out_channels, kernel_size, stride) in enumerate(config.cnn_filter_specifiers):
            # Convolutional layer
            conv_layer = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels, 
                kernel_size=kernel_size,
                stride=stride,
                padding=1  # Keep spatial dimensions manageable
            )
            cnn_layers.append(conv_layer)
            
            # Activation function
            if config.cnn_activation == "relu":
                cnn_layers.append(nn.ReLU())
            elif config.cnn_activation == "tanh": 
                cnn_layers.append(nn.Tanh())
            else:
                cnn_layers.append(nn.ReLU())  # Default fallback
                
            in_channels = out_channels
        
        # Add adaptive pooling if requested
        if config.adaptive_pooling:
            # Adaptive pooling ensures consistent output size regardless of input dimensions
            cnn_layers.append(nn.AdaptiveAvgPool2d((4, 4)))  # Pool to 4x4 spatial size
        
        cnn_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Calculate CNN output size
        self._calculate_cnn_output_size()
        
        # Final MLP to produce desired feature size
        self.final_mlp = nn.Sequential(
            nn.Linear(self._cnn_output_size, config.final_feature_size),
            nn.ReLU() if config.cnn_activation == "relu" else nn.Tanh(),
        )
        
        self._output_dims = (config.final_feature_size,)
    
    def _calculate_cnn_output_size(self):
        """Calculate the output size of CNN layers using a dummy forward pass."""
        # Create dummy input with correct dimensions (B, C, H, W)
        h, w, c = self.config.input_dims
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            cnn_output = self.cnn(dummy_input)
            self._cnn_output_size = cnn_output.shape[1]
    
    @property
    def output_dims(self) -> Tuple[int, ...]:
        """Return output dimensions as tuple."""
        return self._output_dims
    
    def _forward(self, inputs, **kwargs):
        """
        Forward pass through the quantum CNN encoder.
        
        Args:
            inputs: Input tensor of shape (B, H, W, C) or dict containing such tensor
        
        Returns:
            Dict with ENCODER_OUT key containing encoded features
        """
        # Handle different input formats
        if isinstance(inputs, dict):
            # Extract actual image tensor from dict (e.g., from tokenizer)
            if "obs" in inputs:
                x = inputs["obs"] 
            elif len(inputs) == 1:
                x = next(iter(inputs.values()))
            else:
                raise ValueError(f"Unexpected input dict structure: {list(inputs.keys())}")
        else:
            x = inputs
            
        # Handle batch dimension
        if x.dim() == 3:  # Add batch dimension if missing
            x = x.unsqueeze(0)
        
        # Convert from (B, H, W, C) to (B, C, H, W) for PyTorch conv layers
        if x.shape[-1] <= 8:  # Assume channels are last if reasonable channel count
            x = x.permute(0, 3, 1, 2)
        
        # CNN processing
        cnn_features = self.cnn(x)
        
        # Final MLP processing
        output_features = self.final_mlp(cnn_features)
        
        return {ENCODER_OUT: output_features}


class CustomImageCatalog(PPOCatalog):
    """
    Custom catalog for quantum device image processing.
    
    This catalog extends Ray's default catalog to provide quantum-optimized CNN encoders
    for charge stability diagram images while preserving all other RLlib functionality.
    """
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space, 
        model_config_dict: dict,
    ):
        """Initialize the custom image catalog."""
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict=model_config_dict
        )
    
    @classmethod
    @override(PPOCatalog)
    def _get_encoder_config(
        cls,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
    ) -> ModelConfig:
        """
        Override encoder config generation to use quantum-optimized CNN for 3D Box spaces.
        
        This method:
        1. Detects 3D Box observation spaces (images)
        2. Returns CustomCNNConfig for quantum charge stability diagrams
        3. Falls back to parent PPOCatalog implementation for other space types
        """
        from gymnasium.spaces import Box
        
        # Check if we need LSTM wrapper
        use_lstm = model_config_dict.get("use_lstm", False)
        
        if use_lstm:
            # For LSTM case, we need RecurrentEncoderConfig with our quantum CNN as tokenizer
            from ray.rllib.core.models.configs import RecurrentEncoderConfig
            
            # Get the base encoder config (will be our quantum CNN)
            tokenizer_config = cls._get_encoder_config(
                observation_space=observation_space,
                model_config_dict={**model_config_dict, "use_lstm": False},  # Disable LSTM for tokenizer
                action_space=action_space
            )
            
            return RecurrentEncoderConfig(
                input_dims=tokenizer_config.output_dims,
                recurrent_layer_type="lstm",
                hidden_dim=model_config_dict.get("lstm_cell_size", 128),
                hidden_weights_initializer=model_config_dict.get("lstm_kernel_initializer", "xavier_uniform"),
                hidden_weights_initializer_config=model_config_dict.get("lstm_kernel_initializer_kwargs", {}),
                hidden_bias_initializer=model_config_dict.get("lstm_bias_initializer", "zeros"),
                hidden_bias_initializer_config=model_config_dict.get("lstm_bias_initializer_kwargs", {}),
                batch_major=True,
                num_layers=1,
                tokenizer_config=tokenizer_config,  # Our quantum CNN as tokenizer
            )
        
        # Check if this is a 3D Box space (image)
        if isinstance(observation_space, Box) and len(observation_space.shape) == 3:
            # print(f"[CustomImageCatalog] Creating custom CNN encoder for shape {observation_space.shape}")

            # Create quantum-optimized CNN encoder config
            return CustomCNNConfig(
                input_dims=observation_space.shape,
                cnn_activation=model_config_dict.get("conv_activation", "relu"),
                cnn_kernel_initializer=model_config_dict.get("conv_kernel_initializer", "xavier_uniform"),
                cnn_kernel_initializer_config=model_config_dict.get("conv_kernel_initializer_kwargs", {}),
                cnn_bias_initializer=model_config_dict.get("conv_bias_initializer", "zeros"),
                cnn_bias_initializer_config=model_config_dict.get("conv_bias_initializer_kwargs", {}),
                # Custom quantum-specific parameters
                adaptive_pooling=True,
                final_feature_size=model_config_dict.get("quantum_feature_size", 256),
                quantum_filters=model_config_dict.get("quantum_conv_filters", None),
            )
        else:
            # For non-image spaces, use parent PPOCatalog implementation
            return super()._get_encoder_config(
                observation_space=observation_space,
                model_config_dict=model_config_dict,
                action_space=action_space
            )