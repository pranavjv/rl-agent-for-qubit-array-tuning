import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.configs import ModelConfig, MLPEncoderConfig, CNNEncoderConfig, RecurrentEncoderConfig
from ray.rllib.core.models.base import Encoder, ENCODER_OUT
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.models.torch.encoder import TorchLSTMEncoder
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

"""
todo:

handle custom dict observation extraction for both the actor/critic (working)
and for the LSTM tokenizer
we need to override _get_encoder_config from catalog.py
"""

@dataclass
class DictEncoderConfig(ModelConfig):
    """Configuration for Dict observation space encoder with CNN+MLP components."""
    
    # Required fields following Ray's pattern
    image_input_dims: Optional[Tuple[int, ...]] = None
    voltage_input_dims: Optional[Tuple[int, ...]] = None
    
    # CNN configuration for image processing
    cnn_filter_specifiers: Optional[list] = None
    cnn_activation: str = "relu"
    cnn_kernel_initializer: str = "xavier_uniform"
    cnn_kernel_initializer_config: dict = field(default_factory=dict)
    cnn_bias_initializer: str = "zeros"  
    cnn_bias_initializer_config: dict = field(default_factory=dict)
    
    # MLP configuration for voltage processing
    voltage_hidden_dims: list = field(default_factory=lambda: [32, 32])
    voltage_activation: str = "relu"
    voltage_weights_initializer: str = "xavier_uniform"
    voltage_weights_initializer_config: dict = field(default_factory=dict)
    voltage_bias_initializer: str = "zeros"
    voltage_bias_initializer_config: dict = field(default_factory=dict)
    
    # Final MLP configuration for combined features
    fcnet_hiddens: list = field(default_factory=lambda: [256])
    fcnet_activation: str = "relu"
    fcnet_weights_initializer: str = "xavier_uniform"
    fcnet_weights_initializer_config: dict = field(default_factory=dict)
    fcnet_bias_initializer: str = "zeros"
    fcnet_bias_initializer_config: dict = field(default_factory=dict)
    
    @property
    def output_dims(self) -> Tuple[int, ...]:
        """Override to compute output dimensions dynamically following Ray's pattern."""
        if not self.fcnet_hiddens:
            raise ValueError("fcnet_hiddens cannot be empty for DictEncoderConfig")
        # Return final layer dimension as tuple (matching Ray's pattern)
        return (self.fcnet_hiddens[-1],)
    
    def _validate(self, framework: str = "torch") -> None:
        """Validate configuration parameters following Ray's pattern."""
        if framework != "torch":
            raise ValueError(f"Only torch framework supported, got {framework}")
        
        if self.image_input_dims is None or self.voltage_input_dims is None:
            raise ValueError("Both image_input_dims and voltage_input_dims must be specified")
        
        if len(self.image_input_dims) != 3:
            raise ValueError("image_input_dims must be 3D (H, W, C)")
            
        if len(self.voltage_input_dims) != 1:
            raise ValueError("voltage_input_dims must be 1D")
    
    def build(self, framework: str = "torch") -> "DictEncoder":
        """Build the actual Dict encoder following Ray's pattern."""
        if framework != "torch":
            raise ValueError(f"Framework {framework} not supported")
        
        self._validate(framework)
        return DictEncoder(self)


class DictEncoder(TorchModel, Encoder):
    """Custom encoder for Dict observation spaces with CNN for images and MLP for voltage."""
    
    def __init__(self, config: DictEncoderConfig):
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)
        
        # Extract dimensions from config
        h, w, c = config.image_input_dims
        voltage_dim = config.voltage_input_dims[0]
        
        # Build CNN for image processing - simple architecture for quantum device images
        # Use simple CNN layers instead of Ray's get_filter_config to avoid shape restrictions
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(c, 16, kernel_size=8, stride=4),
            nn.ReLU() if config.cnn_activation == "relu" else nn.Tanh(),
            # Second conv block  
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU() if config.cnn_activation == "relu" else nn.Tanh(),
            # Third conv block
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU() if config.cnn_activation == "relu" else nn.Tanh(),
            nn.Flatten()
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            cnn_output = self.cnn(dummy_input)
            self.cnn_output_size = cnn_output.shape[1]
        
        # Build MLP for voltage processing using config parameters
        voltage_layers = []
        prev_dim = voltage_dim
        
        for hidden_dim in config.voltage_hidden_dims:
            voltage_layers.append(nn.Linear(prev_dim, hidden_dim))
            # Apply specified activation
            if config.voltage_activation == "relu":
                voltage_layers.append(nn.ReLU())
            elif config.voltage_activation == "tanh":
                voltage_layers.append(nn.Tanh())
            else:
                voltage_layers.append(nn.ReLU())  # fallback
            prev_dim = hidden_dim
        
        self.voltage_mlp = nn.Sequential(*voltage_layers)
        self.voltage_output_size = prev_dim
        
        # Combined processing using config parameters
        combined_size = self.cnn_output_size + self.voltage_output_size
        
        # Build final MLP layers
        combined_layers = []
        prev_size = combined_size
        
        for hidden_dim in config.fcnet_hiddens:
            combined_layers.append(nn.Linear(prev_size, hidden_dim))
            # Apply specified activation
            if config.fcnet_activation == "relu":
                combined_layers.append(nn.ReLU())
            elif config.fcnet_activation == "tanh":
                combined_layers.append(nn.Tanh())
            else:
                combined_layers.append(nn.ReLU())  # fallback
            prev_size = hidden_dim
        
        self.combined_mlp = nn.Sequential(*combined_layers)
        self._output_dims = prev_size
    
    @property
    def output_dims(self) -> Tuple[int, ...]:
        """Return output dimensions as tuple following Ray's pattern."""
        return (self._output_dims,)
    
    def _forward(self, inputs: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through the Dict encoder."""
        # Handle tokenizer context: inputs contain {Columns.OBS: dict_obs}
        # vs direct context: inputs contain {"image": tensor, "voltage": tensor}
        if Columns.OBS in inputs:
            # Tokenizer context - extract Dict observations
            obs_dict = inputs[Columns.OBS]
            if isinstance(obs_dict, dict):
                image = obs_dict["image"]
                voltage = obs_dict["voltage"]
            else:
                raise ValueError(f"Expected Dict observation in tokenizer context, got {type(obs_dict)}")
        else:
            # Direct Dict observation access (non-tokenizer context)
            image = inputs["image"]  
            voltage = inputs["voltage"]
        
        # Handle batch dimension
        if image.dim() == 3:  # Add batch dimension if missing
            image = image.unsqueeze(0)
            voltage = voltage.unsqueeze(0)
        
        # Convert image from (B, H, W, C) to (B, C, H, W) if needed
        if image.shape[-1] in [1, 2]:  # channels last -> channels first
            image = image.permute(0, 3, 1, 2)
        
        # Process components
        image_features = self.cnn(image)
        voltage_features = self.voltage_mlp(voltage)
        
        # Combine and process
        combined = torch.cat([image_features, voltage_features], dim=-1)
        output = self.combined_mlp(combined)
        
        # Return in the format expected by Ray (both tokenizer and direct contexts)
        return {ENCODER_OUT: output}


@dataclass
class DictRecurrentEncoderConfig(RecurrentEncoderConfig):
    """Custom RecurrentEncoderConfig that handles Dict observation spaces."""
    
    # Dict-specific fields
    image_input_dims: Optional[Tuple[int, ...]] = None
    voltage_input_dims: Optional[Tuple[int, ...]] = None
    
    # CNN configuration for image processing
    cnn_filter_specifiers: Optional[list] = None
    cnn_activation: str = "relu"
    cnn_kernel_initializer: str = "xavier_uniform"
    cnn_kernel_initializer_config: dict = field(default_factory=dict)
    cnn_bias_initializer: str = "zeros"
    cnn_bias_initializer_config: dict = field(default_factory=dict)
    
    # MLP configuration for voltage processing
    voltage_hidden_dims: list = field(default_factory=lambda: [32, 32])
    voltage_activation: str = "relu"
    voltage_weights_initializer: str = "xavier_uniform"
    voltage_weights_initializer_config: dict = field(default_factory=dict)
    voltage_bias_initializer: str = "zeros"
    voltage_bias_initializer_config: dict = field(default_factory=dict)
    
    # Combined MLP configuration
    combined_hidden_dims: list = field(default_factory=lambda: [256])
    combined_activation: str = "relu"
    combined_weights_initializer: str = "xavier_uniform"
    combined_weights_initializer_config: dict = field(default_factory=dict)
    combined_bias_initializer: str = "zeros"
    combined_bias_initializer_config: dict = field(default_factory=dict)
    
    def build(self, framework: str = "torch") -> "DictLSTMEncoder":
        """Build custom LSTM encoder that handles Dict observations."""
        if framework != "torch":
            raise ValueError(f"Framework {framework} not supported")
        
        return DictLSTMEncoder(self)


class DictLSTMEncoder(TorchModel, Encoder):
    """Custom LSTM encoder that processes Dict observations internally."""
    
    def __init__(self, config: DictRecurrentEncoderConfig):
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)
        
        # Build Dict observation processor (similar to DictEncoder)
        h, w, c = config.image_input_dims
        voltage_dim = config.voltage_input_dims[0]
        
        # CNN for image processing - simple architecture for quantum device images
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(c, 16, kernel_size=8, stride=4),
            nn.ReLU() if config.cnn_activation == "relu" else nn.Tanh(),
            # Second conv block
            nn.Conv2d(16, 32, kernel_size=4, stride=2), 
            nn.ReLU() if config.cnn_activation == "relu" else nn.Tanh(),
            # Third conv block
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU() if config.cnn_activation == "relu" else nn.Tanh(),
            nn.Flatten()
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, c, h, w)
            cnn_output = self.cnn(dummy_input)
            cnn_output_size = cnn_output.shape[1]
        
        # MLP for voltage processing
        voltage_layers = []
        prev_dim = voltage_dim
        
        for hidden_dim in config.voltage_hidden_dims:
            voltage_layers.append(nn.Linear(prev_dim, hidden_dim))
            if config.voltage_activation == "relu":
                voltage_layers.append(nn.ReLU())
            elif config.voltage_activation == "tanh":
                voltage_layers.append(nn.Tanh())
            else:
                voltage_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.voltage_mlp = nn.Sequential(*voltage_layers)
        voltage_output_size = prev_dim
        
        # Combined processing
        combined_size = cnn_output_size + voltage_output_size
        combined_layers = []
        prev_size = combined_size
        
        for hidden_dim in config.combined_hidden_dims:
            combined_layers.append(nn.Linear(prev_size, hidden_dim))
            if config.combined_activation == "relu":
                combined_layers.append(nn.ReLU())
            elif config.combined_activation == "tanh":
                combined_layers.append(nn.Tanh())
            else:
                combined_layers.append(nn.ReLU())
            prev_size = hidden_dim
        
        self.dict_processor = nn.Sequential(*combined_layers)
        
        # Build LSTM layer with simple initialization
        self.lstm = nn.LSTM(
            input_size=prev_size,  # Input from dict processor
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=config.batch_major,
            bias=config.use_bias,
        )
        
        # Use PyTorch's default initialization (xavier/glorot uniform for weights, zeros for biases)
        # This is simpler and avoids Ray dependencies
    
    @override(TorchModel)
    def get_initial_state(self):
        """Return initial LSTM state."""
        return {
            "h": torch.zeros(self.config.num_layers, self.config.hidden_dim),
            "c": torch.zeros(self.config.num_layers, self.config.hidden_dim),
        }
    
    def _forward(self, inputs: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass: Dict obs -> encoded tensor -> LSTM -> output."""
        # Extract Dict observations
        obs_dict = inputs[Columns.OBS]
        image = obs_dict["image"]
        voltage = obs_dict["voltage"]
        
        # Handle batch and time dimensions
        batch_size = image.shape[0]
        if image.dim() == 4:  # (B, H, W, C)
            time_dim = 1
        elif image.dim() == 5:  # (B, T, H, W, C)
            time_dim = image.shape[1]
            # Flatten batch and time for processing
            image = image.view(batch_size * time_dim, *image.shape[2:])
            voltage = voltage.view(batch_size * time_dim, *voltage.shape[2:])
        else:
            raise ValueError(f"Unexpected image dimensions: {image.shape}")
        
        # Convert image to channels-first format
        if image.shape[-1] in [1, 2]:
            image = image.permute(0, 3, 1, 2)
        
        # Process Dict observations
        image_features = self.cnn(image)
        voltage_features = self.voltage_mlp(voltage)
        
        # Combine and process
        combined = torch.cat([image_features, voltage_features], dim=-1)
        encoded_features = self.dict_processor(combined)
        
        # Reshape for LSTM: (batch, time, features)
        if time_dim > 1:
            encoded_features = encoded_features.view(batch_size, time_dim, -1)
        else:
            encoded_features = encoded_features.unsqueeze(1)  # Add time dimension
        
        # Handle LSTM state
        if Columns.STATE_IN in inputs and inputs[Columns.STATE_IN] is not None:
            states_in = inputs[Columns.STATE_IN]
            # Convert to layers-first format
            import tree
            states_in = tree.map_structure(
                lambda s: s.transpose(0, 1), states_in
            )
            lstm_state = (states_in["h"], states_in["c"])
        else:
            # Initialize LSTM state
            device = encoded_features.device
            h_state = torch.zeros(self.config.num_layers, batch_size, self.config.hidden_dim, device=device)
            c_state = torch.zeros(self.config.num_layers, batch_size, self.config.hidden_dim, device=device)
            lstm_state = (h_state, c_state)
        
        # LSTM forward pass
        lstm_output, (new_h, new_c) = self.lstm(encoded_features, lstm_state)
        
        # Remove time dimension if it was added
        if time_dim == 1:
            lstm_output = lstm_output.squeeze(1)
        
        # Prepare output
        output = {ENCODER_OUT: lstm_output}
        
        # Add state output
        states_out = {"h": new_h, "c": new_c}
        # Convert back to batch-first format
        import tree
        output[Columns.STATE_OUT] = tree.map_structure(
            lambda s: s.transpose(0, 1), states_out
        )
        
        return output


class CustomCatalog(PPOCatalog):
    """Minimal PPO catalog subclass that handles Dict observation spaces."""
    
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: dict,
    ) -> None:
        super().__init__(observation_space=observation_space, action_space=action_space, model_config_dict=model_config_dict)
    
    @classmethod
    @override(PPOCatalog)
    def _get_encoder_config(
        cls,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
    ) -> ModelConfig:
        """Override to handle Dict observation spaces in both LSTM and non-LSTM paths."""
        use_lstm = model_config_dict["use_lstm"]
        
        # Handle Dict spaces for both LSTM and non-LSTM paths
        if isinstance(observation_space, spaces.Dict):
            print(f"[DEBUG]: custom dict encoder config being created (use_lstm={use_lstm})...")
            
            # Validate Dict observation space
            if "image" not in observation_space.spaces or "voltage" not in observation_space.spaces:
                raise ValueError("Dict observation space must contain 'image' and 'voltage' keys")
            
            image_space = observation_space.spaces["image"] 
            voltage_space = observation_space.spaces["voltage"]
            
            # Extract parameters in Ray's style
            activation = model_config_dict.get("fcnet_activation", "relu")
            hidden_dims = model_config_dict.get("fcnet_hiddens", [256])
            
            # Create DictEncoderConfig for tokenizer (both LSTM and non-LSTM use this)
            dict_config = DictEncoderConfig(
                # Image processing parameters
                image_input_dims=image_space.shape,
                cnn_filter_specifiers=model_config_dict.get("conv_filters"),
                cnn_activation=model_config_dict.get("conv_activation", "relu"),
                cnn_kernel_initializer=model_config_dict.get("conv_kernel_initializer", "xavier_uniform"),
                cnn_kernel_initializer_config=model_config_dict.get("conv_kernel_initializer_kwargs", {}),
                cnn_bias_initializer=model_config_dict.get("conv_bias_initializer", "zeros"),
                cnn_bias_initializer_config=model_config_dict.get("conv_bias_initializer_kwargs", {}),
                
                # Voltage processing parameters  
                voltage_input_dims=voltage_space.shape,
                voltage_hidden_dims=[32, 32],  # Fixed for simplicity
                voltage_activation=activation,
                voltage_weights_initializer=model_config_dict.get("fcnet_kernel_initializer", "xavier_uniform"),
                voltage_weights_initializer_config=model_config_dict.get("fcnet_kernel_initializer_kwargs", {}),
                voltage_bias_initializer=model_config_dict.get("fcnet_bias_initializer", "zeros"),
                voltage_bias_initializer_config=model_config_dict.get("fcnet_bias_initializer_kwargs", {}),
                
                # Final MLP parameters
                fcnet_hiddens=hidden_dims,
                fcnet_activation=activation,
                fcnet_weights_initializer=model_config_dict.get("fcnet_kernel_initializer", "xavier_uniform"),
                fcnet_weights_initializer_config=model_config_dict.get("fcnet_kernel_initializer_kwargs", {}),
                fcnet_bias_initializer=model_config_dict.get("fcnet_bias_initializer", "zeros"),
                fcnet_bias_initializer_config=model_config_dict.get("fcnet_bias_initializer_kwargs", {}),
            )
            
            if use_lstm:
                # Use Ray's RecurrentEncoderConfig with our DictEncoder as tokenizer
                # Ray will create its own LSTM, we just provide the tokenizer
                return RecurrentEncoderConfig(
                    input_dims=dict_config.output_dims,  # LSTM input size from tokenizer output
                    recurrent_layer_type="lstm",
                    hidden_dim=model_config_dict["lstm_cell_size"],
                    hidden_weights_initializer=model_config_dict["lstm_kernel_initializer"],
                    hidden_weights_initializer_config=model_config_dict.get("lstm_kernel_initializer_kwargs", {}),
                    hidden_bias_initializer=model_config_dict["lstm_bias_initializer"],
                    hidden_bias_initializer_config=model_config_dict.get("lstm_bias_initializer_kwargs", {}),
                    batch_major=True,
                    num_layers=1,
                    tokenizer_config=dict_config,  # Our DictEncoder as tokenizer
                )
            else:
                # Non-LSTM path: return the DictEncoderConfig directly
                return dict_config
        else:
            # For non-Dict spaces, use parent implementation
            return super()._get_encoder_config(
                observation_space=observation_space,
                model_config_dict=model_config_dict,
                action_space=action_space
            )