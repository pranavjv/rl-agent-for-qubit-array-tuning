"""
Custom neural networks for RL training
"""

import gymnasium as gym
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.sac.sac_catalog import SACCatalog
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.utils.annotations import override

from .custom_neural_nets import (
    SimpleCNNConfig,
    IMPALAConfig,
    PolicyHeadConfig,
    ValueHeadConfig,
)


class CustomPPOCatalog(PPOCatalog):
    """Custom catalog for quantum neural network components."""
    
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
    
    @override(PPOCatalog)
    def _get_encoder_config(
        self,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
    ) -> ModelConfig:
        from gymnasium.spaces import Box
        
        backbone_config = model_config_dict.get("backbone", {})
        lstm_config = backbone_config.get("lstm", {})
        use_lstm = lstm_config.get("enabled", False)
        
        if use_lstm:
            from ray.rllib.core.models.configs import RecurrentEncoderConfig
            
            # Create CNN tokenizer config (without LSTM)
            backbone_type = backbone_config.get("type", "SimpleCNN")
            
            if backbone_type == "IMPALA":
                tokenizer_config = IMPALAConfig(
                    input_dims=observation_space.shape,
                    cnn_activation=model_config_dict.get("conv_activation", "relu"),
                    conv_layers=backbone_config.get("conv_layers"),
                    feature_size=backbone_config.get("feature_size", 256),
                    adaptive_pooling=backbone_config.get("adaptive_pooling", True),
                    num_res_blocks=backbone_config.get("num_res_blocks", 2),
                )
            elif backbone_type == "SimpleCNN":
                tokenizer_config = SimpleCNNConfig(
                    input_dims=observation_space.shape,
                    cnn_activation=model_config_dict.get("conv_activation", "relu"),
                    conv_layers=backbone_config.get("conv_layers"),
                    feature_size=backbone_config.get("feature_size", 256),
                    adaptive_pooling=backbone_config.get("adaptive_pooling", True),
                )
            
            else:
                raise ValueError(f"Unsupported backbone type: {backbone_type}")
            
            # Wrap CNN with LSTM
            return RecurrentEncoderConfig(
                input_dims=tokenizer_config.output_dims,
                recurrent_layer_type="lstm",
                hidden_dim=lstm_config.get("cell_size", 128),
                num_layers=lstm_config.get("num_layers", 1),
                max_seq_len=lstm_config.get("max_seq_len", 50),
                batch_major=True,
                tokenizer_config=tokenizer_config,
                use_bias=True,
                use_prev_action=lstm_config.get("use_prev_action", False),
                use_prev_reward=lstm_config.get("use_prev_reward", False),
            )
        
        if isinstance(observation_space, Box) and len(observation_space.shape) == 3:
            # Check if IMPALA is specified in the config
            backbone_type = backbone_config.get("type", "SimpleCNN")
            
            if backbone_type == "IMPALA":
                return IMPALAConfig(
                    input_dims=observation_space.shape,
                    cnn_activation=model_config_dict.get("conv_activation", "relu"),
                    conv_layers=backbone_config.get("conv_layers"),
                    feature_size=backbone_config.get("feature_size", 256),
                    adaptive_pooling=backbone_config.get("adaptive_pooling", True),
                    num_res_blocks=backbone_config.get("num_res_blocks", 2),
                )
            elif backbone_type == "SimpleCNN":
                return SimpleCNNConfig(
                    input_dims=observation_space.shape,
                    cnn_activation=model_config_dict.get("conv_activation", "relu"),
                    conv_layers=backbone_config.get("conv_layers"),
                    feature_size=backbone_config.get("feature_size", 256),
                    adaptive_pooling=backbone_config.get("adaptive_pooling", True),
                )
            else:
                raise ValueError(f"Unsupported backbone type: {backbone_type}. Supported types: 'SimpleCNN', 'IMPALA'")
        else:
            return super()._get_encoder_config(
                observation_space=observation_space,
                model_config_dict=model_config_dict,
                action_space=action_space,
            )
    
    @override(PPOCatalog)
    def build_pi_head(self, framework: str = "torch"):
        
        policy_config = self._model_config_dict.get("policy_head", {})
        backbone_config = self._model_config_dict.get("backbone", {})
        lstm_config = backbone_config.get("lstm", {})
        
        # Determine input dimensions based on whether LSTM is enabled
        if lstm_config.get("enabled", False):
            input_dim = lstm_config.get("cell_size", 128)
        else:
            input_dim = backbone_config.get("feature_size", 256)
        
        config = PolicyHeadConfig(
            input_dims=(input_dim,),
            hidden_layers=policy_config.get("hidden_layers", [128, 128]),
            activation=policy_config.get("activation", "relu"),
            use_attention=policy_config.get("use_attention", False),
            output_layer_dim=self.action_space.shape[0] * 2, # mean and log std for each action dimension
        )
        
        return config.build(framework=framework)
    
    @override(PPOCatalog)
    def build_vf_head(self, framework: str = "torch"):
        
        value_config = self._model_config_dict.get("value_head", {})
        backbone_config = self._model_config_dict.get("backbone", {})
        lstm_config = backbone_config.get("lstm", {})
        
        # Determine input dimensions based on whether LSTM is enabled
        if lstm_config.get("enabled", False):
            input_dim = lstm_config.get("cell_size", 128)
        else:
            input_dim = backbone_config.get("feature_size", 256)
        
        config = ValueHeadConfig(
            input_dims=(input_dim,),
            hidden_layers=value_config.get("hidden_layers", [128, 128]),
            activation=value_config.get("activation", "relu"),
            use_attention=value_config.get("use_attention", False),
        )
        
        return config.build(framework=framework)


class CustomSACCatalog(SACCatalog):
    """Custom catalog for SAC quantum neural network components."""
    
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
    
    @override(SACCatalog)
    def _get_encoder_config(
        self,
        observation_space: gym.Space,
        model_config_dict: dict,
        action_space: gym.Space = None,
    ) -> ModelConfig:
        from gymnasium.spaces import Box
        
        backbone_config = model_config_dict.get("backbone", {})
        lstm_config = backbone_config.get("lstm", {})
        use_lstm = lstm_config.get("enabled", False)
        
        if use_lstm:
            from ray.rllib.core.models.configs import RecurrentEncoderConfig
            
            # Create CNN tokenizer config (without LSTM)
            backbone_type = backbone_config.get("type", "SimpleCNN")
            
            if backbone_type == "IMPALA":
                tokenizer_config = IMPALAConfig(
                    input_dims=observation_space.shape,
                    cnn_activation=model_config_dict.get("conv_activation", "relu"),
                    conv_layers=backbone_config.get("conv_layers"),
                    feature_size=backbone_config.get("feature_size", 256),
                    adaptive_pooling=backbone_config.get("adaptive_pooling", True),
                    num_res_blocks=backbone_config.get("num_res_blocks", 2),
                )
            elif backbone_type == "SimpleCNN":
                tokenizer_config = SimpleCNNConfig(
                    input_dims=observation_space.shape,
                    cnn_activation=model_config_dict.get("conv_activation", "relu"),
                    conv_layers=backbone_config.get("conv_layers"),
                    feature_size=backbone_config.get("feature_size", 256),
                    adaptive_pooling=backbone_config.get("adaptive_pooling", True),
                )
            else:
                raise ValueError(f"Unsupported backbone type: {backbone_type}. Supported types: 'SimpleCNN', 'IMPALA'")
            
            # Wrap CNN with LSTM
            return RecurrentEncoderConfig(
                input_dims=tokenizer_config.output_dims,
                recurrent_layer_type="lstm",
                hidden_dim=lstm_config.get("cell_size", 128),
                num_layers=lstm_config.get("num_layers", 1),
                max_seq_len=lstm_config.get("max_seq_len", 50),
                batch_major=True,
                tokenizer_config=tokenizer_config,
                use_bias=True,
                use_prev_action=lstm_config.get("use_prev_action", False),
                use_prev_reward=lstm_config.get("use_prev_reward", False),
            )
        
        if isinstance(observation_space, Box) and len(observation_space.shape) == 3:
            # Check if IMPALA is specified in the config
            backbone_type = backbone_config.get("type", "SimpleCNN")
            
            if backbone_type == "IMPALA":
                return IMPALAConfig(
                    input_dims=observation_space.shape,
                    cnn_activation=model_config_dict.get("conv_activation", "relu"),
                    conv_layers=backbone_config.get("conv_layers"),
                    feature_size=backbone_config.get("feature_size", 256),
                    adaptive_pooling=backbone_config.get("adaptive_pooling", True),
                    num_res_blocks=backbone_config.get("num_res_blocks", 2),
                )
            elif backbone_type == "SimpleCNN":
                return SimpleCNNConfig(
                    input_dims=observation_space.shape,
                    cnn_activation=model_config_dict.get("conv_activation", "relu"),
                    conv_layers=backbone_config.get("conv_layers"),
                    feature_size=backbone_config.get("feature_size", 256),
                    adaptive_pooling=backbone_config.get("adaptive_pooling", True),
                )
            else:
                raise ValueError(f"Unsupported backbone type: {backbone_type}. Supported types: 'SimpleCNN', 'IMPALA'")
        else:
            return super()._get_encoder_config(
                observation_space=observation_space,
                model_config_dict=model_config_dict,
                action_space=action_space,
            )
    
    @override(SACCatalog)
    def build_pi_head(self, framework: str = "torch"):
        
        policy_config = self._model_config_dict.get("policy_head", {})
        backbone_config = self._model_config_dict.get("backbone", {})
        lstm_config = backbone_config.get("lstm", {})
        
        # Determine input dimensions based on whether LSTM is enabled
        if lstm_config.get("enabled", False):
            input_dim = lstm_config.get("cell_size", 128)
        else:
            input_dim = backbone_config.get("feature_size", 256)
        
        config = PolicyHeadConfig(
            input_dims=(input_dim,),
            hidden_layers=policy_config.get("hidden_layers", [128, 128]),
            activation=policy_config.get("activation", "relu"),
            use_attention=policy_config.get("use_attention", False),
            output_layer_dim=self.action_space.shape[0] * 2, # mean and log std for each action dimension
        )
        
        return config.build(framework=framework)
    
    @override(SACCatalog)
    def build_qf_head(self, framework: str = "torch"):
        
        qf_config = self._model_config_dict.get("value_head", {})
        backbone_config = self._model_config_dict.get("backbone", {})
        lstm_config = backbone_config.get("lstm", {})
        
        # Determine input dimensions based on whether LSTM is enabled
        if lstm_config.get("enabled", False):
            input_dim = lstm_config.get("cell_size", 128)
        else:
            input_dim = backbone_config.get("feature_size", 256)
        
        # Q-function takes both state and action as input
        input_dim += self.action_space.shape[0]
        
        config = ValueHeadConfig(
            input_dims=(input_dim,),
            hidden_layers=qf_config.get("hidden_layers", [128, 128]),
            activation=qf_config.get("activation", "relu"),
            use_attention=qf_config.get("use_attention", False),
        )
        
        return config.build(framework=framework)
    
    @override(SACCatalog)
    def build_qf_encoder(self, framework: str = "torch"):
        """Build Q-function encoder (same as main encoder for SAC)."""
        return self.build_encoder(framework=framework)