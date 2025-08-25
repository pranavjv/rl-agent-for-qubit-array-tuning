"""
Configuration loading and validation utilities for RL training.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """Handles loading and validation of training configurations."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = None
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file (overrides instance path)
            
        Returns:
            Configuration dictionary
        """
        path = config_path or self.config_path
        if not path:
            raise ValueError("No configuration path provided")
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Validate configuration
        self._validate_config()
        
        # Apply environment variable substitutions
        self._apply_env_substitutions()
        
        return self.config
    
    def _validate_config(self):
        """Validate configuration structure and required fields."""
        if not self.config:
            raise ValueError("Configuration is empty")
        
        required_sections = [
            "experiment",
            "env", 
            "multi_agent",
            "ppo",
            "ray",
            "logging"
        ]
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate experiment section
        exp_config = self.config["experiment"]
        required_exp_fields = ["name", "project", "seed"]
        for field in required_exp_fields:
            if field not in exp_config:
                raise ValueError(f"Missing required experiment field: {field}")
        
        # Validate PPO section
        ppo_config = self.config["ppo"]
        required_ppo_fields = ["lr", "gamma", "clip_param"]
        for field in required_ppo_fields:
            if field not in ppo_config:
                raise ValueError(f"Missing required PPO field: {field}")
        
        # Validate Ray section
        ray_config = self.config["ray"]
        required_ray_fields = ["num_gpus", "num_workers"]
        for field in required_ray_fields:
            if field not in ray_config:
                raise ValueError(f"Missing required Ray field: {field}")
    
    def _apply_env_substitutions(self):
        """Apply environment variable substitutions in configuration."""
        def substitute_env_vars(obj):
            if isinstance(obj, dict):
                return {k: substitute_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_env_vars(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
                env_var = obj[2:-1]
                return os.environ.get(env_var, obj)
            else:
                return obj
        
        self.config = substitute_env_vars(self.config)
    
    def get_config(self) -> Dict[str, Any]:
        """Get loaded configuration."""
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self.config, updates)
    
    def save_config(self, output_path: str):
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save configuration
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """
    Convenience function to load configuration from file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader(config_path)
    return loader.load_config()
