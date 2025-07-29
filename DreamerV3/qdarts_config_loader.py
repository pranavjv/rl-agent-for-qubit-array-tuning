import yaml
import numpy as np
import os
from typing import Dict, Any, Optional, Union
import warnings


def load_qdarts_config(config_path: str = 'qdarts_config.yaml') -> Dict[str, Any]:
    """
    Load and validate QDarts configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing the validated configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    # Check if file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load YAML file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate and process configuration
    config = _validate_config(config)
    config = _convert_config_types(config)
    
    return config


def _validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the configuration structure and values.
    
    Args:
        config: Raw configuration dictionary
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ['device', 'simulator', 'measurement', 'output', 'logging']
    
    # Check required sections
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate device topology
    topology = config['device']['topology']
    num_dots = topology['num_dots']
    num_gates = topology['num_gates']
    
    # Validate capacitance matrices
    capacitance = config['simulator']['capacitance']
    C_DD = capacitance['C_DD']
    C_DG = capacitance['C_DG']
    
    if len(C_DD) != num_dots or any(len(row) != num_dots for row in C_DD):
        raise ValueError(f"C_DD matrix must be {num_dots}x{num_dots}")
    
    if len(C_DG) != num_dots or any(len(row) != num_gates for row in C_DG):
        raise ValueError(f"C_DG matrix must be {num_dots}x{num_gates}")
    
    # Validate target state
    target_state = config['device']['target_state']
    if len(target_state) != num_dots:
        raise ValueError(f"Target state must have {num_dots} elements")
    
    # Validate measurement configuration
    measurement = config['measurement']
    voltage_range = measurement['voltage_range']
    if len(voltage_range['min']) != 2 or len(voltage_range['max']) != 2:
        raise ValueError("Voltage range must be 2D")
    
    sweep_matrix = measurement['sweep_matrix']
    if len(sweep_matrix) != num_gates or any(len(row) != 2 for row in sweep_matrix):
        raise ValueError(f"Sweep matrix must be {num_gates}x2")
    
    # Validate tunneling configuration if enabled
    if config['simulator']['tunneling']['enabled']:
        tunneling = config['simulator']['tunneling']
        couplings = tunneling['couplings']
        if len(couplings) != num_dots or any(len(row) != num_dots for row in couplings):
            raise ValueError(f"Tunnel couplings matrix must be {num_dots}x{num_dots}")
    
    return config


def _convert_config_types(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert configuration values to appropriate types.
    
    Args:
        config: Validated configuration dictionary
        
    Returns:
        Configuration with converted types
    """
    # Convert lists to numpy arrays
    config['device']['target_state'] = np.array(config['device']['target_state'], dtype=np.int32)
    
    # Convert capacitance matrices to numpy arrays
    config['simulator']['capacitance']['C_DD'] = np.array(
        config['simulator']['capacitance']['C_DD'], dtype=np.float32
    )
    config['simulator']['capacitance']['C_DG'] = np.array(
        config['simulator']['capacitance']['C_DG'], dtype=np.float32
    )
    
    # Convert bounds_limits to numpy array
    bounds_limits = config['simulator']['capacitance']['bounds_limits']
    num_gates = config['device']['topology']['num_gates']
    if isinstance(bounds_limits, (int, float)):
        config['simulator']['capacitance']['bounds_limits'] = bounds_limits * np.ones(num_gates, dtype=np.float32)
    else:
        config['simulator']['capacitance']['bounds_limits'] = np.array(bounds_limits, dtype=np.float32)
    
    # Convert ks to numpy array
    ks = config['simulator']['capacitance']['ks']
    num_dots = config['device']['topology']['num_dots']
    if isinstance(ks, (int, float)):
        config['simulator']['capacitance']['ks'] = ks * np.ones(num_dots, dtype=np.float32)
    else:
        config['simulator']['capacitance']['ks'] = np.array(ks, dtype=np.float32)
    
    # Convert tunneling couplings to numpy array
    if config['simulator']['tunneling']['enabled']:
        config['simulator']['tunneling']['couplings'] = np.array(
            config['simulator']['tunneling']['couplings'], dtype=np.float32
        )
    
    # Convert measurement arrays
    config['measurement']['voltage_range']['min'] = np.array(
        config['measurement']['voltage_range']['min'], dtype=np.float32
    )
    config['measurement']['voltage_range']['max'] = np.array(
        config['measurement']['voltage_range']['max'], dtype=np.float32
    )
    config['measurement']['sweep_matrix'] = np.array(
        config['measurement']['sweep_matrix'], dtype=np.float32
    )
    
    return config


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration values.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'device': {
            'topology': {
                'num_dots': 3,
                'num_gates': 6,
                'inner_dots': [0, 1],
                'sensor_dots': [2],
                'dot_plungers': [0, 1],
                'barrier_plungers': [2, 3, 4],
                'sensor_plungers': [5]
            },
            'target_state': [1, 1, 5]
        },
        'simulator': {
            'capacitance': {
                'C_DD': [[1.0, 0.2, 0.08], [0.2, 1.0, 0.08], [0.08, 0.08, 1.0]],
                'C_DG': [[0.75, 0.1, 0, 0, 0, 0.02], [0.1, 0.75, 0, 0, 0, 0.02], [0, 0, 0, 0, 0, 1.0]],
                'bounds_limits': -2.0,
                'ks': 4
            },
            'tunneling': {
                'enabled': True,
                'temperature': 0.2,
                'couplings': [[0, 30e-6, 30e-6], [30e-6, 0, 30e-6], [30e-6, 30e-6, 0]]
            },
            'sensor': {
                'model_type': 'NoisySensorDot',
                'g_max': 1.0,
                'peak_width_multiplier': 200,
                'noise': {
                    'fast_noise_amplitude': 0.001,
                    'slow_noise_amplitude': 0.005,
                    'correlation_time': 250,
                    'virtual_samples': 200
                }
            }
        },
        'measurement': {
            'voltage_range': {
                'min': [-0.2, -0.2],
                'max': [0.4, 0.4]
            },
            'resolution': 250,
            'sweep_matrix': [[1, 0], [0, 1], [0, 0], [0, 0], [0, 0], [0, 0]]
        },
        'output': {
            'plots': {
                'save_format': 'png',
                'dpi': 300,
                'bbox_inches': 'tight'
            },
            'filenames': {
                'csd_plot': 'qdarts_csd_plot.png',
                'sensor_plot': 'qdarts_sensor_plot.png',
                'noised_plot': 'qdarts_noised_plot.png'
            }
        },
        'logging': {
            'print_logs': True,
            'timing': True,
            'debug': False
        }
    }


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
    """
    # Convert numpy arrays back to lists for YAML serialization
    config_copy = _convert_numpy_to_lists(config)
    
    with open(config_path, 'w') as f:
        yaml.dump(config_copy, f, default_flow_style=False, indent=2)


def _convert_numpy_to_lists(obj: Any) -> Any:
    """
    Recursively convert numpy arrays to lists for YAML serialization.
    
    Args:
        obj: Object to convert
        
    Returns:
        Object with numpy arrays converted to lists
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_numpy_to_lists(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_to_lists(item) for item in obj]
    else:
        return obj


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a summary of the configuration.
    
    Args:
        config: Configuration dictionary
    """
    print("QDarts Configuration Summary")
    print("=" * 40)
    
    # Device topology
    topology = config['device']['topology']
    print(f"Device: {topology['num_dots']} dots, {topology['num_gates']} gates")
    print(f"Target state: {config['device']['target_state']}")
    
    # Capacitance model
    capacitance = config['simulator']['capacitance']
    print(f"Capacitance model: {capacitance['C_DD'].shape[0]}x{capacitance['C_DD'].shape[1]} C_DD, {capacitance['C_DG'].shape[0]}x{capacitance['C_DG'].shape[1]} C_DG")
    
    # Tunneling
    tunneling = config['simulator']['tunneling']
    if tunneling['enabled']:
        print(f"Tunneling: enabled, T={tunneling['temperature']} meV")
    else:
        print("Tunneling: disabled")
    
    # Sensor
    sensor = config['simulator']['sensor']
    print(f"Sensor: {sensor['model_type']}, g_max={sensor['g_max']}, width_mult={sensor['peak_width_multiplier']}")
    
    # Measurement
    measurement = config['measurement']
    print(f"Measurement: {measurement['resolution']}x{measurement['resolution']} resolution")
    print(f"Voltage range: {measurement['voltage_range']['min']} to {measurement['voltage_range']['max']}")
    
    print("=" * 40) 