import yaml
import numpy as np
import os
from typing import Dict, Any, Optional, Union
import warnings


def load_qdarts_config(config_path: str = 'qdarts_config_v5.yaml') -> Dict[str, Any]:
    """
    Load and validate QDarts v5 configuration from YAML file.
    
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
    
    # Validate configuration
    validate_config(config)
    
    # Convert types and ensure numpy arrays
    config = convert_config_types(config)
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the configuration structure and parameters.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required top-level sections
    required_sections = ['device', 'simulator']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required section: {section}")
    
    # Validate device topology
    topology = config['device']['topology']
    required_topology = ['num_dots', 'num_gates', 'sensor_dots']
    for key in required_topology:
        if key not in topology:
            raise ValueError(f"Missing required topology key: {key}")
    
    # Validate capacitance matrices
    capacitance = config['simulator']['capacitance']
    C_DD = capacitance['C_DD']
    C_DG = capacitance['C_DG']
    
    if len(C_DD) != topology['num_dots'] or any(len(row) != topology['num_dots'] for row in C_DD):
        raise ValueError(f"C_DD matrix must be {topology['num_dots']}x{topology['num_dots']}")
    
    if len(C_DG) != topology['num_dots'] or any(len(row) != topology['num_gates'] for row in C_DG):
        raise ValueError(f"C_DG matrix must be {topology['num_dots']}x{topology['num_gates']}")
    
    # Validate barrier configuration
    if 'barrier' not in config['simulator']:
        raise ValueError("Missing barrier configuration section")
    
    barrier_config = config['simulator']['barrier']
    if 'barrier_mappings' not in barrier_config:
        raise ValueError("Missing barrier_mappings in barrier configuration")
    
    # Validate barrier mappings
    for barrier_name, mapping in barrier_config['barrier_mappings'].items():
        required_fields = ['coupling_type', 'base_coupling', 'alpha', 'voltage_offset']
        for field in required_fields:
            if field not in mapping:
                raise ValueError(f"Missing {field} in {barrier_name}")
        
        coupling_type = mapping['coupling_type']
        if coupling_type == "dot_to_dot":
            if 'target_dots' not in mapping:
                raise ValueError(f"Missing target_dots in {barrier_name} for dot_to_dot coupling")
        elif coupling_type in ["reservoir_to_dot", "dot_to_reservoir"]:
            if 'target_dot' not in mapping:
                raise ValueError(f"Missing target_dot in {barrier_name} for {coupling_type} coupling")
        else:
            raise ValueError(f"Unknown coupling_type: {coupling_type}")
    
    # Validate default barrier voltages
    if 'default_barrier_voltages' not in barrier_config:
        raise ValueError("Missing default_barrier_voltages in barrier configuration")
    
    for barrier_name in barrier_config['barrier_mappings'].keys():
        if barrier_name not in barrier_config['default_barrier_voltages']:
            raise ValueError(f"Missing default voltage for {barrier_name}")


def convert_config_types(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert configuration values to appropriate types.
    
    Args:
        config: Raw configuration dictionary
        
    Returns:
        Configuration with proper types
    """
    # Convert capacitance matrices to numpy arrays
    config['simulator']['capacitance']['C_DD'] = np.array(
        config['simulator']['capacitance']['C_DD'], dtype=np.float64
    )
    config['simulator']['capacitance']['C_DG'] = np.array(
        config['simulator']['capacitance']['C_DG'], dtype=np.float64
    )
    
    # Convert bounds_limits
    bounds_limits = config['simulator']['capacitance']['bounds_limits']
    if isinstance(bounds_limits, (int, float)):
        # Single value - create array for all gates
        num_gates = config['device']['topology']['num_gates']
        config['simulator']['capacitance']['bounds_limits'] = np.full(num_gates, float(bounds_limits), dtype=np.float64)
    else:
        config['simulator']['capacitance']['bounds_limits'] = np.array(bounds_limits, dtype=np.float64)
    
    # Convert ks to scalar if it's an array
    ks = config['simulator']['capacitance']['ks']
    if ks is not None:
        if isinstance(ks, (list, np.ndarray)):
            config['simulator']['capacitance']['ks'] = float(ks[0])  # Take first element
        else:
            config['simulator']['capacitance']['ks'] = float(ks)
    
    # Convert tunnel couplings
    config['simulator']['tunneling']['tunnel_couplings'] = np.array(
        config['simulator']['tunneling']['tunnel_couplings'], dtype=np.float64
    )
    
    # Convert barrier configuration values
    barrier_config = config['simulator']['barrier']
    for mapping in barrier_config['barrier_mappings'].values():
        mapping['base_coupling'] = float(mapping['base_coupling'])
        mapping['alpha'] = float(mapping['alpha'])
        mapping['voltage_offset'] = float(mapping['voltage_offset'])
    
    for barrier_name, voltage in barrier_config['default_barrier_voltages'].items():
        barrier_config['default_barrier_voltages'][barrier_name] = float(voltage)
    
    # Convert sensor configuration
    sensor_config = config['simulator']['sensor']
    sensor_config['sensor_dot_indices'] = np.array(sensor_config['sensor_dot_indices'], dtype=int)
    sensor_config['sensor_detunings'] = np.array(sensor_config['sensor_detunings'], dtype=np.float64)
    sensor_config['noise_amplitude']['fast_noise'] = float(sensor_config['noise_amplitude']['fast_noise'])
    sensor_config['noise_amplitude']['slow_noise'] = float(sensor_config['noise_amplitude']['slow_noise'])
    sensor_config['peak_width_multiplier'] = float(sensor_config['peak_width_multiplier'])
    
    # Convert measurement arrays
    if 'measurement' in config:
        measurement = config['measurement']
        if 'voltage_range' in measurement:
            measurement['voltage_range']['min'] = np.array(measurement['voltage_range']['min'], dtype=np.float64)
            measurement['voltage_range']['max'] = np.array(measurement['voltage_range']['max'], dtype=np.float64)
        if 'sweep_matrix' in measurement:
            measurement['sweep_matrix'] = np.array(measurement['sweep_matrix'], dtype=np.float64)
    
    return config


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a summary of the loaded configuration.
    
    Args:
        config: Configuration dictionary
    """
    print("=" * 60)
    print("QDarts v5 Configuration Summary")
    print("=" * 60)
    
    # Device topology
    topology = config['device']['topology']
    print(f"Device: {topology['num_dots']} dots, {topology['num_gates']} gates")
    print(f"  Sensor dots: {topology['sensor_dots']}")
    print(f"  Target state: {config['device']['target_state']}")
    
    # Capacitance matrices
    capacitance = config['simulator']['capacitance']
    C_DD = capacitance['C_DD']
    C_DG = capacitance['C_DG']
    
    print(f"\nCapacitance Matrices:")
    print(f"  C_DD shape: {C_DD.shape}")
    print(f"  C_DG shape: {C_DG.shape}")
    print(f"  ks: {capacitance['ks']}")
    
    # Barrier configuration
    barrier_config = config['simulator']['barrier']
    print(f"\nBarrier Configuration:")
    for barrier_name, mapping in barrier_config['barrier_mappings'].items():
        coupling_type = mapping['coupling_type']
        alpha = mapping['alpha']
        if coupling_type == "dot_to_dot":
            target = mapping['target_dots']
        else:
            target = mapping['target_dot']
        print(f"  {barrier_name}: {coupling_type} -> {target} (α={alpha})")
    
    print(f"  Default voltages: {barrier_config['default_barrier_voltages']}")
    
    # Tunneling configuration
    tunneling = config['simulator']['tunneling']
    print(f"\nTunneling Configuration:")
    print(f"  Temperature: {tunneling['temperature']} K")
    print(f"  Energy range factor: {tunneling['energy_range_factor']}")
    
    # Sensor configuration
    sensor = config['simulator']['sensor']
    print(f"\nSensor Configuration:")
    print(f"  Sensor dots: {sensor['sensor_dot_indices']}")
    print(f"  Sensor detunings: {sensor['sensor_detunings']} meV")
    print(f"  Peak width multiplier: {sensor['peak_width_multiplier']}")
    
    # Measurement configuration
    if 'measurement' in config:
        measurement = config['measurement']
        print(f"\nMeasurement Configuration:")
        print(f"  Voltage range: {measurement['voltage_range']['min']} to {measurement['voltage_range']['max']}")
        print(f"  Resolution: {measurement['resolution']}")
    
    print("=" * 60)


def get_barrier_voltage_vector(config: Dict[str, Any], barrier_voltages: Dict[str, float]) -> np.ndarray:
    """
    Convert barrier voltage dictionary to voltage vector for all gates.
    
    Args:
        config: Configuration dictionary
        barrier_voltages: Dictionary of barrier voltages
        
    Returns:
        Voltage vector for all gates
    """
    num_gates = config['device']['topology']['num_gates']
    voltage_vector = np.zeros(num_gates)
    
    # Set barrier voltages
    for barrier_name, voltage in barrier_voltages.items():
        # Extract barrier index from name (e.g., "barrier_2" -> 2)
        barrier_idx = int(barrier_name.split('_')[1])
        if barrier_idx < num_gates:
            voltage_vector[barrier_idx] = voltage
    
    return voltage_vector 