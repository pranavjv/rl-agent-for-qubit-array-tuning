import yaml
import numpy as np
import os
from typing import Dict, Any, Optional, Union
import warnings


def load_qdarts_config(config_path: str = 'qdarts_config_v4.yaml') -> Dict[str, Any]:
    """
    Load and validate QDarts v4 configuration from YAML file.
    
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
    
    # Preprocess configuration for experiment.py compatibility
    config = preprocess_config(config)
    
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
    required_topology = ['num_dots', 'num_gates', 'inner_dots', 'sensor_dots', 
                        'plunger_gates', 'barrier_gates', 'sensor_gates']
    for key in required_topology:
        if key not in topology:
            raise ValueError(f"Missing required topology key: {key}")
    
    # Validate gate indices consistency
    plunger_gates = set(topology['plunger_gates'])
    barrier_gates = set(topology['barrier_gates'])
    sensor_gates = set(topology['sensor_gates'])
    
    # Check for overlapping gate assignments
    if plunger_gates & barrier_gates:
        raise ValueError("Plunger gates and barrier gates cannot overlap")
    if plunger_gates & sensor_gates:
        raise ValueError("Plunger gates and sensor gates cannot overlap")
    if barrier_gates & sensor_gates:
        raise ValueError("Barrier gates and sensor gates cannot overlap")
    
    # Check total number of gates
    total_gates = len(plunger_gates | barrier_gates | sensor_gates)
    if total_gates != topology['num_gates']:
        raise ValueError(f"Total number of gates ({total_gates}) doesn't match num_gates ({topology['num_gates']})")
    
    # Validate capacitance matrices
    capacitance = config['simulator']['capacitance']
    C_DD = np.array(capacitance['C_DD'])
    C_DG = np.array(capacitance['C_DG'])
    
    if C_DD.shape != (topology['num_dots'], topology['num_dots']):
        raise ValueError(f"C_DD matrix shape {C_DD.shape} doesn't match num_dots {topology['num_dots']}")
    
    if C_DG.shape != (topology['num_dots'], topology['num_gates']):
        raise ValueError(f"C_DG matrix shape {C_DG.shape} doesn't match (num_dots, num_gates)")
    
    # Validate barrier configuration
    if 'barrier' not in config['simulator']:
        raise ValueError("Missing barrier configuration section")
    
    barrier_config = config['simulator']['barrier']
    if 'barrier_gates' not in barrier_config:
        raise ValueError("Missing barrier_gates in barrier configuration")
    
    if set(barrier_config['barrier_gates']) != barrier_gates:
        raise ValueError("Barrier gates in topology and barrier config don't match")
    
    # Validate barrier mappings
    if 'barrier_mappings' not in barrier_config:
        raise ValueError("Missing barrier_mappings in barrier configuration")
    
    for barrier_name, mapping in barrier_config['barrier_mappings'].items():
        if 'coupling_type' not in mapping:
            raise ValueError(f"Missing coupling_type in {barrier_name}")
        
        coupling_type = mapping['coupling_type']
        if coupling_type == "dot_to_dot":
            if 'target_dots' not in mapping:
                raise ValueError(f"Missing target_dots in {barrier_name} for dot_to_dot coupling")
        elif coupling_type in ["reservoir_to_dot", "dot_to_reservoir"]:
            if 'target_dot' not in mapping:
                raise ValueError(f"Missing target_dot in {barrier_name} for {coupling_type} coupling")
        else:
            raise ValueError(f"Unknown coupling_type: {coupling_type}")


def preprocess_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess configuration for experiment.py compatibility.
    
    This function:
    1. Separates plunger and barrier gates in C_DG
    2. Creates experiment.py compatible capacitance config
    3. Creates experiment.py compatible tunneling config
    4. Creates experiment.py compatible sensor config
    
    Args:
        config: Raw configuration dictionary
        
    Returns:
        Preprocessed configuration dictionary
    """
    topology = config['device']['topology']
    plunger_gates = topology['plunger_gates']
    barrier_gates = topology['barrier_gates']
    sensor_gates = topology['sensor_gates']
    
    # Extract and cast original C_DG matrix
    C_DG_full = np.array(config['simulator']['capacitance']['C_DG'], dtype=np.float64)
    # Cast C_DD to float
    C_DD = np.array(config['simulator']['capacitance']['C_DD'], dtype=np.float64)
    # Create plunger-only C_DG for experiment.py
    # experiment.py expects C_DG to only contain plunger gates
    C_DG_plunger = C_DG_full[:, plunger_gates]
    
    # Create barrier-only C_DG for barrier control
    C_DG_barrier = C_DG_full[:, barrier_gates]
    
    # Create plunger + sensor C_DG for experiment.py (includes sensor gates)
    plunger_and_sensor_gates = plunger_gates + sensor_gates
    C_DG_plunger_sensor = C_DG_full[:, plunger_and_sensor_gates]
    
    # Cast bounds_limits to float
    bounds_limits = np.array(config['simulator']['capacitance']['bounds_limits'], dtype=np.float64)
    # Filter bounds_limits to only include plunger and sensor gates
    bounds_limits_plunger_sensor = bounds_limits[plunger_and_sensor_gates]
    # Cast ks to float if present
    ks = config['simulator']['capacitance'].get('ks', None)
    if ks is not None:
        ks = np.array(ks, dtype=np.float64)
    # Tunnel couplings as float
    tunnel_couplings = np.array(config['simulator']['tunneling']['tunnel_couplings'], dtype=np.float64)
    # Create experiment.py compatible capacitance config
    capacitance_config = {
        'C_DD': C_DD,
        'C_Dg': C_DG_plunger_sensor,  # Plunger + sensor gates
        'bounds_limits': bounds_limits_plunger_sensor,
        'ks': ks
    }
    # Create experiment.py compatible tunneling config
    tunneling_config = {
        'tunnel_couplings': tunnel_couplings,
        'temperature': config['simulator']['tunneling']['temperature'],
        'energy_range_factor': config['simulator']['tunneling']['energy_range_factor']
    }
    # Create experiment.py compatible sensor config
    # Convert noise_amplitude structure to match experiment.py expectations
    noise_amplitude_config = config['simulator']['sensor']['noise_amplitude']
    sensor_config = {
        'sensor_dot_indices': config['simulator']['sensor']['sensor_dot_indices'],
        'sensor_detunings': config['simulator']['sensor']['sensor_detunings'],
        'noise_amplitude': {
            'fast_noise': float(noise_amplitude_config['fast_noise']),
            'slow_noise': float(noise_amplitude_config['slow_noise'])
        },
        'peak_width_multiplier': config['simulator']['sensor']['peak_width_multiplier']
    }
    # Store barrier configuration separately
    barrier_config = config['simulator']['barrier']
    # Ensure all base_coupling values are float
    for mapping in barrier_config.get('barrier_mappings', {}).values():
        if 'base_coupling' in mapping:
            mapping['base_coupling'] = float(mapping['base_coupling'])
    # Create preprocessed config
    preprocessed_config = {
        'device': config['device'],
        'simulator': {
            'capacitance': capacitance_config,
            'tunneling': tunneling_config,
            'sensor': sensor_config,
            'barrier': barrier_config,
            # Store original matrices for barrier control
            'C_DG_full': C_DG_full.tolist(),
            'C_DG_barrier': C_DG_barrier.tolist(),
            'plunger_gates': plunger_gates,
            'barrier_gates': barrier_gates,
            'sensor_gates': sensor_gates
        },
        'measurement': config.get('measurement', {}),
        'features': config.get('features', {})
    }
    return preprocessed_config


def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a summary of the loaded configuration.
    
    Args:
        config: Configuration dictionary
    """
    print("=" * 60)
    print("QDarts v4 Configuration Summary")
    print("=" * 60)
    
    # Device topology
    topology = config['device']['topology']
    print(f"Device: {topology['num_dots']} dots, {topology['num_gates']} gates")
    print(f"  Inner dots: {topology['inner_dots']}")
    print(f"  Sensor dots: {topology['sensor_dots']}")
    print(f"  Plunger gates: {topology['plunger_gates']}")
    print(f"  Barrier gates: {topology['barrier_gates']}")
    print(f"  Sensor gates: {topology['sensor_gates']}")
    
    # Capacitance matrices
    sim = config['simulator']
    C_DD = np.array(sim['capacitance']['C_DD'])
    C_DG_plunger = np.array(sim['capacitance']['C_Dg'])
    C_DG_barrier = np.array(sim['C_DG_barrier'])
    
    print(f"\nCapacitance Matrices:")
    print(f"  C_DD shape: {C_DD.shape}")
    print(f"  C_DG (plunger) shape: {C_DG_plunger.shape}")
    print(f"  C_DG (barrier) shape: {C_DG_barrier.shape}")
    
    # Barrier configuration
    barrier_config = sim['barrier']
    print(f"\nBarrier Configuration:")
    for barrier_name, mapping in barrier_config['barrier_mappings'].items():
        coupling_type = mapping['coupling_type']
        if coupling_type == "dot_to_dot":
            target = mapping['target_dots']
        else:
            target = mapping['target_dot']
        print(f"  {barrier_name}: {coupling_type} -> {target}")
    
    # Tunneling configuration
    tunneling = sim['tunneling']
    print(f"\nTunneling Configuration:")
    print(f"  Temperature: {tunneling['temperature']} K")
    print(f"  Energy range factor: {tunneling['energy_range_factor']}")
    
    # Sensor configuration
    sensor = sim['sensor']
    print(f"\nSensor Configuration:")
    print(f"  Sensor dots: {sensor['sensor_dot_indices']}")
    print(f"  Sensor detunings: {sensor['sensor_detunings']} meV")
    
    print("=" * 60)


def get_barrier_voltage_vector(config: Dict[str, Any], barrier_voltages: Dict[str, float]) -> np.ndarray:
    """
    Convert barrier voltage dictionary to voltage vector for all gates.
    
    Args:
        config: Configuration dictionary
        barrier_voltages: Dictionary of barrier voltages
        
    Returns:
        Voltage vector for all gates (plunger + barrier + sensor)
    """
    topology = config['device']['topology']
    plunger_gates = topology['plunger_gates']
    barrier_gates = topology['barrier_gates']
    sensor_gates = topology['sensor_gates']
    
    # Create voltage vector for all gates
    num_gates = topology['num_gates']
    voltage_vector = np.zeros(num_gates)
    
    # Set barrier voltages
    for barrier_name, voltage in barrier_voltages.items():
        # Extract barrier index from name (e.g., "barrier_2" -> 2)
        barrier_idx = int(barrier_name.split('_')[1])
        if barrier_idx in barrier_gates:
            gate_idx = barrier_gates.index(barrier_idx)
            voltage_vector[barrier_idx] = voltage
    
    return voltage_vector


def get_plunger_voltage_vector(config: Dict[str, Any], plunger_voltages: Dict[int, float]) -> np.ndarray:
    """
    Convert plunger voltage dictionary to voltage vector for plunger gates only.
    
    Args:
        config: Configuration dictionary
        plunger_voltages: Dictionary of plunger gate voltages
        
    Returns:
        Voltage vector for plunger gates only
    """
    plunger_gates = config['device']['topology']['plunger_gates']
    voltage_vector = np.zeros(len(plunger_gates))
    
    for gate_idx, voltage in plunger_voltages.items():
        if gate_idx in plunger_gates:
            vector_idx = plunger_gates.index(gate_idx)
            voltage_vector[vector_idx] = voltage
    
    return voltage_vector 