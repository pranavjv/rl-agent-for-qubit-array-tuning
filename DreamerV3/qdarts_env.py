import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import json
import os
import sys
import time
from typing import Dict, Any, Optional, Union, Tuple

# Add qdarts to path
sys.path.append(os.path.dirname(__file__))

# Import QDARTS components
from qdarts.experiment_with_barriers import Experiment

# Set matplotlib backend before importing pyplot to avoid GUI issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io


class QDartsEnv(gym.Env):
    """
    qdarts-based quantum device environment with barrier support.
    
    This environment extends the original qarray_env.py to support:
    - 5 gates (2 plungers + 3 barriers)
    - qdarts physics model with exponential barrier effects
    - Flexible observation space based on config resolution
    - Multi-objective reward function
    """
    
    # Rendering info
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config_path='qdarts_env_config.yaml', render_mode=None, **kwargs):
        """
        Initialize the qdarts environment.
        
        Args:
            config_path: Path to the QDARTS environment configuration file
            render_mode: Rendering mode for the environment
            **kwargs: Additional keyword arguments
        """
        super().__init__()

        # --- Load Configuration ---
        config_path = os.path.join(os.path.dirname(__file__), config_path)
        self.env_config, self.qdarts_config = self._load_qdarts_env_config(config_path)

        # Extract environment configuration
        self.debug = self.env_config['training']['debug']
        self.seed = self.env_config['training']['seed']
        self.max_steps = self.env_config['env']['max_steps']
        self.current_step = 0
        self.tolerance = self.env_config['env']['tolerance']

        # --- Define Action Space ---
        # Combined action space: 2 plungers + 3 barriers
        plunger_config = self.env_config['env']['action_space']['plungers']
        barrier_config = self.env_config['env']['action_space']['barriers']
        
        self.plunger_voltage_min = plunger_config['voltage_range'][0]
        self.plunger_voltage_max = plunger_config['voltage_range'][1]
        self.barrier_voltage_min = barrier_config['voltage_range'][0]
        self.barrier_voltage_max = barrier_config['voltage_range'][1]
        
        # Combined action space: [plunger1, plunger2, barrier1, barrier2, barrier3]
        self.action_space = spaces.Box(
            low=np.array([self.plunger_voltage_min] * 2 + [self.barrier_voltage_min] * 3, dtype=np.float32),
            high=np.array([self.plunger_voltage_max] * 2 + [self.barrier_voltage_max] * 3, dtype=np.float32),
            shape=(5,),  # 2 plungers + 3 barriers
            dtype=np.float32
        )

        # --- Define Observation Space ---
        obs_config = self.env_config['env']['observation_space']
        self.obs_channels = obs_config['channels']
        self.obs_normalization_range = obs_config['normalization_range']
        self.obs_dtype = obs_config['dtype']
        
        # Get resolution from QDARTS config
        resolution = self.qdarts_config['measurement']['resolution']
        self.obs_image_size = [resolution, resolution]
        
        # Define multi-modal observation space using Dict
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(self.obs_image_size[0], self.obs_image_size[1], self.obs_channels),
                dtype=np.uint8
            ),
            'voltages': spaces.Box(
                low=np.array([self.plunger_voltage_min] * 2 + [self.barrier_voltage_min] * 3, dtype=np.float32),
                high=np.array([self.plunger_voltage_max] * 2 + [self.barrier_voltage_max] * 3, dtype=np.float32),
                shape=(5,),  # 2 plungers + 3 barriers
                dtype=np.float32
            )
        })

        # --- Initialize Model Variable ---
        self.experiment = None
        
        # --- Initialize normalization parameters ---
        self._init_normalization_params()

        # --- For Rendering --- 
        self.render_mode = render_mode or self.env_config['training']['render_mode']
        
        # --- Initialize observation storage ---
        self.z = None
        self.device_state = {}


    def _load_qdarts_env_config(self, config_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load consolidated qdarts environment configuration.
        
        Args:
            config_path: Path to the qdarts environment configuration file
            
        Returns:
            Tuple of (env_config, qdarts_config)
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load consolidated config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Split config into env_config and qdarts_config for backward compatibility
        env_config = {
            'env': config['env'],
            'training': config['training']
        }
        
        qdarts_config = {
            'device': config['device'],
            'simulator': config['simulator'],
            'measurement': config['measurement']
        }
        
        # Set observation space resolution based on measurement config
        resolution = qdarts_config['measurement']['resolution']
        env_config['env']['observation_space']['image_size'] = [resolution, resolution]
        
        return env_config, qdarts_config


    def _sample_random_ranges(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sample random values from ranges in the configuration.
        
        Args:
            config: Configuration dictionary with ranges
            
        Returns:
            Configuration with random values sampled from ranges
        """
        sampled_config = {}
        
        # Deep copy the config to avoid modifying the original
        import copy
        sampled_config = copy.deepcopy(config)
        
        # Sample capacitance matrices
        if 'simulator' in sampled_config and 'capacitance' in sampled_config['simulator']:
            self._sample_capacitance_ranges(sampled_config['simulator']['capacitance'])
        
        # Sample tunneling parameters
        if 'simulator' in sampled_config and 'tunneling' in sampled_config['simulator']:
            self._sample_tunneling_ranges(sampled_config['simulator']['tunneling'])
        
        # Sample barrier parameters
        if 'simulator' in sampled_config and 'barrier' in sampled_config['simulator']:
            self._sample_barrier_ranges(sampled_config['simulator']['barrier'])
        
        # Sample sensor parameters
        if 'simulator' in sampled_config and 'sensor' in sampled_config['simulator']:
            self._sample_sensor_ranges(sampled_config['simulator']['sensor'])
        
        # Convert measurement configuration
        if 'measurement' in sampled_config:
            self._convert_measurement_config(sampled_config['measurement'])
        
        return sampled_config


    def _sample_capacitance_ranges(self, capacitance_config: Dict[str, Any]) -> None:
        """Sample random values for capacitance matrices."""
        # Sample C_DD matrix
        if 'C_DD' in capacitance_config:
            C_DD = capacitance_config['C_DD']
            for i, row in enumerate(C_DD):
                for j, range_dict in enumerate(row):
                    if isinstance(range_dict, dict) and 'min' in range_dict and 'max' in range_dict:
                        min_val = float(range_dict['min'])
                        max_val = float(range_dict['max'])
                        C_DD[i][j] = np.random.uniform(min_val, max_val)
            # Convert to numpy array
            capacitance_config['C_DD'] = np.array(C_DD, dtype=np.float64)
        
        # Sample C_DG matrix
        if 'C_DG' in capacitance_config:
            C_DG = capacitance_config['C_DG']
            for i, row in enumerate(C_DG):
                for j, range_dict in enumerate(row):
                    if isinstance(range_dict, dict) and 'min' in range_dict and 'max' in range_dict:
                        min_val = float(range_dict['min'])
                        max_val = float(range_dict['max'])
                        C_DG[i][j] = np.random.uniform(min_val, max_val)
            # Convert to numpy array
            capacitance_config['C_DG'] = np.array(C_DG, dtype=np.float64)
        
        # Handle non-range values
        if 'bounds_limits' in capacitance_config:
            capacitance_config['bounds_limits'] = float(capacitance_config['bounds_limits'])
        
        if 'ks' in capacitance_config:
            capacitance_config['ks'] = float(capacitance_config['ks'])
            

    def _sample_tunneling_ranges(self, tunneling_config: Dict[str, Any]) -> None:
        """Sample random values for tunneling parameters."""
        # Sample tunnel_couplings matrix
        if 'tunnel_couplings' in tunneling_config:
            tunnel_couplings = tunneling_config['tunnel_couplings']
            for i, row in enumerate(tunnel_couplings):
                for j, range_dict in enumerate(row):
                    if isinstance(range_dict, dict) and 'min' in range_dict and 'max' in range_dict:
                        min_val = float(range_dict['min'])
                        max_val = float(range_dict['max'])
                        tunnel_couplings[i][j] = np.random.uniform(min_val, max_val)
            # Convert to numpy array
            tunneling_config['tunnel_couplings'] = np.array(tunnel_couplings, dtype=np.float64)
        
        # Sample temperature
        if 'temperature' in tunneling_config and isinstance(tunneling_config['temperature'], dict):
            temp_range = tunneling_config['temperature']
            min_temp = float(temp_range['min'])
            max_temp = float(temp_range['max'])
            tunneling_config['temperature'] = float(np.random.uniform(min_temp, max_temp))
        
        # Sample energy_range_factor
        if 'energy_range_factor' in tunneling_config and isinstance(tunneling_config['energy_range_factor'], dict):
            energy_range = tunneling_config['energy_range_factor']
            min_energy = float(energy_range['min'])
            max_energy = float(energy_range['max'])
            tunneling_config['energy_range_factor'] = float(np.random.uniform(min_energy, max_energy))

    def _sample_barrier_ranges(self, barrier_config: Dict[str, Any]) -> None:
        """Sample random values for barrier parameters."""
        if 'barrier_mappings' in barrier_config:
            for barrier_name, mapping in barrier_config['barrier_mappings'].items():
                for param_name, range_dict in mapping.items():
                    if isinstance(range_dict, dict) and 'min' in range_dict and 'max' in range_dict:
                        min_val = float(range_dict['min'])
                        max_val = float(range_dict['max'])
                        mapping[param_name] = float(np.random.uniform(min_val, max_val))

    def _sample_sensor_ranges(self, sensor_config: Dict[str, Any]) -> None:
        """Sample random values for sensor parameters."""
        # Sample sensor_detunings
        if 'sensor_detunings' in sensor_config:
            for i, range_dict in enumerate(sensor_config['sensor_detunings']):
                if isinstance(range_dict, dict) and 'min' in range_dict and 'max' in range_dict:
                    min_val = float(range_dict['min'])
                    max_val = float(range_dict['max'])
                    sensor_config['sensor_detunings'][i] = np.random.uniform(min_val, max_val)
            # Convert to numpy array
            sensor_config['sensor_detunings'] = np.array(sensor_config['sensor_detunings'], dtype=np.float64)
        
        # Sample noise_amplitude
        if 'noise_amplitude' in sensor_config:
            for noise_type, range_dict in sensor_config['noise_amplitude'].items():
                if isinstance(range_dict, dict) and 'min' in range_dict and 'max' in range_dict:
                    min_val = float(range_dict['min'])
                    max_val = float(range_dict['max'])
                    sensor_config['noise_amplitude'][noise_type] = float(np.random.uniform(min_val, max_val))
        
        # Sample peak_width_multiplier
        if 'peak_width_multiplier' in sensor_config and isinstance(sensor_config['peak_width_multiplier'], dict):
            range_dict = sensor_config['peak_width_multiplier']
            min_val = float(range_dict['min'])
            max_val = float(range_dict['max'])
            sensor_config['peak_width_multiplier'] = float(np.random.uniform(min_val, max_val))
        
        # Handle non-range values
        if 'sensor_dot_indices' in sensor_config:
            sensor_config['sensor_dot_indices'] = np.array(sensor_config['sensor_dot_indices'], dtype=int)
        
        if 'sensor_gate_indices' in sensor_config:
            sensor_config['sensor_gate_indices'] = np.array(sensor_config['sensor_gate_indices'], dtype=int)

    def _convert_measurement_config(self, measurement_config: Dict[str, Any]) -> None:
        """Convert measurement configuration to proper types."""
        if 'voltage_range' in measurement_config:
            if 'min' in measurement_config['voltage_range']:
                measurement_config['voltage_range']['min'] = np.array(measurement_config['voltage_range']['min'], dtype=np.float64)
            if 'max' in measurement_config['voltage_range']:
                measurement_config['voltage_range']['max'] = np.array(measurement_config['voltage_range']['max'], dtype=np.float64)
        
        if 'sweep_matrix' in measurement_config:
            measurement_config['sweep_matrix'] = np.array(measurement_config['sweep_matrix'], dtype=np.float64)

    def _compute_ground_truth_barrier_targets(self, barrier_config: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, bool], Dict[str, float]]:
        """
        Compute the ground truth barrier voltages that achieve cutoff tunnel coupling.
        
        Args:
            barrier_config: Sampled barrier configuration for this episode
            
        Returns:
            targets_clamped: Dict mapping barrier_name to clamped target voltage
            targets_raw: Dict mapping barrier_name to raw computed target voltage
            reachable: Dict mapping barrier_name to whether target is reachable within bounds
            epsilons_used: Dict mapping barrier_name to epsilon threshold used
        """
        targets_clamped = {}
        targets_raw = {}
        reachable = {}
        epsilons_used = {}
        
        epsilon_abs = barrier_config.get('cutoff_coupling_abs', 3e-5)
        # Ensure epsilon is positive and not too small for numerical stability
        epsilon_abs = max(float(epsilon_abs), 1e-12)
        
        barrier_mappings = barrier_config['barrier_mappings']
        
        for barrier_name, mapping in barrier_mappings.items():
            try:
                # Extract parameters for this barrier
                base_coupling = float(mapping['base_coupling'])
                alpha = float(mapping['alpha'])
                voltage_offset = float(mapping['voltage_offset'])
                
                # Validate parameters
                if base_coupling <= 0 or alpha <= 0:
                    # Invalid parameters, mark as unreachable
                    targets_raw[barrier_name] = self.barrier_voltage_max
                    targets_clamped[barrier_name] = self.barrier_voltage_max
                    reachable[barrier_name] = False
                    epsilons_used[barrier_name] = epsilon_abs
                    continue
                
                # Compute target voltage for coupling <= epsilon
                # coupling(V) = base_cobarrier_nameupling * exp(-alpha * (V - voltage_offset))
                # Solve for coupling <= epsilon: V = offset + (ln(base) - ln(epsilon)) / alpha
                if base_coupling <= epsilon_abs:
                    # Base coupling is already below threshold, any voltage >= offset works
                    V_epsilon_raw = voltage_offset
                else:
                    V_epsilon_raw = voltage_offset + (np.log(base_coupling) - np.log(epsilon_abs)) / alpha
                
                # Clamp to action bounds
                V_epsilon_clamped = np.clip(V_epsilon_raw, self.barrier_voltage_min, self.barrier_voltage_max)
                
                # Check reachability: target is reachable if Vmax >= V_epsilon_raw
                is_reachable = self.barrier_voltage_max >= V_epsilon_raw
                
                # Store results
                targets_raw[barrier_name] = V_epsilon_raw
                targets_clamped[barrier_name] = V_epsilon_clamped
                reachable[barrier_name] = is_reachable
                epsilons_used[barrier_name] = epsilon_abs
                
                if self.debug:
                    print(f"Barrier {barrier_name}: base={base_coupling:.2e}, alpha={alpha:.2f}, offset={voltage_offset:.3f}")
                    print(f"  Target raw: {V_epsilon_raw:.3f}, clamped: {V_epsilon_clamped:.3f}, reachable: {is_reachable}")
                
            except (KeyError, ValueError, TypeError) as e:
                # Handle any errors in parameter extraction
                if self.debug:
                    print(f"Error computing target for {barrier_name}: {e}")
                
                # Default to maximum voltage if computation fails
                targets_raw[barrier_name] = self.barrier_voltage_max
                targets_clamped[barrier_name] = self.barrier_voltage_max
                reachable[barrier_name] = False
                epsilons_used[barrier_name] = epsilon_abs
        
        return targets_clamped, targets_raw, reachable, epsilons_used

    def _check_couplings_below_cutoff(self) -> Tuple[bool, Dict[str, float], Dict[str, bool]]:
        """
        Check if current tunnel couplings are below the cutoff threshold.
        
        Returns:
            all_below_cutoff: True if all relevant couplings are below threshold
            current_couplings: Dict mapping coupling location to current value
            couplings_below: Dict mapping coupling location to whether it's below threshold
        """
        if self.experiment is None:
            return False, {}, {}
        
        try:
            # Get current tunnel couplings from the experiment
            current_couplings_matrix = self.experiment.get_current_tunnel_couplings()
            
            # Extract relevant couplings based on barrier mappings
            barrier_config = self.experiment.barrier_config['barrier_mappings']
            epsilon_abs = barrier_config.get('cutoff_coupling_abs', 3e-5)
            
            current_couplings = {}
            couplings_below = {}
            
            # Check each barrier's effect on couplings
            for barrier_name, mapping in barrier_config['barrier_mappings'].items():
                coupling_type = mapping['coupling_type']
                
                if coupling_type == "dot_to_dot":
                    # Dot-to-dot coupling affects symmetric entries
                    target_dots = mapping['target_dots']
                    if len(target_dots) == 2:
                        i, j = target_dots[0], target_dots[1]
                        coupling_value = current_couplings_matrix[i, j]
                        coupling_key = f"dot_{i}_to_dot_{j}"
                        current_couplings[coupling_key] = coupling_value
                        couplings_below[coupling_key] = coupling_value <= epsilon_abs
                        
                        # Also check symmetric entry
                        coupling_value_sym = current_couplings_matrix[j, i]
                        coupling_key_sym = f"dot_{j}_to_dot_{i}"
                        current_couplings[coupling_key_sym] = coupling_value_sym
                        couplings_below[coupling_key_sym] = coupling_value_sym <= epsilon_abs
                
                elif coupling_type == "reservoir_to_dot":
                    # Reservoir coupling affects one entry
                    target_dot = mapping['target_dot']
                    coupling_value = current_couplings_matrix[target_dot, target_dot]  # Diagonal entry for reservoir
                    coupling_key = f"reservoir_to_dot_{target_dot}"
                    current_couplings[coupling_key] = coupling_value
                    couplings_below[coupling_key] = coupling_value <= epsilon_abs
                
                elif coupling_type == "dot_to_reservoir":
                    # Dot to reservoir coupling affects one entry
                    target_dot = mapping['target_dot']
                    coupling_value = current_couplings_matrix[target_dot, target_dot]  # Diagonal entry for reservoir
                    coupling_key = f"dot_{target_dot}_to_reservoir"
                    current_couplings[coupling_key] = coupling_value
                    couplings_below[coupling_key] = coupling_value <= epsilon_abs
            
            # Check if all relevant couplings are below threshold
            all_below_cutoff = all(couplings_below.values()) if couplings_below else False
            
            if self.debug:
                print(f"Current couplings: {current_couplings}")
                print(f"Couplings below cutoff: {couplings_below}")
                print(f"All below cutoff: {all_below_cutoff}")
            
            return all_below_cutoff, current_couplings, couplings_below
            
        except Exception as e:
            if self.debug:
                print(f"Error checking couplings: {e}")
            return False, {}, {}



    def _init_normalization_params(self):
        """
        Initialize adaptive normalization parameters.
        Start with conservative bounds and update them as new data is encountered.
        """
        if self.debug:
            print("Initializing adaptive normalization parameters...")
        
        # Start with conservative bounds based on typical charge sensor data ranges
        # These will be updated as we encounter actual data
        self.data_min = 0.13
        self.data_max = 0.87
        self.global_min = float('inf')
        self.global_max = float('-inf')
        self.episode_min = float('inf')
        self.episode_max = float('-inf')
        self.update_count = 0 

    def _load_model(self):
        """
        Load qdarts model with barrier support using sampled configuration.
        """
        # Sample random values from ranges for this episode
        sampled_qdarts_config = self._sample_random_ranges(self.qdarts_config)
        
        # Extract configurations from sampled config
        capacitance_config = sampled_qdarts_config['simulator']['capacitance']
        tunneling_config = sampled_qdarts_config['simulator']['tunneling']
        sensor_config = sampled_qdarts_config['simulator']['sensor']
        barrier_config = sampled_qdarts_config['simulator']['barrier']
        
        # Create QDARTS experiment
        self.experiment = Experiment(
            capacitance_config=capacitance_config,
            tunneling_config=tunneling_config,
            sensor_config=sensor_config,
            barrier_config=barrier_config,
            print_logs=self.debug
        )
        
        # Set initial barrier voltages
        initial_barrier_voltages = barrier_config['default_barrier_voltages']
        self.experiment.update_tunnel_couplings(initial_barrier_voltages)
        
        return self.experiment

    def _get_obs(self):
        """
        Get current observation with flexible resolution based on config.
        """
        # Get charge sensor data using QDARTS
        current_plungers = self.device_state["current_plunger_voltages"]
        current_barrier_voltages = self.device_state["current_barrier_voltages"]

        current_voltages = self._build_voltage_grid(current_plungers)

        self.z = self._get_charge_sensor_data_qdarts(current_voltages)
        z = self.z
        
        # Extract and normalize image observation (flexible resolution)
        channel_data = z[:, :, 0]  # Shape: (resolution, resolution)

        image_obs = self._normalize_observation(channel_data)
        
        # Combine all voltages
        all_voltages = np.concatenate([current_plungers, current_barrier_voltages])

        observation = {
            'image': image_obs,  # Shape: (resolution, resolution, 1)
            'voltages': all_voltages  # Shape: (5,)
        }
        
        return observation

    def _get_charge_sensor_data_qdarts(self, voltages):
        """
        Get charge sensor data using qdarts 
        """
        measurement = self.qdarts_config['measurement']
        
        # Unpack the tuple: (x_voltages, y_voltages, barrier_voltages)
        x_voltages, y_voltages, _ = voltages


        sweep_matrix = measurement['sweep_matrix']
        plane_axes = []
        for i in range(2):
            for j, row in enumerate(sweep_matrix):
                if abs(row[i]) > 1e-10:
                    plane_axes.append(j)
                    break
        
        # Ensure we have exactly 2 plane axes
        if len(plane_axes) < 2:
            plane_axes = [0, 1]  # Default to first two gates
        
        # Generate CSD using qdarts
        try:
            xout, yout, _, polytopes, sensor_values, v_offset = self.experiment.generate_CSD(
                x_voltages=x_voltages,
                y_voltages=y_voltages,
                plane_axes=plane_axes,
                target_state=self.qdarts_config['device']['target_state'],
                use_sensor_signal=True,  # Get sensor response
                compensate_sensors=True,
                compute_polytopes=False
            )
            return sensor_values  # Shape: (resolution, resolution, num_sensors)
            
        except Exception as e:
            if self.debug:
                print(f"Error generating CSD: {e}")
            # Return zeros if CSD generation fails
            resolution = measurement['resolution']
            return np.zeros((resolution, resolution, 1))

    def _normalize_observation(self, raw_data):
        """
        Normalize observation data to the specified range.
        """
        # Update normalization bounds
        self._update_normalization_bounds(raw_data)
        
        # Normalize to [0, 255] for uint8 image observation
        normalized = (raw_data - self.data_min) / (self.data_max - self.data_min)
        normalized = np.clip(normalized, 0, 1)
        normalized = (normalized * 255).astype(np.uint8)
        
        # Add channel dimension if needed
        if len(normalized.shape) == 2:
            normalized = normalized[:, :, np.newaxis]
        
        return normalized

    def _update_normalization_bounds(self, raw_data):
        """
        Update adaptive normalization bounds.
        """
        current_min = np.min(raw_data)
        current_max = np.max(raw_data)
        
        # Update episode bounds
        self.episode_min = min(self.episode_min, current_min)
        self.episode_max = max(self.episode_max, current_max)
        
        # Update global bounds
        self.global_min = min(self.global_min, current_min)
        self.global_max = max(self.global_max, current_max)
        
        # Update adaptive bounds (exponential moving average)
        alpha = 0.1  # Learning rate for bounds update
        self.data_min = (1 - alpha) * self.data_min + alpha * current_min
        self.data_max = (1 - alpha) * self.data_max + alpha * current_max
        
        self.update_count += 1

    def _extract_voltage_centers(self, voltages):
        """
        Extract voltage centers from the voltage arrays.
        """
        # Unpack the tuple: (x_voltages, y_voltages, barrier_voltages)
        x_voltages, y_voltages, _ = voltages
        
        # Extract center voltages from the 1D arrays
        center_x = x_voltages[len(x_voltages) // 2]  # Middle of x array
        center_y = y_voltages[len(y_voltages) // 2]  # Middle of y array
        
        return np.array([center_x, center_y])
    
    def _build_voltage_grid(self, center):
        """
        Build voltage arrays centered around the given center.
        Returns 1D arrays for x and y axes plus barrier voltages.
        """
        measurement = self.qdarts_config['measurement']
        resolution = measurement['resolution']
        
        # Get current barrier voltages from device state
        current_barriers = self.device_state.get("current_barrier_voltages", np.zeros(3))
        
        # Create 1D voltage arrays centered around the current plunger positions
        x_voltages = np.linspace(
            measurement['voltage_range']['min'][0] + center[0], 
            measurement['voltage_range']['max'][0] + center[0], 
            resolution
        )
        y_voltages = np.linspace(
            measurement['voltage_range']['min'][1] + center[1], 
            measurement['voltage_range']['max'][1] + center[1], 
            resolution
        )
        
        # Return tuple of 1D arrays: (x_voltages, y_voltages, barrier_voltages)
        return x_voltages, y_voltages, current_barriers
        

    def _apply_voltages(self, action):
        """
        Apply voltage settings to the quantum device.
        
        Args:
            action: np.ndarray(5,) containing [plunger1, plunger2, barrier1, barrier2, barrier3]
        """
        # Split action into plungers and barriers
        plungers = action[:2]  # First 2 elements
        barriers = action[2:]  # Last 3 elements

        self.device_state["current_plunger_voltages"] = plungers
        self.device_state["current_barrier_voltages"] = barriers


        barrier_voltages = {
            'barrier_2': barriers[0],
            'barrier_3': barriers[1], 
            'barrier_4': barriers[2]
        }
        self.experiment.update_tunnel_couplings(barrier_voltages)

    def _get_distance(self):
        """
        Get the distance vector between the current state and the target state.
        """
        plunger_truth_center = self.device_state["ground_truth_plungers"][:2]
        barrier_truth_center = self.device_state["ground_truth_barrier_voltages"]
        
        plunger_voltages = self.device_state["current_plunger_voltages"]
        barrier_voltages = self.device_state["current_barrier_voltages"]


        plunger_difference_vector = plunger_truth_center - plunger_voltages
        barrier_difference_vector = (barrier_truth_center - barrier_voltages)*0.5

        total_difference_vector = np.concatenate([plunger_difference_vector, barrier_difference_vector])
        return np.linalg.norm(total_difference_vector)

    def _get_reward(self):
        """
        Enhanced reward function considering both plunger alignment and barrier optimization.
        """
        terminated = False
        
        # Difference Vector
        distance = self._get_distance()
        previous_distance = self.device_state["previous_distance"]
        
        # Progress reward minus distance penalty to encourage quick termination
        total_reward = (previous_distance - distance) - distance * 0.1

        self.device_state["previous_distance"] = distance

        if self.debug:
            print(f"distance: {distance}")
            print(f"previous_distance: {previous_distance}")
            print(f"total_reward: {total_reward}")

        # Termination bonus
        if distance < self.tolerance:
            total_reward += 200.0
            terminated = True
            if self.debug:
                print(f"terminated")
        
        return total_reward, terminated



    def _random_center(self):
        """
        Generate a random center for the voltage window.
        """
        # Generate plunger random center within the voltage range
        env_config = self.env_config['env']
        x_range = env_config['action_space']['plungers']['voltage_range'][1] - env_config['action_space']['plungers']['voltage_range'][0]
        y_range = env_config['action_space']['plungers']['voltage_range'][1] - env_config['action_space']['plungers']['voltage_range'][0]
        
        center_x = np.random.uniform(
            env_config['action_space']['plungers']['voltage_range'][0] + x_range * 0.1,
            env_config['action_space']['plungers']['voltage_range'][1] - x_range * 0.1
        )
        center_y = np.random.uniform(
            env_config['action_space']['plungers']['voltage_range'][0] + y_range * 0.1,
            env_config['action_space']['plungers']['voltage_range'][1] - y_range * 0.1
        )

        # Generate barrier random center within the voltage range
        barrier_range = env_config['action_space']['barriers']['voltage_range'][1] - env_config['action_space']['barriers']['voltage_range'][0]

        center_1 = np.random.uniform(
            env_config['action_space']['barriers']['voltage_range'][0] + barrier_range * 0.1,
            env_config['action_space']['barriers']['voltage_range'][1] - barrier_range * 0.1
        )

        center_2 = np.random.uniform(
            env_config['action_space']['barriers']['voltage_range'][0] + barrier_range * 0.1,
            env_config['action_space']['barriers']['voltage_range'][1] - barrier_range * 0.1
        )

        center_3 = np.random.uniform(   
            env_config['action_space']['barriers']['voltage_range'][0] + barrier_range * 0.1,
            env_config['action_space']['barriers']['voltage_range'][1] - barrier_range * 0.1
        )
        
        return np.array([center_x, center_y, center_1, center_2, center_3]) 
    

    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns the initial observation.

        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Additional options for reset.

        Returns:
            observation (dict): The initial observation of the space.
            info (dict): A dictionary with auxiliary diagnostic information.
        """
        # Handle seed if provided
        if seed is not None:
            super().reset(seed=seed)
        else:
            super().reset(seed=None)

        # --- Reset the environment's state ---
        self.current_step = 0

        # Initialize episode-specific normalization statistics
        self.episode_min = float('inf')
        self.episode_max = float('-inf')

        # --- Initialize episode-specific Model ---
        self.experiment = self._load_model()
        
        # Compute ground-truth center of target state's polytope (first two gates)
        target_state = np.array(self.qdarts_config['device']['target_state'], dtype=int)
        _gt_point = self.experiment.capacitance_sim.boundaries(target_state).point_inside
        ground_truth_center = _gt_point[:2] if _gt_point is not None else np.zeros(2)

        
        # Initialize episode-specific voltage state
        # Center of current window
        centers = self._random_center()
        barrier_voltages = {
            'barrier_2': centers[-3],
            'barrier_3': centers[-2], 
            'barrier_4': centers[-1]
        }
        self.experiment.update_tunnel_couplings(barrier_voltages)
        

        
        # Compute ground truth barrier targets for cutoff tunnel coupling
        barrier_config = self.experiment.barrier_config
        targets_clamped, _, _, _ = self._compute_ground_truth_barrier_targets(barrier_config)
        barrier_names = ['barrier_2', 'barrier_3', 'barrier_4']
        ground_truth_barrier_voltages = np.array([targets_clamped[name] for name in barrier_names])

                
        # Device state variables (episode-specific)
        self.device_state = {
            "experiment": self.experiment,
            "current_plunger_voltages": centers[:2],
            "current_barrier_voltages": centers[2:], 
            "ground_truth_plungers": ground_truth_center,
            "ground_truth_barrier_voltages": ground_truth_barrier_voltages,
            "previous_distance": None, #updated each time you calculate reward, intialised two lines below this
        }

        #calculate the distance between the ground truth  voltages and the current voltages
        self.device_state["previous_distance"] = self._get_distance()


        # --- Return the initial observation ---
        observation = self._get_obs() 
        info = self._get_info() 

        return observation, info

    def step(self, action):
        """
        Updates the environment state based on the agent's action.

        Args:
            action: An action provided by the agent (np.ndarray of shape (5,))

        Returns:
            observation (dict): The observation of the environment's state.
            reward (float): The amount of reward returned after previous action.
            terminated (bool): Whether the episode has ended (e.g., reached a goal).
            truncated (bool): Whether the episode was cut short (e.g., time limit).
            info (dict): A dictionary with auxiliary diagnostic information.
        """
        # --- Update the environment's state based on the action ---
        self.current_step += 1
        
        # Apply voltages to the device
        self._apply_voltages(action)

        # --- Get the current observation ---
        observation = self._get_obs()

        # --- Calculate the reward ---
        reward, terminated = self._get_reward()

        # --- Check if the episode has timed out ---
        truncated = self.current_step >= self.max_steps

        # --- Get additional information ---
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_info(self):
        """
        Helper method to get auxiliary information about the environment's state.

        Can be used for debugging or logging, but the agent should not use it for learning.
        """
        return {
            "device_state": self.device_state,
            "current_step": self.current_step,
            "normalization_range": [self.data_min, self.data_max],
            "normalization_updates": self.update_count,
            "global_data_range": [self.global_min, self.global_max],
            "episode_data_range": [self.episode_min, self.episode_max],
            "observation_structure": {
                "image_shape": (self.obs_image_size[0], self.obs_image_size[1], self.obs_channels),
                "voltage_shape": (5,),
                "total_modalities": 2
            },
            "qdarts_config": {
                "resolution": self.qdarts_config['measurement']['resolution'],
                "num_dots": self.qdarts_config['device']['topology']['num_dots'],
                "num_gates": self.qdarts_config['device']['topology']['num_gates']
            },
            "barrier_targets": {
                "ground_truth_voltages": self.device_state.get("ground_truth_barrier_voltages", np.zeros(3)).tolist(),
            }
        } 

    def render(self, title=None):
        """
        Render the environment.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame(title=title)
            return None

    def _render_frame(self, inference_plot=False, title=None):
        """
        Render a frame of the environment.
        """
        if self.z is None:
            return None
        
        # Get the full measurement voltage range from the config
        measurement = self.qdarts_config['measurement']
        measurement_x_min, measurement_x_max = measurement['voltage_range']['min'][0], measurement['voltage_range']['max'][0]
        measurement_y_min, measurement_y_max = measurement['voltage_range']['min'][1], measurement['voltage_range']['max'][1]
        
        # Get current plunger positions for the measurement window
        current_plungers = self.device_state["current_plunger_voltages"]
        voltage_grid = self._build_voltage_grid(current_plungers)
        x_voltages, y_voltages, barrier_voltages = voltage_grid
        
        # Get the full plunger voltage range for the position plot
        plunger_config = self.env_config['env']['action_space']['plungers']
        full_x_min, full_x_max = plunger_config['voltage_range']
        full_y_min, full_y_max = plunger_config['voltage_range']
        
        # Create figure with 3 subplots: CSD, voltage positions, and barrier info
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Charge Sensor Data with FULL measurement voltage range
        im1 = axes[0].imshow(self.z[:, :, 0], cmap='viridis', aspect='auto', 
                             extent=[measurement_x_min, measurement_x_max, measurement_y_min, measurement_y_max], origin='lower')
        axes[0].set_title('Charge Sensor Data (Full Measurement Range)')
        axes[0].set_xlabel(f'Plunger 1 Voltage (V)\nRange: [{measurement_x_min:.3f}, {measurement_x_max:.3f}]')
        axes[0].set_ylabel(f'Plunger 2 Voltage (V)\nRange: [{measurement_y_min:.3f}, {measurement_y_max:.3f}]')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot 2: Voltage positions with FULL plunger voltage range
        ground_truth_plungers = self.device_state["ground_truth_plungers"]
        current_plungers = self.device_state["current_plunger_voltages"]
        
        axes[1].scatter(current_plungers[0], current_plungers[1], c='red', s=50)
        axes[1].scatter(ground_truth_plungers[0], ground_truth_plungers[1], c='green', s=50)
        axes[1].set_xlabel(f'Plunger 1 Voltage (V)\nFull Range: [{full_x_min:.3f}, {full_x_max:.3f}]')
        axes[1].set_ylabel(f'Plunger 2 Voltage (V)\nFull Range: [{full_y_min:.3f}, {full_y_max:.3f}]')
        axes[1].set_title('Voltage Positions (Full Range)')
        axes[1].grid(True)
        
        # Set axis limits for the full plunger range
        axes[1].set_xlim(full_x_min, full_x_max)
        axes[1].set_ylim(full_y_min, full_y_max)
        
        # Add a rectangle showing the current measurement window
        from matplotlib.patches import Rectangle
        current_x_min, current_x_max = x_voltages.min(), x_voltages.max()
        current_y_min, current_y_max = y_voltages.min(), y_voltages.max()
        rect = Rectangle((current_x_min, current_y_min), current_x_max - current_x_min, current_y_max - current_y_min, 
                        linewidth=2, edgecolor='blue', facecolor='none', alpha=0.7, linestyle='--')
        axes[1].add_patch(rect)


        # Plot 3: Barrier voltage information
        current_barriers = self.device_state["current_barrier_voltages"]
        ground_truth_barriers = self.device_state["ground_truth_barrier_voltages"]
        
        # Create barrier voltage display
        barrier_names = ['Barrier 2', 'Barrier 3', 'Barrier 4']
        barrier_text = "Barrier Voltages:\n\n"
        
        for i, (name, current, target) in enumerate(zip(barrier_names, current_barriers, ground_truth_barriers)):
            # Color code based on how close to target
            distance = abs(current - target)
            if distance < 0.1:
                color = 'green'
            elif distance < 0.3:
                color = 'orange'
            else:
                color = 'red'
            
            barrier_text += f"{name}:\n"
            barrier_text += f"  Current: {current:.3f}V\n"
            barrier_text += f"  Target:  {target:.3f}V\n"
            barrier_text += f"  Diff:    {current - target:+.3f}V\n\n"
        
        # Add plunger information
        plunger_distance = np.linalg.norm(ground_truth_plungers - current_plungers)
        barrier_text += f"Plunger Distance to Target:\n{plunger_distance:.3f}V\n\n"
        barrier_text += f"Step: {self.current_step}/{self.max_steps}"
        
        # Display barrier info as text
        axes[2].text(0.1, 0.9, barrier_text, transform=axes[2].transAxes, 
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        axes[2].set_title('Device State')
        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].axis('off')  # Hide axis for text display
        
        plt.tight_layout()
        
        # Convert to RGB array
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')

        if self.render_mode == "human":
            plt.savefig(f'{title}.png', format='png', dpi=100, bbox_inches='tight')

        buf.seek(0)
        img_array = plt.imread(buf)
        plt.close()
        
        return img_array

 
    def get_raw_observation(self):
        """
        Get raw (unnormalized) observation data for debugging purposes.
        
        Returns:
            np.ndarray: Raw charge sensor data of shape (height, width)
        """
        if hasattr(self, 'z') and self.z is not None:
            return self.z[:, :, 0]
        else:
            return np.zeros((self.obs_image_size[0], self.obs_image_size[1]))

    def seed(self, seed=None):
        """
        Set the random seed for the environment.
        """
        if seed is not None:
            np.random.seed(seed)
        return [seed] 


def get_ground_truths(env):
    """
    Get the ground truths for the plunger and barrier voltages.
    """
    return np.concatenate([env.device_state["ground_truth_plungers"], env.device_state["ground_truth_barrier_voltages"]])  

if __name__ == "__main__":
    env = QDartsEnv()
    env.reset()
    env.render(title="initial")

    #test setting action to ground truth plunger and barrier voltages
    ground_truths = get_ground_truths(env)
    print(f"Ground truths: {ground_truths}")
    action = ground_truths
    env.step(action*2)
    env.render(title="ground_truth")
