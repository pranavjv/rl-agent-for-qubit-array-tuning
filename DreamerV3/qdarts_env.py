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
    QDARTS-based quantum device environment with barrier support.
    
    This environment extends the original qarray_env.py to support:
    - 5 gates (2 plungers + 3 barriers)
    - QDARTS physics model with exponential barrier effects
    - Flexible observation space based on config resolution
    - Multi-objective reward function
    """
    
    # Rendering info
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config_path='qdarts_env_config.yaml', render_mode=None, **kwargs):
        """
        Initialize the QDARTS environment.
        
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

        # --- Initialize Model ---
        self.experiment = None
        
        # --- Initialize normalization parameters ---
        self._init_normalization_params()

        # --- For Rendering --- 
        self.render_fps = self.env_config['training']['render_fps']
        self.render_mode = render_mode or self.env_config['training']['render_mode']
        
        # --- Initialize observation storage ---
        self.z = None
        self.device_state = {}

    def _load_qdarts_env_config(self, config_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load consolidated qdarts environment configuration.
        
        Args:
            config_path: Path to the QDARTS environment configuration file
            
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
        Load QDARTS model with barrier support using sampled configuration.
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
        current_voltages = self.device_state["current_voltages"]
        self.z = self._get_charge_sensor_data_qdarts(current_voltages)
        z = self.z
        
        # Extract and normalize image observation (flexible resolution)
        channel_data = z[:, :, 0]  # Shape: (resolution, resolution)
        image_obs = self._normalize_observation(channel_data)
        
        # Extract voltage centers for plungers
        plunger_centers = self._extract_voltage_centers(current_voltages)
        
        # Get barrier voltages
        barrier_voltages = self.device_state.get("current_barrier_voltages", np.zeros(3))
        
        # Combine all voltages
        all_voltages = np.concatenate([plunger_centers, barrier_voltages])
        
        observation = {
            'image': image_obs,  # Shape: (resolution, resolution, 1)
            'voltages': all_voltages  # Shape: (5,)
        }
        
        return observation

    def _get_charge_sensor_data_qdarts(self, voltages):
        """
        Get charge sensor data using QDARTS (based on qdarts_v5.py).
        """
        # Extract measurement configuration (based on qdarts_v5.py)
        measurement = self.qdarts_config['measurement']
        
        # Create voltage ranges (based on qdarts_v5.py)
        x_voltages = np.linspace(
            measurement['voltage_range']['min'][0], 
            measurement['voltage_range']['max'][0], 
            measurement['resolution']
        )
        y_voltages = np.linspace(
            measurement['voltage_range']['min'][1], 
            measurement['voltage_range']['max'][1], 
            measurement['resolution']
        )
        
        # Convert sweep matrix to plane axes (based on qdarts_v5.py)
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
        
        # Generate CSD using QDARTS (based on qdarts_v5.py)
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
        Extract voltage centers from the voltage grid.
        """
        # For QDARTS, we need to extract the center voltages from the grid
        # This is similar to the original implementation but adapted for QDARTS
        if len(voltages.shape) == 3:
            # 3D voltage grid: (height, width, num_gates)
            height, width, num_gates = voltages.shape
            center_i, center_j = height // 2, width // 2
            centers = voltages[center_i, center_j, :2]  # Only first 2 gates (plungers)
        else:
            # Fallback: use zeros if voltage grid is not available
            centers = np.zeros(2)
        
        return centers 

    def _apply_voltages(self, action):
        """
        Apply voltage settings to the quantum device.
        
        Args:
            action: np.ndarray(5,) containing [plunger1, plunger2, barrier1, barrier2, barrier3]
        """
        # Split action into plungers and barriers
        plungers = action[:2]  # First 2 elements
        barriers = action[2:]  # Last 3 elements
        
        # Apply plunger voltages (same logic as current)
        self._apply_plunger_voltages(plungers)
        
        # Apply barrier voltages (new - based on qdarts_v5.py)
        self._apply_barrier_voltages(barriers)

    def _apply_plunger_voltages(self, plungers):
        """
        Apply plunger voltages (same logic as current _apply_voltages).
        """
        # Create 2D grids centered around plunger voltages
        measurement = self.qdarts_config['measurement']
        resolution = measurement['resolution']
        
        x_grid = np.linspace(
            measurement['voltage_range']['min'][0], 
            measurement['voltage_range']['max'][0], 
            resolution
        )
        y_grid = np.linspace(
            measurement['voltage_range']['min'][1], 
            measurement['voltage_range']['max'][1], 
            resolution
        )
        
        X, Y = np.meshgrid(x_grid + plungers[0], y_grid + plungers[1])
        
        # Update voltage grids - create 3D array for compatibility
        if 'current_voltages' not in self.device_state:
            self.device_state["current_voltages"] = np.zeros((resolution, resolution, 6))  # 6 gates total
        
        self.device_state["current_voltages"][:,:,0] = X
        self.device_state["current_voltages"][:,:,1] = Y

    def _apply_barrier_voltages(self, barriers):
        """
        Apply barrier voltages and update tunnel couplings (based on qdarts_v5.py).
        """
        # Convert barriers to dictionary format expected by QDARTS
        barrier_voltages = {
            'barrier_2': barriers[0],
            'barrier_3': barriers[1], 
            'barrier_4': barriers[2]
        }
        
        # Update tunnel couplings in QDARTS experiment (exactly like qdarts_v5.py)
        if self.experiment is not None:
            self.experiment.update_tunnel_couplings(barrier_voltages)
        
        # Store current barrier voltages
        self.device_state["current_barrier_voltages"] = barriers

    def _get_reward(self):
        """
        Enhanced reward function considering both plunger alignment and barrier optimization.
        """
        # Plunger alignment reward (same as current)
        ground_truth_center = self.device_state["ground_truth_center"][:2]
        current_plunger_center = self._extract_voltage_centers(self.device_state["current_voltages"])
        plunger_distance = np.linalg.norm(ground_truth_center - current_plunger_center)
        
        # Barrier optimization reward (new)
        barrier_reward = self._calculate_barrier_reward()
        
        # Combined reward
        plunger_reward = max(self.max_possible_distance - plunger_distance, 0) * 0.1
        total_reward = plunger_reward + barrier_reward * 0.05  # Weight barrier reward less
        
        # Termination bonus
        if self._is_at_target():
            total_reward += 200.0
        
        return total_reward

    def _calculate_barrier_reward(self):
        """
        Calculate reward based on barrier optimization.
        """
        # Get current tunnel couplings
        if self.experiment is not None:
            try:
                current_couplings = self.experiment.get_current_tunnel_couplings()
            except:
                # If we can't get current couplings, return 0
                return 0.0
        else:
            return 0.0
        
        # Define target coupling strengths (could be configurable)
        target_couplings = np.array([
            [0, 30e-4, 0],
            [30e-4, 0, 0], 
            [0, 0, 0]
        ])
        
        # Calculate coupling alignment reward
        coupling_distance = np.linalg.norm(current_couplings - target_couplings)
        max_coupling_distance = np.linalg.norm(target_couplings)
        
        return max(max_coupling_distance - coupling_distance, 0)

    def _is_at_target(self):
        """
        Check if the current state is at the target.
        """
        ground_truth_center = self.device_state["ground_truth_center"][:2]
        current_plunger_center = self._extract_voltage_centers(self.device_state["current_voltages"])
        distance = np.linalg.norm(ground_truth_center - current_plunger_center)
        
        return distance < self.tolerance

    def _random_center(self):
        """
        Generate a random center for the voltage window.
        """
        # Generate random center within the voltage range
        measurement = self.qdarts_config['measurement']
        x_range = measurement['voltage_range']['max'][0] - measurement['voltage_range']['min'][0]
        y_range = measurement['voltage_range']['max'][1] - measurement['voltage_range']['min'][1]
        
        center_x = np.random.uniform(
            measurement['voltage_range']['min'][0] + x_range * 0.1,
            measurement['voltage_range']['max'][0] - x_range * 0.1
        )
        center_y = np.random.uniform(
            measurement['voltage_range']['min'][1] + y_range * 0.1,
            measurement['voltage_range']['max'][1] - y_range * 0.1
        )
        
        return np.array([center_x, center_y]) 

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

        # Initialize episode-specific voltage state
        # Center of current window
        center = self._random_center()

        # Create voltage grids centered around the random center
        measurement = self.qdarts_config['measurement']
        resolution = measurement['resolution']
        
        x_grid = np.linspace(
            measurement['voltage_range']['min'][0], 
            measurement['voltage_range']['max'][0], 
            resolution
        )
        y_grid = np.linspace(
            measurement['voltage_range']['min'][1], 
            measurement['voltage_range']['max'][1], 
            resolution
        )
        
        X, Y = np.meshgrid(x_grid + center[0], y_grid + center[1])
        
        # Initialize voltage state
        current_voltages = np.zeros((resolution, resolution, 6))  # 6 gates total
        current_voltages[:,:,0] = X
        current_voltages[:,:,1] = Y

        # Device state variables (episode-specific)
        self.device_state = {
            "experiment": self.experiment,
            "current_voltages": current_voltages,
            "ground_truth_center": center,
            "current_barrier_voltages": np.zeros(3)
        }

        # Calculate max possible distance for reward normalization
        self.max_possible_distance = np.sqrt(
            (measurement['voltage_range']['max'][0] - measurement['voltage_range']['min'][0])**2 +
            (measurement['voltage_range']['max'][1] - measurement['voltage_range']['min'][1])**2
        )

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
        reward = self._get_reward()

        # --- Check if the episode has ended ---
        terminated = self._is_at_target()
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
            }
        } 

    def render(self):
        """
        Render the environment.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
            return None

    def _render_frame(self, inference_plot=False):
        """
        Render a frame of the environment.
        """
        if self.z is None:
            return None
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot charge sensor data
        im1 = axes[0].imshow(self.z[:, :, 0], cmap='viridis', aspect='auto')
        axes[0].set_title('Charge Sensor Data')
        axes[0].set_xlabel('Gate 1 Voltage')
        axes[0].set_ylabel('Gate 2 Voltage')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot voltage centers
        if 'current_voltages' in self.device_state:
            current_voltages = self.device_state["current_voltages"]
            plunger_centers = self._extract_voltage_centers(current_voltages)
            ground_truth = self.device_state["ground_truth_center"]
            
            axes[1].scatter(plunger_centers[0], plunger_centers[1], c='red', s=100, label='Current Position')
            axes[1].scatter(ground_truth[0], ground_truth[1], c='green', s=100, label='Target Position')
            axes[1].set_xlabel('Gate 1 Voltage')
            axes[1].set_ylabel('Gate 2 Voltage')
            axes[1].set_title('Voltage Positions')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        
        # Convert to RGB array
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.savefig('test.png', format='png', dpi=100, bbox_inches='tight')

        buf.seek(0)
        img_array = plt.imread(buf)
        plt.close()
        
        return img_array

    def close(self):
        """
        Clean up resources.
        """
        if hasattr(self, 'experiment') and self.experiment is not None:
            # Clean up QDARTS experiment if needed
            pass

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

if __name__ == "__main__":
    env = QDartsEnv()
    env.reset()
    env.render()
    env.close()