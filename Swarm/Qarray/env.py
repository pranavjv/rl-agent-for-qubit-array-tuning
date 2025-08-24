import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import os
from qarray_base_class import QarrayBaseClass
# Set matplotlib backend before importing pyplot to avoid GUI issues
import matplotlib
matplotlib.use('Agg')


class QuantumDeviceEnv(gym.Env):
    """
    Simulator environment that handles all env related logic
    loads in the qarray base model to extract observations
    """
    def __init__(self, training=True, config_path="env_config.yaml"):
        super().__init__()

        self.config = self._load_config(config_path)
        self.training = training # if we are training or not

        self.num_dots = self.config['num_dots']
        self.qarray = QarrayBaseClass(num_dots=self.num_dots)

        self.capacitance_model = None # to add


        # --- environment parameters ---
        self.max_steps = self.config['simulator']['max_steps']
        self.tolerance = self.config['simulator']['tolerance']
        self.current_step = 0

        optimal_center_dots = self.config['simulator']['measurement']['optimal_VG_center']['dots']
        optimal_center_sensor = self.config['simulator']['measurement']['optimal_VG_center']['sensor']
        self.optimal_VG_center = [optimal_center_dots] * num_dots + [optimal_center_sensor]
        self.barrier_ground_truth = None #TODO

        self.num_gate_voltages = num_dots
        self.num_barrier_voltages = num_dots - 1
        self.gate_voltage_min = self.config['simulator']['measurement']['gate_voltage_range']['min']
        self.gate_voltage_max = self.config['simulator']['measurement']['gate_voltage_range']['max']
        self.barrier_voltage_min = self.config['simulator']['measurement']['barrier_voltage_range']['min']
        self.barrier_voltage_max = self.config['simulator']['measurement']['barrier_voltage_range']['max']

        self.action_space = spaces.Dict({
            'action_gate_voltages': spaces.Box(
                low=self.gate_voltage_min,
                high=self.gate_voltage_max,
                shape=(self.num_gate_voltages,),
                dtype=np.float32
            ),
            'action_barrier_voltages': spaces.Box(
                low=self.barrier_voltage_min,
                high=self.barrier_voltage_max,
                shape=(self.num_barrier_voltages,),
                dtype=np.float32
            ),
        })

        # obs voltage min/max define the range over which we sweep the 2d csd pairs
        self.obs_voltage_min = self.config['simulator']['measurement']['gate_voltage_sweep_range']['min']
        self.obs_voltage_max = self.config['simulator']['measurement']['gate_voltage_sweep_range']['max']

        self.obs_channels = self.num_dots - 1
        self.obs_normalization_range = [0., 1.]

        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(self.obs_image_size, self.obs_image_size, self.obs_channels),
                dtype=np.uint8
            ),
            'obs_gate_voltages': spaces.Box(
                low=self.gate_voltage_min,
                high=self.gate_voltage_max,
                shape=(self.num_gate_voltages,),
                dtype=np.float32
            ),
            'obs_barrier_voltages': spaces.Box(
                low=self.barrier_voltage_min,
                high=self.barrier_voltage_max,
                shape=(self.num_barrier_voltages,),
                dtype=np.float32
            )
        })

        self.initial_virtual_gate_matrix = np.zeros((self.num_dots, self.num_dots), dtype=np.float32)


    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns the initial observation.

        This method is called at the beginning of each new episode. It should
        reset the state of the environment and return the first observation that
        the agent will see.

        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Additional options for reset.

        Returns:
            observation (np.ndarray): The initial observation of the space.
            info (dict): A dictionary with auxiliary diagnostic information.
        """

        self._increment_global_counter()

        # Handle seed if provided
        if seed is not None:
            super().reset(seed=seed)
        else:
            super().reset(seed=None)

        # --- Reset the environment's state ---
        self.current_step = 0
        
        # Reset episode-specific normalization statistics
        self.episode_min = float('inf')
        self.episode_max = float('-inf')

        # Initialize episode-specific voltage state
        #random actions scaling
        self.qarray._init_random_action_scaling()
        #center of current window
        center = self.qarray._random_center()

        # Device state variables (episode-specific)
        self.device_state = {
            "current_gate_voltages": center["gates"],
            "current_barrier_voltages": center["barriers"],
            "virtual_gate_matrix": self.initial_virtual_gate_matrix.copy()
        }


        # --- Return the initial observation ---
        observation = self.qarray._get_obs()
        info = self.qarray._get_info()

        # --- reset the base class ---
        self.qarray._reset()

        return observation, info







    
    def _update_virtual_gate_matrix(self):
        pass

    
    
    def step(self, action):
        """
        Updates the environment state based on the agent's action.

        This method is the core of the environment. It takes an action from the
        agent and calculates the next state, the reward, and whether the
        episode has ended.

        Args:
            action: An action provided by the agent.

        Returns:
            observation (np.ndarray): The observation of the environment's state.
            reward (float): The amount of reward returned after previous action.
            terminated (bool): Whether the episode has ended (e.g., reached a goal).
            truncated (bool): Whether the episode was cut short (e.g., time limit).
            info (dict): A dictionary with auxiliary diagnostic information.
        """
         self.current_step += 1
        # action is now a numpy array of shape (num_voltages,) containing voltage values

        # voltages, capacitances = action['action_voltages'], action['capacitances']
        gate_voltages = action['action_gate_voltages']
        barrier_voltages = action['action_barrier_voltages']

        if self.debug:
            print(f'Raw voltage outputs: {gate_voltages, barrier_voltages}')

        gate_voltages = np.array(gate_voltages).flatten().astype(np.float32)
        barrier_voltages = np.array(barrier_voltages).flatten().astype(np.float32)

        # apply random transformation (if we are training)
        if self.training:
            gate_voltages = self.action_scale_factor * gate_voltages + self.action_offset

        self.device_state["current_gate_voltages"] = gate_voltages
        self.device_state["current_barrier_voltages"] = barrier_voltages


        reward, at_target = self.qarray._get_reward(gate_voltages, barrier_voltages)
        terminated = truncated = False

        if self.current_step >= self.max_steps:
            truncated = True
            if self.debug:
                print("Max steps reached")

        if at_target:
            terminated = True
            if self.debug:
                print("Target reached!")

        observation = self.qarray._get_obs(gate_voltages, barrier_voltages)
        info = self.qarray._get_info()

        return observation, reward, terminated, truncated, info


    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        return config