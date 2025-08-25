import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import os
from qarray_base_class import QarrayBaseClass
# Set matplotlib backend before importing pyplot to avoid GUI issues
import matplotlib
matplotlib.use('Agg')

"""
todo:

"""


class QuantumDeviceEnv(gym.Env):
    """
    Simulator environment that handles all env related logic
    loads in the qarray/ device model to extract observations
    """
    def __init__(self, training=True, config_path="env_config.yaml"):
        super().__init__()

        self.config = self._load_config(config_path)
        self.training = training # if we are training or not

        self.num_dots = self.config['num_dots']
        self.array = QarrayBaseClass(num_dots=self.num_dots)


        # --- capacitance model ---
        self.capacitance_model = None

        # --- environment parameters ---
        self.max_steps = self.config['simulator']['max_steps']
        self.tolerance = self.config['simulator']['tolerance']
        self.current_step = 0

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

        self.initial_virtual_gate_matrix = np.eye((self.num_dots, self.num_dots), dtype=np.float32)

        self.reset()


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

        if seed is not None:
            super().reset(seed=seed)
        else:
            super().reset(seed=None)

        # --- Reset the environment's state ---
        self.current_step = 0

        # --- Get random initial scaling and position ---
        self._init_random_action_scaling()
        center = self._random_center()

        # need to recompute the ground truths if we re-randomise qarray params
        optimal_vg_center = self.array.model.optimal_Vg(self.array.optimal_VG_center)
        barrier_ground_truth = self._compute_barrier_ground_truth()


        self.device_state = {
            "gate_ground_truth": optimal_vg_center,
            "barrier_ground_truth": barrier_ground_truth,
            "current_gate_voltages": center["gates"],
            "current_barrier_voltages": center["barriers"],
            "virtual_gate_matrix": self.initial_virtual_gate_matrix.copy()
        }

        # --- Return the initial observation ---
        raw_observation = self.array._get_obs(self.device_state["current_gate_voltages"], self.device_state["current_barrier_voltages"])
        observation = self._normalise_obs(raw_observation)

        info = self._get_info()

        return observation, info

    
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


        reward, at_target = self._get_reward()
        terminated = truncated = False

        if self.current_step >= self.max_steps:
            truncated = True
            if self.debug:
                print("Max steps reached")

        if np.all(at_target):
            terminated = True
            if self.debug:
                print("Target reached!")

        raw_observation = self.array._get_obs(gate_voltages, barrier_voltages)
        observation = self._normalise_obs(raw_observation)

        info = self._get_info()

        return observation, reward, terminated, truncated, info # note we are returning reward as a dict of lists (one reward per agent)


    def _get_reward(self):
        """
        Get the reward for the current state.

        Reward is based on the distance from the target voltage sweep center, with maximum reward
        when the agent aligns the centers of the voltage sweeps. The reward is calculated
        as: max_possible_distance - current_distance, where max_possible_distance is the maximum
        possible distance in the along the 1D voltage axis.

        A separate reward is given to each gate and barrier agent.

        The reward is also penalized by the number of steps taken to encourage efficiency.
        """

        gate_ground_truth = self.device_state["gate_ground_truth"]
        current_gate_voltages = self.device_state["current_gate_voltages"]
        gate_distances = np.linalg.norm(gate_ground_truth - current_gate_voltages)

        barrier_ground_truth = self.device_state["barrier_ground_truth"]
        current_barrier_voltages = self.device_state["current_barrier_voltages"]
        barrier_distances = np.linalg.norm(barrier_ground_truth - current_barrier_voltages)

        max_gate_distance = self.action_voltage_max - self.action_voltage_min # only gives reward when gt is visible
        max_barrier_distance = self.barrier_voltage_max - self.barrier_voltage_min # always gives reward


        if self.current_step == self.max_steps:
            gate_rewards = (1 - gate_distances / max_gate_distance) * 100
            barrier_rewards = (1 - barrier_distances / max_barrier_distance) * 100
        else:
            gate_rewards = np.zeros_like(gate_distances)
            barrier_rewards = np.zeros_like(barrier_distances)

        
        gate_rewards -= self.current_step * 0.1
        # we don't give time penalty to the barriers are they are not responsible for navigation

        at_target = np.abs(gate_ground_truth - current_gate_voltages) <= self.tolerance

        gate_rewards[at_target] += 200.0


        rewards = {
            "gates": gate_rewards,
            "barriers": barrier_rewards
        }

        return rewards, at_target


    def _get_info(self):
        return {}


    def _normalise_obs(self, obs):
        """
        Normalize observations from 0 to 1 based on the middle 99% of data.
        Clips the outer 0.5% to 0 and 1 on either end.
        
        Args:
            obs (dict): Observation dictionary containing 'image' and voltage data
            
        Returns:
            dict: Normalized observation dictionary
        """
        assert 'image' in obs, "Image data is required for normalization"

        normalized_obs = obs.copy()
        
        # Normalize the image data
        if 'image' in obs:
            image_data = obs['image']
            
            # Calculate percentiles for the middle 99% of data
            p_low = np.percentile(image_data, 0.5)   # 0.5th percentile
            p_high = np.percentile(image_data, 99.5) # 99.5th percentile
            
            # Normalize to [0, 1] based on middle 99% range
            if p_high > p_low:
                normalized_image = (image_data - p_low) / (p_high - p_low)
            else:
                # Handle edge case where all values are the same
                normalized_image = np.zeros_like(image_data)
            
            # Clip to [0, 1] range (this clips the outer 0.5% on each end)
            normalized_image = np.clip(normalized_image, 0.0, 1.0)
            
            # Convert to uint8 range [0, 255] for consistency with observation space
            normalized_obs['image'] = (normalized_image * 255).astype(np.uint8)
        
        return normalized_obs

    
    def _update_virtual_gate_matrix(self, obs):
        image = obs["image"]

        ### update here

        vgm = None
        self.array._update_virtual_gate_matrix(vgm)


    def _compute_barrier_ground_truth(self):
        pass

    
    def _init_random_action_scaling(self):
        """
        Initialize random scaling and offset for gate voltages.
        Each gate voltage dimension gets:
        - A random scale factor near 1.0 (e.g., 0.8 to 1.2)
        - A random offset near 0.0 (e.g., -0.1 to 0.1)
        """
        if self.training:
            # Random scale factors near 1.0 (between 0.8 and 1.2)
            self.action_scale_factor = np.random.uniform(0.8, 1.2, self.num_gate_voltages).astype(np.float32)
            
            # Random offsets near 0.0 (between -0.1 and 0.1)
            self.action_offset = np.random.uniform(-0.1, 0.1, self.num_gate_voltages).astype(np.float32)
        else:
            # No scaling during inference
            self.action_scale_factor = np.ones(self.num_gate_voltages, dtype=np.float32)
            self.action_offset = np.zeros(self.num_gate_voltages, dtype=np.float32)


    def _random_center(self):
        """
        Randomly generate a center voltage for the voltage sweep.
        """
        gate_centers = np.random.uniform(self.gate_voltage_min-self.obs_voltage_min, self.gate_voltage_max-self.obs_voltage_max, self.num_gate_voltages)
        barrier_centers = np.random.uniform(self.barrier_voltage_min, self.barrier_voltage_max, self.num_barrier_voltages)
        return {
            "gates": gate_centers,
            "barriers": barrier_centers
        }


    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        return config


    def _cleanup(self):
        pass