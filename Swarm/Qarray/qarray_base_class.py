import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import os
from qarray import ChargeSensedDotArray, WhiteNoise, TelegraphNoise, LatchingModel
# Set matplotlib backend before importing pyplot to avoid GUI issues
import matplotlib
matplotlib.use('Agg')
import time
import matplotlib.pyplot as plt
import io
import fcntl
import json
from torch.nn.functional import sigmoid

# For block diagonal matrix construction
from scipy.linalg import block_diag

"""
Qarray base class with full randomisation

supports environment initialisation for both training and inference
"""

class QarrayBaseClass:
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, num_dots, config_path='qarray_config.yaml', randomise_actions=True, render_mode=None, counter_file=None, **kwargs):
        assert num_dots % 4 == 0, "Currently we only support multiples of 4 dots"
        print(f'Initialising qarray env with {num_dots} dots ...')

        # --- Load Configuration ---
        config_path = os.path.join(os.path.dirname(__file__), config_path)
        self.config = self._load_config(config_path)

        self.debug = self.config['init']['debug']
        self.seed = self.config['init']['seed']

        # these to be added to the main env.py file
        # self.max_steps = self.config['env']['max_steps']
        # self.current_step = 0
        # self.tolerance = self.config['env']['tolerance']

        optimal_center_dots = self.config['simulator']['measurement']['optimal_VG_center']['dots']
        optimal_center_sensor = self.config['simulator']['measurement']['optimal_VG_center']['sensor']
        optimal_VG_center = [optimal_center_dots] * num_dots + [optimal_center_sensor]

        # --- Define Action and Observation Spaces ---
        self.num_dots = num_dots
        self.num_gate_voltages = num_dots
        self.num_barrier_voltages = num_dots - 1
        
        self.obs_image_size = self.config['simulator']['measurement']['resolution']

        self._init_random_action_scaling()

        # --- Initialize Model (one-time setup) ---
        self.model = self._load_model()

        self.gate_ground_truth = self.model.optimal_Vg(optimal_VG_center) # only change if we update the model itself
        self.barrier_ground_truth = None # TODO

        # --- Initialize normalization parameters ---
        self._init_normalization_params()

    
    def _init_random_action_scaling(self):
        """
        internally scales the action after receiving it from the environment
        """
        if self.randomise_actions:
            self.action_scale_factor = []
            self.action_offset = []
            for _ in range(self.num_dots):
                self.action_scale_factor.append(np.random.uniform(self.config['env']['action_scale_factor']['min'], self.config['env']['action_scale_factor']['max']))
                action_offset_fraction = np.random.uniform(self.config['env']['action_offset_fraction']['min'], self.config['env']['action_offset_fraction']['max'])
                self.action_offset.append(action_offset_fraction * (self.obs_voltage_max - self.obs_voltage_min))
            self.action_scale_factor = np.array(self.action_scale_factor).astype(np.float32)
            self.action_offset = np.array(self.action_offset).astype(np.float32)
        else:
            self.action_scale_factor = np.ones(self.num_dots).astype(np.float32)
            self.action_offset = np.zeros(self.num_dots).astype(np.float32)


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
        self.data_max = 0.16
        self.bounds_initialized = False
        
        # Track statistics for adaptive updates
        self.episode_min = float('inf')
        self.episode_max = float('-inf')
        self.global_min = float('inf')
        self.global_max = float('-inf')
        self.update_count = 0
        
        if self.debug:
            print(f"Initial normalization range: [{self.data_min:.4f}, {self.data_max:.4f}]")


    def _update_normalization_bounds(self, raw_data):
        """
        Update normalization bounds based on new data encountered.
        Uses adaptive approach to gradually expand bounds while maintaining stability.
        
        Args:
            raw_data (np.ndarray): Raw charge sensor data
        """
        # Update episode statistics
        self.episode_min = min(self.episode_min, np.min(raw_data))
        self.episode_max = max(self.episode_max, np.max(raw_data))
        
        # Update global statistics
        self.global_min = min(self.global_min, np.min(raw_data))
        self.global_max = max(self.global_max, np.max(raw_data))
        
        # Check if we need to expand bounds
        needs_update = False
        new_min = self.data_min
        new_max = self.data_max
        
        # Expand lower bound if needed (with safety margin)
        if self.global_min < self.data_min:
            safety_margin = (self.data_max - self.data_min) * 0.05  # 5% safety margin
            new_min = self.global_min - safety_margin
            needs_update = True
            
        # Expand upper bound if needed (with safety margin)
        if self.global_max > self.data_max:
            safety_margin = (self.data_max - self.data_min) * 0.05  # 5% safety margin
            new_max = self.global_max + safety_margin
            needs_update = True
        
        # Update bounds if needed
        if needs_update:
            self.data_min = new_min
            self.data_max = new_max
            self.update_count += 1
            
            if self.debug:
                print(f"Updated normalization bounds to [{self.data_min:.4f}, {self.data_max:.4f}] (update #{self.update_count})")


    def _normalize_observation(self, raw_data):
        """
        Normalize the raw charge sensor data to the observation space range.
        Uses adaptive bounds that update as new data is encountered.
        
        Args:
            raw_data (np.ndarray): Raw charge sensor data of shape (height, width)
            
        Returns:
            np.ndarray: Normalized observation of shape (height, width, 1)
        """
        # Ensure input is 2D
        if raw_data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {raw_data.shape}")
        
        # Update bounds based on new data
        self._update_normalization_bounds(raw_data)
        
        # Normalize to [0, 1] range
        normalized = (raw_data - self.data_min) / (self.data_max - self.data_min)
        
        # Clip to ensure values are within bounds
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # Scale to uint8 range [0, 255]
        normalized = (normalized * 255).astype(np.uint8)
        
        # Reshape to add channel dimension
        normalized = normalized.reshape(normalized.shape[0], normalized.shape[1], 1)
        
        return normalized


    def _get_charge_sensor_data(self, voltages, gate1, gate2):
        """
        Get charge sensor data for given voltages.
        
        Args:
            voltages (np.ndarray): 2D voltage grid or voltage center configuration
            
        Returns:
            np.ndarray: Charge sensor data of shape (height, width, channels)
        """

        z, _ = self.device_state["model"].do2d_open(
            gate1, voltages[0]+self.obs_voltage_min, voltages[0]+self.obs_voltage_max, self.config['simulator']['measurement']['resolution'],
            gate2, voltages[1]+self.obs_voltage_min, voltages[1]+self.obs_voltage_max, self.config['simulator']['measurement']['resolution']
        )
        return z


    def _reset(self):
        pass

    
    # def reset(self, seed=None, options=None): # done
    #     """
    #     Resets the environment to an initial state and returns the initial observation.

    #     This method is called at the beginning of each new episode. It should
    #     reset the state of the environment and return the first observation that
    #     the agent will see.

    #     Args:
    #         seed (int, optional): Random seed for reproducibility.
    #         options (dict, optional): Additional options for reset.

    #     Returns:
    #         observation (np.ndarray): The initial observation of the space.
    #         info (dict): A dictionary with auxiliary diagnostic information.
    #     """

    #     self._increment_global_counter()

    #     # Handle seed if provided
    #     if seed is not None:
    #         super().reset(seed=seed)
    #     else:
    #         super().reset(seed=None)

    #     # --- Reset the environment's state ---
    #     self.current_step = 0
        
    #     # Reset episode-specific normalization statistics
    #     self.episode_min = float('inf')
    #     self.episode_max = float('-inf')

    #     # Initialize episode-specific voltage state
    #     #random actions scaling
    #     self._init_random_action_scaling()
    #     #center of current window
    #     center = self._random_center()

    #     optimal_VG_center = self.model.optimal_Vg(self.optimal_VG_center)

    #     # Device state variables (episode-specific)
    #     self.device_state = {
    #         "model": self.model,
    #         "gate_ground_truth": optimal_VG_center,
    #         "barrier_ground_truth": self.barrier_ground_truth,
    #         "current_gate_voltages": center["gates"],
    #         "current_barrier_voltages": center["barriers"],
    #     }


    #     # --- Return the initial observation ---
    #     observation = self._get_obs()
    #     info = self._get_info() 

    #     return observation, info
        

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

        # --- Update the environment's state based on the action ---
        self.current_step += 1
        # action is now a numpy array of shape (num_voltages,) containing voltage values

        # voltages, capacitances = action['action_voltages'], action['capacitances']
        gate_voltages = action['action_gate_voltages']
        barrier_voltages = action['action_barrier_voltages']

        if self.debug:
            print(f'Raw voltage outputs: {gate_voltages, barrier_voltages}')

        # apply random transformation
        gate_voltages = np.array(gate_voltages).flatten().astype(np.float32)
        gate_voltages = self.action_scale_factor * gate_voltages + self.action_offset

        barrier_voltages = np.array(barrier_voltages).flatten().astype(np.float32)

        if self.debug:
            print(f'Scaled voltage outputs: {gate_voltages}')

        self._apply_voltages({"gates": gate_voltages, "barriers": barrier_voltages}) #this step will update the voltages stored in self.device_state

        # --- Determine the reward ---
        reward = self._get_reward()  #will compare current state to target state
        if self.debug:
            print(f"reward: {reward}")

        # --- Check for termination or truncation conditions ---
        terminated = False
        truncated = False
        
        if self.current_step >= self.max_steps:
            truncated = True
            if self.debug:
                print("Max steps reached")
        
        # Check if the centers of the voltage sweeps are aligned
        ground_truth_center = self.device_state["gate_ground_truth"]

        # Get current voltage settings (what the agent controls)
        current_voltage_center = self.device_state["current_gate_voltages"]

        # Compare only the first num_voltages dimensions (ignoring last dimension)
        at_target = np.all(np.abs(ground_truth_center - current_voltage_center) <= self.tolerance)
        
        if at_target:
            terminated = True
            if self.debug:
                print("Target voltage sweep center reached")

        # --- Get the new observation and info ---
        observation = self._get_obs() #new state
        info = self._get_info() #diagnostic info
        
        return observation, reward, terminated, truncated, info
    
    def _random_center(self): # done
        """
        Randomly generate a center voltage for the voltage sweep.
        """
        gate_centers = np.random.uniform(self.gate_voltage_min-self.obs_voltage_min, self.gate_voltage_max-self.obs_voltage_max, self.num_gate_voltages)
        barrier_centers = np.random.uniform(self.barrier_voltage_min, self.barrier_voltage_max, self.num_barrier_voltages)
        return {
            "gates": gate_centers,
            "barriers": barrier_centers
        }

    def _get_reward(self, done=None):
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


        # if self.current_step == self.max_steps:
        #     # reward = max(max_possible_distance - distance, 0)*0.01
        #     reward = (1 - gate_distance / max_possible_distance) * 100
        # else:
        #     reward = 0.0

        # reward -= self.current_step * 0.1

        gate_rewards -= self.current_step * 0.1
        # we don't give time penalty to the barriers are they are not responsible for navigation


        at_target = np.all(np.abs(ground_truth_center - current_voltage_center) <= self.tolerance)
        if at_target:
            reward += 200.0

        if done is not None:
            prob = sigmoid(done)
            if prob > self.done_threshold:
                if at_target:
                    reward += 100.0
                else:
                    reward -= 10.0
            elif prob <= self.done_threshold:
                if at_target:
                    reward -= 100.0
        
        rewards = {
            "gates": gate_rewards,
            "barriers": barrier_rewards
        }

        return rewards


    def _get_obs(self):
        """
        Helper method to get the current observation of the environment.
        
        Returns a multi-modal observation with image and voltage data as numpy arrays.
        """
        # Get current voltage configuration
        # current_voltages = self.device_state["current_voltages"]
        voltage_centers = self.device_state["voltage_centers"]
        
        # Get charge sensor data
        # self.z = self._get_charge_sensor_data(current_voltages, gate1, gate2)
        allgates = list(range(1, self.num_dots+1))
        self.all_z = []
        for (gate1, gate2) in zip(allgates[:-1], allgates[1:]):
            z = self._get_charge_sensor_data(voltage_centers, gate1, gate2)
            self.all_z.append(z)


        all_images = []
        voltage_centers = self.device_state["voltage_centers"]

        expected_voltage_shape = (self.num_dots,)
        
        if voltage_centers.shape != expected_voltage_shape:
            raise ValueError(f"Voltage observation shape {voltage_centers.shape} does not match expected {expected_voltage_shape}")


        for z in self.all_z:
            # Extract first channel and normalize for image observation
            channel_data = z[:, :, 0]  # Shape: (height, width)
            image_obs = self._normalize_observation(channel_data)  # Shape: (height, width, 1)
            
            # Create multi-modal observation dictionary with numpy arrays 
            all_images.append(image_obs)

        all_images = np.concatenate(all_images, axis=-1)
        # all_images = all_images.squeeze(-1).transpose(1, 2, 0)
            
        # Validate observation structure
        expected_image_shape = (self.obs_image_size, self.obs_image_size, self.obs_channels)

        if all_images.shape != expected_image_shape:
            raise ValueError(f"Image observation shape {all_images.shape} does not match expected {expected_image_shape}")

        return {
            "image": all_images, # creates a multi-channel image with each adjacent pair of voltage sweeps
            "obs_gate_voltages": gate_voltages,
            "obs_barrier_voltages": barrier_voltages
        }
    
    # --- #


    def _gen_random_qarray_params(self, rng: np.random.Generator = None):
        """
        Generate random parameters for the quantum device.
        """
        model_config = self.config['simulator']['model']
        measurement_config = self.config['simulator']['measurement']

        if rng is None:
            rng = np.random.default_rng()

        rb = {
            'Cdd': model_config['Cdd'],
            'Cgd': model_config['Cgd'],
            'Cds': model_config['Cds'],
            'Cgs': model_config['Cgs'],
            'white_noise_amplitude': model_config['white_noise_amplitude'],
            'telegraph_noise_parameters': model_config['telegraph_noise_parameters'],
            'latching_model_parameters': model_config['latching_model_parameters'],
            'T': model_config['T'],
            'coulomb_peak_width': model_config['coulomb_peak_width'],
            'algorithm': model_config['algorithm'],
            'implementation': model_config['implementation'],
            'max_charge_carriers': model_config['max_charge_carriers'],
            'sensor_gate_voltage': measurement_config['sensor_gate_voltage'],
            'optimal_VG_center': measurement_config['optimal_VG_center']  # Add this for optimal voltage calculation
        }

        latching = True

        cdd_diag = rb["Cdd"]["diagonal"]
        cdd_nn = rng.uniform(rb["Cdd"]["nearest_neighbor"]["min"], 
                            rb["Cdd"]["nearest_neighbor"]["max"])
        cdd_next = rng.uniform(rb["Cdd"]["next_nearest"]["min"], 
                              rb["Cdd"]["next_nearest"]["max"])  
        cdd_far = rng.uniform(rb["Cdd"]["furthest"]["min"], 
                             rb["Cdd"]["furthest"]["max"])
        

        # Create the 4x4 Cdd block
        Cdd_block = [
            [cdd_diag, cdd_nn, cdd_next, cdd_far],
            [cdd_nn, cdd_diag, cdd_nn, cdd_next],
            [cdd_next, cdd_nn, cdd_diag, cdd_nn],
            [cdd_far, cdd_next, cdd_nn, cdd_diag]
        ]

        # Make Cdd a block diagonal matrix of shape (self.num_dots, self.num_dots)
        num_blocks = self.num_dots // 4
        Cdd = block_diag(*([Cdd_block] * num_blocks))


        # Create the 4x5 Cgd block
        Cgd_block = [[0.0 for _ in range(5)] for _ in range(4)]
        # Fill diagonal (primary couplings)
        for i in range(4):
            Cgd_block[i][i] = rng.uniform(rb["Cgd"][i][i]["min"], rb["Cgd"][i][i]["max"])
        # Fill symmetric cross-couplings for plunger gates (gates 0-3)
        for i in range(4):
            for j in range(4):
                if i < j:  # Only fill upper triangle, then mirror
                    coupling = rng.uniform(rb["Cgd"][i][j]["min"], rb["Cgd"][i][j]["max"])
                    Cgd_block[i][j] = coupling
                    Cgd_block[j][i] = coupling
        # Fill sensor gate couplings (gate 4)
        for i in range(4):
            Cgd_block[i][4] = rng.uniform(rb["Cgd"][i][4]["min"], rb["Cgd"][i][4]["max"])

        # Make Cgd a block diagonal matrix of shape (self.num_dots, self.num_dots+1)
        num_blocks = self.num_dots // 4
        # block_diag pads with zeros to the right/bottom if blocks are not square, so we need to trim the result
        Cgd_full = block_diag(*([Cgd_block] * num_blocks))
        # Cgd_full will be (self.num_dots, num_blocks*5), but we want (self.num_dots, self.num_dots+1)
        Cgd = Cgd_full[:, :self.num_dots+1]

        Cds = [[rng.uniform(rb["Cds"]["dots"]["min"], rb["Cds"]["dots"]["max"]) for i in range(self.num_dots)]]
        Cgs = [[rng.uniform(rb["Cgs"]["dots"]["min"], rb["Cgs"]["dots"]["max"]) for i in range(self.num_dots)] + [rng.uniform(rb["Cgs"]["sensor"]["min"], rb["Cgs"]["sensor"]["max"])]]

        # Generate 4x4 p_inter matrix for latching model (must be symmetric)
        p_inter = [[0.0 for _ in range(4)] for _ in range(4)]  # Initialize with zeros
        
        # Fill upper triangle and mirror to lower triangle for symmetry
        for i in range(4):
            for j in range(i+1, 4):  # Only fill upper triangle
                coupling = rng.uniform(rb["latching_model_parameters"]["p_inter"]["min"],
                                     rb["latching_model_parameters"]["p_inter"]["max"])
                p_inter[i][j] = coupling
                p_inter[j][i] = coupling  # Ensure symmetry
        # Diagonal elements remain 0.0 (no self-interaction)
        
        # Generate 4-element p_leads array
        p_leads = [rng.uniform(rb["latching_model_parameters"]["p_leads"]["min"],
                              rb["latching_model_parameters"]["p_leads"]["max"]) 
                   for _ in range(4)]
        
        p01 = rng.uniform(rb["telegraph_noise_parameters"]["p01"]["min"], 
                         rb["telegraph_noise_parameters"]["p01"]["max"])
        

        # # Generate fixed gate voltages for outer gates (these will be replaced by optimal voltages later)
        # fixed_gate_voltages = {}
        # for gate_idx in rb["fixed_gates"]:
        #     if gate_idx == 4:  # Sensor gate (will be replaced with optimal)
        #         fixed_gate_voltages[gate_idx] = rng.uniform(
        #             rb["sensor_gate_voltage"]["min"], 
        #             rb["sensor_gate_voltage"]["max"]
        #         )
        #     else:  # Plunger gates (will be replaced with optimal)
        #         fixed_gate_voltages[gate_idx] = rng.uniform(
        #             rb["fixed_gate_voltages"]["min"], 
        #             rb["fixed_gate_voltages"]["max"]
        #         )
        

        model_params = {
            "Cdd": Cdd,
            "Cgd": Cgd,
            "Cds": Cds,
            "Cgs": Cgs,
            "white_noise_amplitude": rng.uniform(rb["white_noise_amplitude"]["min"], 
                                               rb["white_noise_amplitude"]["max"]),
            "telegraph_noise_parameters": {
                "p01": p01,
                "p10": rng.uniform(rb["telegraph_noise_parameters"]["p10_factor"]["min"], 
                                 rb["telegraph_noise_parameters"]["p10_factor"]["max"]) * p01,
                "amplitude": rng.uniform(rb["telegraph_noise_parameters"]["amplitude"]["min"], 
                                       rb["telegraph_noise_parameters"]["amplitude"]["max"]),
            },
            "latching_model_parameters": {
                "Exists": latching,
                "n_dots": 4,
                "p_leads": p_leads,
                "p_inter": p_inter,
            },
            "T": rng.uniform(rb["T"]["min"], rb["T"]["max"]),
            "coulomb_peak_width": rng.uniform(rb["coulomb_peak_width"]["min"], 
                                            rb["coulomb_peak_width"]["max"]),
            "algorithm": rb["algorithm"],
            "implementation": rb["implementation"],
            "max_charge_carriers": rb["max_charge_carriers"],
            "optimal_VG_center": rb["optimal_VG_center"]
        }
        
        return model_params


    def _load_model(self):
        """
        Load the model from the config file.
        """

        model_params = self._gen_random_qarray_params()

        white_noise = WhiteNoise(amplitude=model_params['white_noise_amplitude'])
        telegraph_noise = TelegraphNoise(**model_params['telegraph_noise_parameters'])
        noise_model = white_noise + telegraph_noise
        latching_params = model_params['latching_model_parameters']
        latching_model = LatchingModel(**{k: v for k, v in latching_params.items() if k != "Exists"}) if latching_params["Exists"] else None

        Cdd_base = model_params['Cdd']
        Cgd_base = model_params['Cgd']
        Cds_base = model_params['Cds']
        Cgs_base = model_params['Cgs']

        model_mats = []
        for mat in [Cdd_base, Cgd_base]:
            block_size = np.array(mat).shape[0]
            num_blocks = self.num_dots // block_size
            out_mat = block_diag(*([mat]*num_blocks))
            model_mats.append(out_mat)

        Cdd, Cgd = model_mats
        Cds = [np.array(Cds_base).flatten().tolist() * (self.num_dots // 4)]
        Cgs = [np.array(Cgs_base).flatten().tolist() * (self.num_dots // 4)]

        # print(np.array(Cdd).shape)
        # print(np.array(Cgd).shape)
        # print(np.array(Cds).shape)
        # print(np.array(Cgs).shape)

        model = ChargeSensedDotArray(
            Cdd=Cdd,
            Cgd=Cgd,
            Cds=Cds,
            Cgs=Cgs,
            coulomb_peak_width=model_params['coulomb_peak_width'],
            T=model_params['T'],
            noise_model=noise_model,
            latching_model=latching_model,
            algorithm=model_params['algorithm'],
            implementation=model_params['implementation'],
            max_charge_carriers=model_params['max_charge_carriers'],
        )

        # model.gate_voltage_composer.virtual_gate_matrix = self.config['simulator']['virtual_gate_matrix']
        # TODO update virtual gate matrix

        return model


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
                "image_shape": (self.obs_image_size, self.obs_image_size, self.obs_channels),
                "voltage_shape": (self.num_dots,),
                "total_modalities": 2
            }
        }

    def _apply_voltages(self, voltages): # done
        """
        Apply voltage settings to the quantum device.
        
        Args:
            voltages (np.ndarray): Array of voltage values for each gate
        """
        gate_voltages = voltages["gates"]
        barrier_voltages = voltages["barriers"]

        assert len(gate_voltages) == self.num_gate_voltages, f"Expected voltages to be of size {self.num_gate_voltages}, got {len(gate_voltages)}"
        assert len(barrier_voltages) == self.num_barrier_voltages, f"Expected voltages to be of size {self.num_barrier_voltages}, got {len(barrier_voltages)}"

        self.device_state["current_voltage_center"] = np.clip(gate_voltages, self.gate_voltage_min, self.gate_voltage_max)
        self.device_state["current_barrier_voltages"] = np.clip(barrier_voltages, self.barrier_voltage_min, self.barrier_voltage_max)
         

    def render(self):
        """
        Render the environment state.
        
        Returns:
            np.ndarray: RGB array representation of the environment state
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            # For human mode, save the plot and return None
            self._render_frame()
            return None
        else:
            return None


    def _render_frame(self, inference_plot=False):
        """
        Internal method to create the render image.

        We plot the CSD between gate1 and its successor
        
        Returns:
            np.ndarray: RGB array representation of the environment state
        """
        z = self.z

        if inference_plot:
            vmin, vmax = (self.obs_voltage_min, self.obs_voltage_max)

            plt.figure(figsize=(5, 5))
            plt.imshow(z, extent=[vmin, vmax, vmin, vmax], origin='lower', aspect='auto', cmap='viridis')
            plt.xlabel('$Vx$')
            plt.ylabel('$Vy$')
            plt.title('$z$')
            plt.axis('equal')
            #plt.savefig('test_image.png')
            #plt.close()
            #return None

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plt.close()

            from PIL import Image
            img = Image.open(buf)
            img = np.array(img)
            return img

        # Get the normalized observation that the agent sees
        channel_data = z[:, :, 0]  # Shape: (height, width)
        normalized_obs = self._normalize_observation(channel_data)  # Shape: (height, width, 1)
        normalized_data = normalized_obs[:, :, 0]  # Remove channel dimension for plotting
        
        # Create figure and plot
        vmin, vmax = (self.obs_voltage_min, self.obs_voltage_max)
        num_ticks = 5
        tick_values = np.linspace(vmin, vmax, num_ticks)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(normalized_data, cmap='viridis', aspect='auto', vmin=0.0, vmax=1.0)
        
        # Set x and y axis ticks to correspond to voltage range
        ax.set_xticks(np.linspace(0, z.shape[1]-1, num_ticks))
        ax.set_xticklabels([f'{v:.0f}' for v in tick_values])
        ax.set_yticks(np.linspace(0, z.shape[0]-1, num_ticks))
        ax.set_yticklabels([f'{v:.0f}' for v in tick_values])
        
        ax.set_xlabel("$\Delta$PL (V)")
        ax.set_ylabel("$\Delta$PR (V)")
        ax.set_title("Normalized $|S_{11}|$ (Agent Observation)")
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
        cbar.set_ticklabels([f"{tick:.2f}" for tick in cbar.get_ticks()])


        if self.render_mode == "human":
            # Save plot for human mode
            script_dir = os.path.dirname(os.path.abspath(__file__))
            plot_path = os.path.join(script_dir, 'quantum_dot_plot.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Plot saved as '{plot_path}'")
            return None
        else:
            # Convert to RGB array for rgb_array mode
            fig.canvas.draw()
            
            # Simple approach: save to bytes and load as image
            try:
                # Save to bytes buffer
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                
                # Load as image using PIL
                from PIL import Image
                img = Image.open(buf)
                data = np.array(img)
                
                # Convert RGBA to RGB if needed
                if data.shape[-1] == 4:
                    data = data[:, :, :3]
                    
            except Exception as e:
                print(f"Error getting RGB data from canvas: {e}")
                # Last resort: create a simple colored array
                height, width = normalized_data.shape
                data = np.zeros((height, width, 3), dtype=np.uint8)
                # Create a simple visualization based on the normalized data
                data[:, :, 0] = (normalized_data * 255).astype(np.uint8)  # Red channel
                data[:, :, 1] = (normalized_data * 255).astype(np.uint8)  # Green channel
                data[:, :, 2] = (normalized_data * 255).astype(np.uint8)  # Blue channel
            
            plt.close()
            return data
 


    def close(self):
        """
        Performs any necessary cleanup.
        """
  
        pass
    

    def _load_config(self, config_path):
        """
        Load configuration from YAML file.
        
        Args:
            config_path (str): Path to the YAML configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        return config


if __name__ == "__main__":
    import sys
    env = QarrayBaseClass(num_dots=4, randomise_actions=False)
    env.reset()

    sys.exit(0)

    gt = env.model.optimal_Vg(env.optimal_VG_center)
    print(gt)
    print(env.device_state['voltage_centers'])
    env.step(gt[1:3])
    print(env.device_state['voltage_centers'])
    frame = env._render_frame(inference_plot=True)
    path = "quantum_dot_plot.png"
    plt.imsave(path, frame, cmap='viridis')
    env.close()