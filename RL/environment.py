import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import os
from qarray import ChargeSensedDotArray, WhiteNoise, TelegraphNoise, LatchingModel
import matplotlib.pyplot as plt

class QuantumDeviceEnv(gym.Env):
    """
    Represents the device with its quantum dots 
    """
    #rendering info
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, config_path='env_config.yaml', render_mode=None):
        """
        constructor for the environment

        define action and observation spaces

        init state and variables
        """
        super().__init__()

        # --- Load Configuration ---
        self.config = self._load_config(config_path)

        self.debug = self.config['training']['debug']
        self.seed = self.config['training']['seed']
        self.max_steps = self.config['env']['max_steps']
        self.current_step = 0


        # --- Define Action and Observation Spaces ---
        self.num_voltages = self.config['env']['action_space']['num_voltages']  # Default to 2 gate voltages and 3 barrier voltages
        self.voltage_min = self.config['env']['action_space']['voltage_range'][0]  # Minimum voltage need to confirm what this should be physical
        self.voltage_max = self.config['env']['action_space']['voltage_range'][1]   # Maximum voltage need to confirm what this should be physical
        
        self.action_space = spaces.Box(
            low=self.voltage_min, 
            high=self.voltage_max, 
            shape=(self.num_voltages,), 
            dtype=np.float32
        )

        # Observation space for quantum device state - 1-channel 128x128 image
        obs_config = self.config['env']['observation_space']
        self.obs_image_size = obs_config['image_size']
        self.obs_channels = obs_config['channels']
        self.obs_normalization_range = obs_config['normalization_range']
        self.obs_dtype = obs_config['dtype']
        
        self.observation_space = spaces.Box(
            low=self.obs_normalization_range[0],
            high=self.obs_normalization_range[1],
            shape=(self.obs_image_size[0], self.obs_image_size[1], self.obs_channels),
            dtype=np.float32
        )

        # --- Initialize Model (one-time setup) ---
        self.model = self._load_model()
        
        # --- Initialize normalization parameters ---
        self._init_normalization_params()

        # --- For Rendering --- 
        self.render_fps = self.config['training']['render_fps'] #unused for now
        self.render_mode = render_mode or self.config['training']['render_mode']

    def _init_normalization_params(self):
        """
        Initialize normalization parameters for consistent scaling across episodes.
        This method samples a few voltage configurations to estimate the data range.
        Crude way of doing it but solid for now, currently outputs roughly 0.0862 -> 0.2376+-0.001
        """
        if self.debug:
            print("Initializing normalization parameters...")
        
        # Sample a few voltage center configurations to estimate data range
        sample_centers = [
            [0.0, 0.0, 0.0],  # Center
            [self.voltage_min, self.voltage_min, self.voltage_min],  # Min
            [self.voltage_max, self.voltage_max, self.voltage_max],  # Max
        ]
        
        min_val = float('inf')
        max_val = float('-inf')
        
        for center in sample_centers:
            try:
                # Create 2D voltage grid with this center
                vg = self.model.gate_voltage_composer.do2d(
                    "vP1", self.config['simulator']['measurement']['vx_min'], self.config['simulator']['measurement']['vx_max'], self.config['simulator']['measurement']['resolution'],
                    "vP2", self.config['simulator']['measurement']['vy_min'], self.config['simulator']['measurement']['vy_max'], self.config['simulator']['measurement']['resolution']
                )
                vg_with_offset = vg + self.model.optimal_Vg(center)
                
                z, _ = self.model.charge_sensor_open(vg_with_offset)
                channel_data = z[:, :, 0]  # First channel
                min_val = min(min_val, np.min(channel_data))
                max_val = max(max_val, np.max(channel_data))
            except Exception as e:
                if self.debug:
                    print(f"Warning: Could not sample center {center}: {e}")
        
        # Set normalization parameters with some safety margin
        if min_val != float('inf') and max_val != float('-inf'):
            self.data_min = min_val
            self.data_max = max_val
            # Add small margin to avoid edge cases
            margin = (max_val - min_val) * 0.01
            self.data_min -= margin
            self.data_max += margin
        else:
            # Fallback values if sampling fails
            self.data_min = -1.0
            self.data_max = 1.0
            
        if self.debug:
            print(f"Normalization range: [{self.data_min:.4f}, {self.data_max:.4f}]")

    def _normalize_observation(self, raw_data):
        """
        Normalize the raw charge sensor data to the observation space range.
        
        Args:
            raw_data (np.ndarray): Raw charge sensor data of shape (height, width)
            
        Returns:
            np.ndarray: Normalized observation of shape (height, width, 1)
        """
        # Ensure input is 2D
        if raw_data.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {raw_data.shape}")
        
        # Normalize to [0, 1] range
        normalized = (raw_data - self.data_min) / (self.data_max - self.data_min)
        
        # Clip to ensure values are within bounds
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # Reshape to add channel dimension
        normalized = normalized.reshape(normalized.shape[0], normalized.shape[1], 1)
        
        # Ensure correct dtype
        normalized = normalized.astype(np.float32)
        
        return normalized

    def _get_charge_sensor_data(self, voltages):
        """
        Get charge sensor data for given voltages.
        
        Args:
            voltages (np.ndarray): 2D voltage grid or voltage center configuration
            
        Returns:
            np.ndarray: Charge sensor data of shape (height, width, channels)
        """
        z, _ = self.device_state["model"].charge_sensor_open(voltages)
        return z

    def get_raw_observation(self):
        """
        Get raw (unnormalized) observation data for debugging purposes.
        
        Returns:
            np.ndarray: Raw charge sensor data of shape (height, width)
        """
        current_voltages = self.device_state["current_voltages"]
        z = self._get_charge_sensor_data(current_voltages)
        return z[:, :, 0]

    def reset(self):
        """
        Resets the environment to an initial state and returns the initial observation.

        This method is called at the beginning of each new episode. It should
        reset the state of the environment and return the first observation that
        the agent will see.

        Returns:
            observation (np.ndarray): The initial observation of the space.
            info (dict): A dictionary with auxiliary diagnostic information.
        """
        #seed the random number generator
        super().reset(seed=self.seed)

        # --- Reset the environment's state ---
        self.current_step = 0
        
        # Initialize episode-specific voltage state
        vg = self.model.gate_voltage_composer.do2d(
            "vP1", self.config['simulator']['measurement']['vx_min'], self.config['simulator']['measurement']['vx_max'], self.config['simulator']['measurement']['resolution'],
            "vP2", self.config['simulator']['measurement']['vy_min'], self.config['simulator']['measurement']['vy_max'], self.config['simulator']['measurement']['resolution']
        )
        
        vg_ground_truth = vg + self.model.optimal_Vg(self.config['simulator']['measurement']['optimal_VG_center'])
        
        # Device state variables (episode-specific)
        self.device_state = {
            "model": self.model,
            "current_voltages": vg,
            "ground_truth_voltages": vg_ground_truth,
        }

        # --- Return the initial observation ---
        observation = self._get_obs() 
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

        # --- Update the environment's state based on the action ---
        self.current_step += 1
        # action is now a numpy array of shape (num_voltages,) containing voltage values

        self._apply_voltages(action) #this step will update the qarray parameters stored in self.device_state

        # --- Determine the reward ---
        reward = 0.0  #will compare current state to target state
        
        # --- Check for termination or truncation conditions ---
        terminated = False
        truncated = False
        
        if self.current_step >= self.max_steps:
            truncated = True


        # --- Get the new observation and info ---
        observation = self._get_obs() #new state
        info = self._get_info() #diagnostic info
        
        return observation, reward, terminated, truncated, info

    def _load_model(self):
        """
        Load the model from the config file.
        """
        white_noise = WhiteNoise(amplitude=self.config['simulator']['model']['white_noise_amplitude'])
        telegraph_noise = TelegraphNoise(**self.config['simulator']['model']['telegraph_noise_parameters'])
        noise_model = white_noise + telegraph_noise
        latching_params = self.config['simulator']['model']['latching_model_parameters']
        latching_model = LatchingModel(**{k: v for k, v in latching_params.items() if k != "Exists"}) if latching_params["Exists"] else None


        model = ChargeSensedDotArray(
            Cdd=self.config['simulator']['model']['Cdd'],
            Cgd=self.config['simulator']['model']['Cgd'],
            Cds=self.config['simulator']['model']['Cds'],
            Cgs=self.config['simulator']['model']['Cgs'],
            coulomb_peak_width=self.config['simulator']['model']['coulomb_peak_width'],
            T=self.config['simulator']['model']['T'],
            noise_model=noise_model,
            latching_model=latching_model,
            algorithm=self.config['simulator']['model']['algorithm'],
            implementation=self.config['simulator']['model']['implementation'],
            max_charge_carriers=self.config['simulator']['model']['max_charge_carriers']
        )
        
        model.gate_voltage_composer.virtual_gate_matrix = self.config['simulator']['virtual_gate_matrix']


        return model

    def _get_obs(self):
        """
        Helper method to get the current observation of the environment.
        
        Returns a normalized 1-channel image observation that conforms to self.observation_space.
        """
        # Get current voltage configuration
        current_voltages = self.device_state["current_voltages"]
        
        # Get charge sensor data
        z = self._get_charge_sensor_data(current_voltages)
        
        # Extract first channel and normalize
        channel_data = z[:, :, 0]  # Shape: (height, width)
        observation = self._normalize_observation(channel_data)  # Shape: (height, width, 1)
        
        # Validate observation shape
        expected_shape = (self.obs_image_size[0], self.obs_image_size[1], self.obs_channels)
        if observation.shape != expected_shape:
            raise ValueError(f"Observation shape {observation.shape} does not match expected {expected_shape}")
        
        return observation


    def _get_info(self):
        """
        Helper method to get auxiliary information about the environment's state.

        Can be used for debugging or logging, but the agent should not use it for learning.
        """
        return {
            "device_state": self.device_state,
            "current_step": self.current_step,
            "normalization_range": [self.data_min, self.data_max],
            "observation_shape": (self.obs_image_size[0], self.obs_image_size[1], self.obs_channels)
        }

    def _apply_voltages(self, voltages):
        """
        Apply voltage settings to the quantum device.
        
        Args:
            voltages (np.ndarray): Array of voltage values for each gate
        """
        
        # Update current voltage settings in device state
        for i in range(self.num_voltages):
            self.device_state["current_voltages"][:,:,i] += voltages[i]    # voltage[i] affects gate i (what is the third dimension?)

        # Ensure voltages are within bounds
        self.device_state["current_voltages"] = np.clip(self.device_state["current_voltages"], self.voltage_min, self.voltage_max)
    
         

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


    def _render_frame(self):
        """
        Internal method to create the render image.
        
        Returns:
            np.ndarray: RGB array representation of the environment state
        """
        z, n = self.device_state["model"].charge_sensor_open(self.device_state["current_voltages"])
        
        # Create figure and plot

        vmin, vmax = (-20,20)
        num_ticks = 5
        tick_values = np.linspace(vmin, vmax, num_ticks)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(z[:, :, 0], cmap='viridis', aspect='auto')
        
        # Set x and y axis ticks to correspond to voltage range
        ax.set_xticks(np.linspace(0, z.shape[1]-1, num_ticks))
        ax.set_xticklabels([f'{v:.0f}' for v in tick_values])
        ax.set_yticks(np.linspace(0, z.shape[0]-1, num_ticks))
        ax.set_yticklabels([f'{v:.0f}' for v in tick_values])
        
        ax.set_xlabel("$\Delta$PL (mV)")
        ax.set_ylabel("$\Delta$PR (mV)")
        ax.set_title("$|S_{11}|$ (arb.)")
        
        cbar = plt.colorbar(im, ax=ax)
        c_vmin, c_vmax = im.get_clim()
        c_tick_values = np.linspace(c_vmin, c_vmax, num_ticks)
        cbar.set_ticks(list(c_tick_values))


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
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8) #type: ignore
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
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
    env = QuantumDeviceEnv()
    env.reset()
    env.render()  # This will save the initial state plot
    env.step(env.action_space.sample())
    env.render()  # This will save the plot after the action 
    env.close()