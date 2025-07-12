import gymnasium as gym
from gymnasium import spaces
import numpy as np


class QuantumDeviceEnv(gym.Env):
    """
    Represents the device with its quantum dots 
    """
    #rendering info
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, **kwargs):
        """
        constructor for the environment

        define action and observation spaces

        init state and variables
        """
        super().__init__()

        # --- Define Action and Observation Spaces ---
        
        self.num_voltages = kwargs.get('num_voltages', 5)  # Default to 2 gate voltages and 3 barrier voltages
        
        # Voltage range, no clue what this should be physically
        self.voltage_min = kwargs.get('voltage_min', -2.0)  # Minimum voltage
        self.voltage_max = kwargs.get('voltage_max', 2.0)   # Maximum voltage
        
        # Continuous action space for N voltage controls
        # Each action is a vector of N voltage values
        self.action_space = spaces.Box(
            low=self.voltage_min, 
            high=self.voltage_max, 
            shape=(self.num_voltages,), 
            dtype=np.float32
        )

        # Observation space for quantum device state
        ######################################################### TODO: define observation space

        # --- Initialize State ---
        # Current voltage settings for each gate
        self.current_voltages = np.zeros(self.num_voltages, dtype=np.float32)
        
        # Target state configuration
        self.target_state = kwargs.get('target_state', None)
        
        # Device state variables
        self.device_state = {} #make this the qarray params.

        # --- For Rendering --- Think implementing this will be v useful for debugging
        self.window = None
        self.clock = None


    def reset(self, seed=None, options=None):
        """
        Resets the environment to an initial state and returns the initial observation.

        This method is called at the beginning of each new episode. It should
        reset the state of the environment and return the first observation that
        the agent will see.

        Args:
            seed (int, optional): The seed that is used to initialize the environment's RNG.
            options (dict, optional): Can be used to provide additional information to reset the environment.

        Returns:
            observation (np.ndarray): The initial observation of the space.
            info (dict): A dictionary with auxiliary diagnostic information.
        """
        #seed the random number generator
        super().reset(seed=seed)

        # --- Reset the environment's state ---
        #reset the qarray params
        self.device_state = {}

        # --- Return the initial observation ---
        observation = self._get_obs()
        info = self._get_info()

        # If you are using a human-rendering mode render here
        if self.render_mode == "human":
            self._render_frame()

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
        # action is now a numpy array of shape (num_voltages,) containing voltage values
        self._apply_voltages(action) #this step will update the qarray parameters stored in self.device_state

        # --- Determine the reward ---
        reward = 0.0  #will compare current state to target state
        
        # --- Check for termination or truncation conditions ---
        terminated = False 
        
        truncated = False # time limit reached etc

        # --- Get the new observation and info ---
        observation = self._get_obs() #new state
        info = self._get_info() #diagnostic info
        
        # render here
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        """
        Helper method to get the current observation of the environment. (parameter space)

        Should return a value that conforms to self.observation_space.
        """
        #currently just copies existing voltages
        observation = self.device_state.copy() #this is a dict of qarray params
        
        return observation


    def _get_info(self):
        """
        Helper method to get auxiliary information about the environment's state.

        Can be used for debugging or logging, but the agent should not use it for learning.
        """
        return {
            "current_voltages": self.current_voltages.copy(),
            "target_state": self.target_state,
            "device_state": self.device_state
        }

    def _apply_voltages(self, voltages):
        """
        Apply voltage settings to the quantum device.
        
        Args:
            voltages (np.ndarray): Array of voltage values for each gate
        """
        # Ensure voltages are within bounds
        voltages = np.clip(voltages, self.voltage_min, self.voltage_max)
        
        # Update current voltage settings
        self.current_voltages = voltages.copy()
        
        #map from voltage to qarray params
        



    def render(self):
        """
        Renders the environment.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()


    def _render_frame(self):
        """
        Internal method to create the render image.
        """
        #render the csd
        pass 


    def close(self):
        """
        Performs any necessary cleanup.
        """
  
        pass