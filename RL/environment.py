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

        define action and obervation spaces

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
        
        # Target voltage configuration
        self.target_voltages = kwargs.get('target_voltages', None)
        
        # Device state variables, probably rest of params here?
        self.device_state = {}

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
        # Apply the voltage settings to the quantum device
        # action is now a numpy array of shape (num_voltages,) containing voltage values
        self._apply_voltages(action)

        # --- Determine the reward ---
        # The reward function is crucial for training the agent.
        # For quantum dot tuning, reward could be based on:
        # - Coulomb blockade visibility
        # - Charge stability diagram quality
        # - Distance from target charge configuration
        # - Device stability metrics
        reward = 0.0  # Placeholder - implement based on your quantum device metrics
        # Example: reward = self._calculate_quantum_quality_metric()
        
        # --- Check for termination or truncation conditions ---
        # `terminated` is True if the agent reaches a terminal state (e.g., wins or loses).
        terminated = False # Example: self._agent_location == self._target_location
        
        # `truncated` is True if the episode is ended for reasons not related to the task itself
        # (e.g., a time limit is reached).
        truncated = False # Example: self.current_step > 200

        # --- Get the new observation and info ---
        observation = self._get_obs()
        info = self._get_info()
        
        # If you are using a human-rendering mode, you might want to render here
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        """
        Helper method to get the current observation of the environment.

        Should return a value that conforms to self.observation_space.
        """
        # Return current voltage settings as observation
        # You might also want to include:
        # - Measured quantum state parameters
        # - Device stability metrics
        # - Coulomb blockade measurements
        # - Charge stability diagram features
        
        # For now, return the current voltages
        # In practice, you'd want to include measurements from your quantum device
        observation = self.current_voltages.copy()
        
        # If you have additional measurements, concatenate them:
        # quantum_measurements = self._get_quantum_measurements()
        # observation = np.concatenate([self.current_voltages, quantum_measurements])
        
        return observation


    def _get_info(self):
        """
        Helper method to get auxiliary information about the environment's state.

        Can be used for debugging or logging, but the agent should not use it for learning.
        """
        return {
            "current_voltages": self.current_voltages.copy(),
            "target_voltages": self.target_voltages,
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
        
        # Here you would integrate with your actual quantum device
        # For example:
        # - Send voltage commands to hardware
        # - Update device simulation model
        # - Measure resulting quantum state
        
        # Placeholder for device interaction
        # self._update_device_simulation(voltages)
        # self._measure_quantum_state()


    def render(self):
        """
        Renders the environment.

        The "human" mode typically pops up a window for visualization.
        The "rgb_array" mode returns a numpy array of the rendered image.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
        # The 'human' mode is handled by the step and reset methods.

    def _render_frame(self):
        """
        Internal method to create the render image.
        """
        # This is where you would use a library like Pygame or Matplotlib to draw
        # the current state of the environment.
        # For "human" mode, you'd display this to the screen.
        # For "rgb_array" mode, you'd return it as a numpy array.
        #
        # Example with Pygame:
        # if self.window is None and self.render_mode == "human":
        #     pygame.init()
        #     self.window = pygame.display.set_mode((500, 500))
        # if self.clock is None and self.render_mode == "human":
        #     self.clock = pygame.time.Clock()
        #
        # canvas = pygame.Surface((500, 500))
        # canvas.fill((255, 255, 255)) # White background
        # # ... draw your elements on the canvas ...
        #
        # if self.render_mode == "human":
        #     self.window.blit(canvas, canvas.get_rect())
        #     pygame.event.pump()
        #     pygame.display.update()
        #     self.clock.tick(self.metadata["render_fps"])
        # else: # "rgb_array"
        #     return np.transpose(
        #         np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        #     )
        pass # This should be implemented if you want to render


    def close(self):
        """
        Performs any necessary cleanup.

        This is called when the environment is no longer needed. For example, you
        can close the rendering window here.
        """
        # if self.window is not None:
        #     pygame.display.quit()
        #     pygame.quit()
        pass