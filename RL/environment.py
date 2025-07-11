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
        # These must be gymnasium.spaces objects.
        # Example for a discrete action space with 2 actions (e.g., left or right):
        self.action_space = spaces.Discrete(2)

        # Example for a continuous observation space (e.g., a 4-dimensional vector):
        # The shape is (4,) and values can be between -infinity and +infinity.
        # For pixel observations, you would use spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        # --- Initialize State ---
        # It's good practice to initialize the state of the environment.
        # However, the actual state for an episode is set in the reset() method.
        self._agent_location = None # Example internal state
        self._target_location = None # Example internal state

        # --- For Rendering (Optional) ---
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
        # We need the following line to seed the random number generator
        super().reset(seed=seed)

        # --- Reset the environment's state ---
        # Example: Initialize the agent's location to a random spot.
        # self._agent_location = self.np_random.integers(0, 10)
        # self._target_location = 5 # Or some fixed/random value

        # --- Return the initial observation ---
        # The observation should match the self.observation_space.
        # For example, it might be a numpy array representing the agent's state.
        observation = self._get_obs()
        info = self._get_info()

        # If you are using a human-rendering mode, you might want to render here
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
        # This is where you implement the "rules" of your game or simulation.
        # For example:
        # if action == 0:
        #     self._agent_location -= 1
        # else:
        #     self._agent_location += 1

        # --- Determine the reward ---
        # The reward function is crucial for training the agent.
        # reward = 0
        # if self._agent_location == self._target_location:
        #     reward = 1.0
        
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
        # For example, return a numpy array of the agent and target locations
        # return np.array([self._agent_location, self._target_location], dtype=np.float32)
        raise NotImplementedError("This method must be implemented")


    def _get_info(self):
        """
        Helper method to get auxiliary information about the environment's state.

        Can be used for debugging or logging, but the agent should not use it for learning.
        """
        # For example, return the distance to the target
        # return {"distance": np.abs(self._agent_location - self._target_location)}
        return {}


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