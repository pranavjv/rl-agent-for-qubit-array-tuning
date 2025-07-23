import numpy as np
import gymnasium as gym
from gymnasium import spaces


class QArrayWrapper(gym.ObservationWrapper):
    """
    Wrapper to convert multi-modal quantum array observations to single modality.
    This fixes CUDA memory access issues by ensuring consistent data types and shapes.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Get the original observation space
        orig_obs_space = env.observation_space
        
        # Ensure we have the expected structure
        assert isinstance(orig_obs_space, spaces.Dict), "Expected Dict observation space"
        assert 'image' in orig_obs_space.spaces, "Expected 'image' key in observation space"
        assert 'voltages' in orig_obs_space.spaces, "Expected 'voltages' key in observation space"
        
        # Extract image and voltage spaces
        image_space = orig_obs_space['image']
        voltage_space = orig_obs_space['voltages']
        
        # Store dimensions for processing
        self.image_shape = image_space.shape  # (H, W, C)
        self.voltage_shape = voltage_space.shape  # (num_voltages,)
        
        # Create new observation space - just use the image for now
        # This simplifies the observation to avoid multi-modal complexity
        self.observation_space = spaces.Dict({
            'image': image_space,
            'vector': voltage_space  # Rename voltages to vector for DreamerV3
        })
        
        print(f"QArrayWrapper: Original obs space: {orig_obs_space}")
        print(f"QArrayWrapper: New obs space: {self.observation_space}")
    
    def observation(self, obs):
        """
        Transform the observation to ensure consistent data types and formats.
        """
        # Ensure image is uint8 as expected by DreamerV3
        image = obs['image']
        if image.dtype != np.uint8:
            # Convert to uint8 if not already
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Ensure voltages are float32
        voltages = obs['voltages'].astype(np.float32)
        
        # Create new observation
        new_obs = {
            'image': image,
            'vector': voltages  # Rename to 'vector' for DreamerV3
        }
        
        return new_obs


class QArraySimplifiedWrapper(gym.ObservationWrapper):
    """
    Simplified wrapper that only uses image observations to avoid multi-modal issues.
    This is a fallback if the multi-modal approach still causes problems.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Get the original observation space
        orig_obs_space = env.observation_space
        
        # Ensure we have the expected structure
        assert isinstance(orig_obs_space, spaces.Dict), "Expected Dict observation space"
        assert 'image' in orig_obs_space.spaces, "Expected 'image' key in observation space"
        
        # Use only the image observation
        image_space = orig_obs_space['image']
        
        # Create new observation space - single modality
        self.observation_space = image_space
        
        print(f"QArraySimplifiedWrapper: Original obs space: {orig_obs_space}")
        print(f"QArraySimplifiedWrapper: New obs space: {self.observation_space}")
    
    def observation(self, obs):
        """
        Return only the image observation, ensuring it's uint8.
        """
        image = obs['image']
        
        # Ensure image is uint8 as expected by DreamerV3
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        return image
