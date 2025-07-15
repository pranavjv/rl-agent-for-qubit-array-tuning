import torch
import numpy as np
from typing import Dict, Union, Tuple

#just a bunch of helpers

def preprocess_observation(observation: Dict[str, np.ndarray], device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """
    Preprocess multi-modal observation for neural network input.
    
    Args:
        observation (dict): Raw observation with 'image' and 'voltages' keys
        device (str): Device to place tensors on
        
    Returns:
        dict: Preprocessed observation as tensors
    """
    processed = {}
    
    # Process image data
    if 'image' in observation:
        image = observation['image']
        # Ensure correct shape and dtype
        if image.ndim == 3 and image.shape[-1] == 1:
            # Already in (H, W, C) format
            pass
        elif image.ndim == 2:
            # Add channel dimension
            image = image[..., np.newaxis]
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        # Convert to tensor and add batch dimension if needed
        image_tensor = torch.tensor(image, dtype=torch.float32, device=device)
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        processed['image'] = image_tensor
    
    # Process voltage data
    if 'voltages' in observation:
        voltages = observation['voltages']
        # Ensure 1D array
        if voltages.ndim > 1:
            voltages = voltages.flatten()
        
        # Convert to tensor and add batch dimension if needed
        voltage_tensor = torch.tensor(voltages, dtype=torch.float32, device=device)
        if voltage_tensor.ndim == 1:
            voltage_tensor = voltage_tensor.unsqueeze(0)  # Add batch dimension
        
        processed['voltages'] = voltage_tensor
    
    return processed


def batch_observations(observations: list) -> Dict[str, torch.Tensor]:
    """
    Batch multiple observations into a single tensor batch.
    
    Args:
        observations (list): List of observation dictionaries
        
    Returns:
        dict: Batched observations
    """
    if not observations:
        raise ValueError("Empty observation list")
    
    # Get device from first observation
    first_obs = observations[0]
    device = next(iter(first_obs.values())).device
    
    batched = {}
    
    # Batch images
    if 'image' in first_obs:
        images = [obs['image'] for obs in observations]
        batched['image'] = torch.cat(images, dim=0)
    
    # Batch voltages
    if 'voltages' in first_obs:
        voltages = [obs['voltages'] for obs in observations]
        batched['voltages'] = torch.cat(voltages, dim=0)
    
    return batched


def normalize_observation(observation: Dict[str, np.ndarray], 
                         image_range: Tuple[float, float] = (0.0, 1.0),
                         voltage_range: Tuple[float, float] = (-2.0, 2.0)) -> Dict[str, np.ndarray]:
    """
    Normalize observation components to specified ranges.
    
    Args:
        observation (dict): Raw observation
        image_range (tuple): Target range for image data
        voltage_range (tuple): Target range for voltage data
        
    Returns:
        dict: Normalized observation
    """
    normalized = {}
    
    # Normalize image data
    if 'image' in observation:
        image = observation['image'].astype(np.float32)
        # Image should be in [0, 1] range from environment, this is to ensure that the image is in the correct range
        min_val, max_val = image_range
        normalized['image'] = image * (max_val - min_val) + min_val
    
    # Normalize voltage data
    if 'voltages' in observation:
        voltages = observation['voltages'].astype(np.float32)
        # Assuming voltages are already in valid range from environment
        # Clip to ensure they're within bounds
        min_val, max_val = voltage_range
        normalized['voltages'] = np.clip(voltages, min_val, max_val)
    
    return normalized


def validate_observation(observation: Dict[str, Union[np.ndarray, torch.Tensor]], 
                        expected_shapes: Dict[str, Tuple]) -> bool:
    """
    Validate observation structure and shapes.
    
    Args:
        observation (dict): Observation to validate
        expected_shapes (dict): Expected shapes for each component
        
    Returns:
        bool: True if valid, raises ValueError if not
    """
    for key, expected_shape in expected_shapes.items():
        if key not in observation:
            raise ValueError(f"Missing observation key: {key}")
        
        actual_shape = observation[key].shape
        if actual_shape != expected_shape:
            raise ValueError(f"Shape mismatch for {key}: expected {expected_shape}, got {actual_shape}")
    
    return True


def extract_observation_info(observation: Dict[str, Union[np.ndarray, torch.Tensor]]) -> Dict:
    """
    Extract metadata about observation structure.
    
    Args:
        observation (dict): Multi-modal observation
        
    Returns:
        dict: Observation metadata
    """
    info = {
        'modalities': list(observation.keys()),
        'shapes': {},
        'dtypes': {},
        'ranges': {}
    }
    
    for key, value in observation.items():
        info['shapes'][key] = value.shape
        info['dtypes'][key] = str(value.dtype)
        
        if hasattr(value, 'min') and hasattr(value, 'max'):
            info['ranges'][key] = (float(value.min()), float(value.max()))
    
    return info 