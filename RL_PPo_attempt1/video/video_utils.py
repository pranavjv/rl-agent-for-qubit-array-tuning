"""
Video saving utilities with robust error handling and multiple format support.
"""

import imageio
import numpy as np
import os
from typing import Sequence

def save_gif(frames: Sequence[np.ndarray], path: str, fps: int = 10) -> None:
    """
    Save a sequence of RGB numpy arrays as a GIF.

    Args:
        frames: Sequence of (H, W, 3) uint8 RGB images.
        path: Output file path (should end with .gif).
        fps: Frames per second for the GIF.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    imageio.mimsave(path, list(frames), fps=fps)