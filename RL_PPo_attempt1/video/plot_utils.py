import matplotlib.pyplot as plt
import os
from typing import Sequence

def plot_rewards(rewards: Sequence[float], path: str) -> None:
    """
    Plot rewards vs. step and save as a PNG image.

    Args:
        rewards: Sequence of reward values.
        path: Output file path (should end with .png).
    """
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, label='Reward')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Reward per Step')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close() 