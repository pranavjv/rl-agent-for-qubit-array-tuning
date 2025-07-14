import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional

def plot_rewards(rewards: List[float], save_path: Optional[str] = None):
    plt.figure(figsize=(8, 4))
    plt.plot(rewards, label='Reward')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Reward per Step')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def plot_actions(actions: np.ndarray, save_path: Optional[str] = None):
    plt.figure(figsize=(8, 4))
    for i in range(actions.shape[1]):
        plt.plot(actions[:, i], label=f'Action {i+1}')
    plt.xlabel('Step')
    plt.ylabel('Action Value')
    plt.title('Actions per Step')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()

def summarize_episodes(log_paths: List[str]):
    """Print summary stats for a list of episode log .npz files."""
    stats = []
    for path in log_paths:
        data = np.load(path, allow_pickle=True)
        rewards = data['rewards']
        total_reward = np.sum(rewards)
        mean_reward = np.mean(rewards)
        length = len(rewards)
        stats.append({'file': path, 'total_reward': total_reward, 'mean_reward': mean_reward, 'length': length})
    for s in stats:
        print(f"{s['file']}: Total={s['total_reward']:.2f}, Mean={s['mean_reward']:.2f}, Length={s['length']}")
    return stats 