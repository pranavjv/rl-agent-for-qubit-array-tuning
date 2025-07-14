"""
Example usage of the enhanced video rendering module.

This demonstrates the new modular design with better error handling,
configuration management, and comprehensive analysis capabilities.
"""

import numpy as np
import sys
import os
import torch

# Add parent directories to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from environment import QuantumDeviceEnv
from agent.ppo_agent import PPOAgent
from video_utils import save_gif
from plot_utils import plot_rewards

def main():
    # Load environment with rgb_array render mode
    env = QuantumDeviceEnv(config_path="RL/env_config.yaml", render_mode="rgb_array")
    
    # Load trained PPO agent
    agent = PPOAgent(env, config_path="RL/agent/config/ppo_config.yaml")
    agent.load_model("RL/agent/models/final_model.pth")
    
    # Run one episode
    obs, _ = env.reset()
    frames = []
    rewards = []
    actions = []
    
    max_steps = 100
    step = 0
    
    print(f"Running episode for up to {max_steps} steps...")
    
    while step < max_steps:
        # Render current state
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        # Get action from trained agent
        action, _, _ = agent.get_action(obs)
        actions.append(action)
        
        # Take step in environment
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        
        step += 1
        
        if terminated or truncated:
            print(f"Episode ended after {step} steps")
            break
    
    # Save results
    if frames:
        save_gif(frames, 'RL/video/outputs/episode.gif', fps=5)
        print(f"Saved {len(frames)} frames as GIF")
    else:
        print("No frames to save")
    
    if rewards:
        plot_rewards(rewards, 'RL/video/outputs/episode_rewards.png')
        print(f"Saved reward plot for {len(rewards)} steps")
        print(f"Total reward: {sum(rewards):.3f}")
        print(f"Mean reward: {np.mean(rewards):.3f}")
    else:
        print("No rewards to plot")
    
    env.close()

if __name__ == "__main__":
    main() 