"""
Example usage of the enhanced video rendering module.

This demonstrates the new modular design with better error handling,
configuration management, and comprehensive analysis capabilities.
"""

import numpy as np
import sys
import os
import torch
from stable_baselines3 import PPO

# Add parent directory and agent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'agent'))

# Import modules using absolute imports
from environment import QuantumDeviceEnv
from video.video_utils import save_gif
from video.plot_utils import plot_rewards

def main():
    # Load environment with rgb_array render mode
    env = QuantumDeviceEnv(config_path="../env_config.yaml", render_mode="rgb_array")
    
    # Load trained SB3 PPO agent
    model_path = "../agent/models/best/best_model.zip"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Available models:")
        for root, dirs, files in os.walk("agent/models"):
            for file in files:
                print(f"  {os.path.join(root, file)}")
        return
    
    print(f"Loading model from: {model_path}")
    agent = PPO.load(model_path)
    
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
        
        # Get action from trained agent using SB3 API
        action, _ = agent.predict(obs, deterministic=True)
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
        save_gif(frames, 'video/outputs/episode.gif', fps=5)
        print(f"Saved {len(frames)} frames as GIF")
    else:
        print("No frames to save")
    
    if rewards:
        plot_rewards(rewards, 'video/outputs/episode_rewards.png')
        print(f"Saved reward plot for {len(rewards)} steps")
        print(f"Total reward: {sum(rewards):.3f}")
        print(f"Mean reward: {np.mean(rewards):.3f}")
    else:
        print("No rewards to plot")
    
    env.close()

if __name__ == "__main__":
    main() 