#!/usr/bin/env python3
"""
Example usage of the VideoRenderer with organized directory structure.
All outputs are saved within RL/video/outputs/
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from RL.video.renderer import VideoRenderer
from RL.agent.networks.multi_modal_net import MultiModalNetwork

def main():
    # Initialize renderer
    renderer = VideoRenderer(
        config_path="RL/env_config.yaml",
        render_mode="rgb_array",
        fps=10
    )
    
    # Load trained model (replace with your actual model path)
    model_path = "RL/agent/models/final_model.pth"
    
    try:
        agent = renderer.load_trained_model(model_path, MultiModalNetwork)
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model not found at {model_path}")
        print("Please train a model first or update the model_path")
        return
    
    # Example 1: Render a single debug episode
    print("\n=== Rendering Debug Episode ===")
    frames, actions, rewards = renderer.render_debug_episode(agent)
    print(f"Debug episode completed with {len(frames)} frames")
    print(f"Total reward: {sum(rewards):.3f}")
    
    # Example 2: Render multiple episodes for batch evaluation
    print("\n=== Rendering Multiple Episodes ===")
    episode_stats = renderer.render_multiple_episodes(agent, num_episodes=3)
    
    # Print summary
    print("\n=== Episode Summary ===")
    for stats in episode_stats:
        print(f"Episode {stats['episode']}: "
              f"Total Reward={stats['total_reward']:.3f}, "
              f"Length={stats['episode_length']}, "
              f"Mean Reward={stats['mean_reward']:.3f}")
    
    print(f"\nAll outputs saved to RL/video/outputs/")
    print("- Debug episode: RL/video/outputs/debug/")
    print("- Batch episodes: RL/video/outputs/batch/")

if __name__ == "__main__":
    main() 