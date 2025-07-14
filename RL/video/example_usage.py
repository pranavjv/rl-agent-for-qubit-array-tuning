import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from RL.video.renderer import VideoRenderer

def main():
    # Initialize renderer with absolute paths
    renderer = VideoRenderer(
        config_path="../env_config.yaml",
        render_mode="rgb_array",
        fps=10
    )
    
    # Load trained model (replace with your actual model path)
    model_path = "../agent/models/final_model.pth"
    config_path = "../agent/config/ppo_config.yaml"
    
    try:
        agent = renderer.load_trained_model(model_path, config_path)
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError as e:
        print(f"Model or config not found: {e}")
        print("Please train a model first or update the paths")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Example 1: Render a single debug episode
    print("\n=== Rendering Debug Episode ===")
    try:
        frames, actions, rewards = renderer.render_debug_episode(agent)
        print(f"Debug episode completed with {len(frames)} frames")
        print(f"Total reward: {sum(rewards):.3f}")
    except Exception as e:
        print(f"Error rendering debug episode: {e}")
        return
    
    # Example 2: Render multiple episodes for batch evaluation
    print("\n=== Rendering Multiple Episodes ===")
    try:
        episode_stats = renderer.render_multiple_episodes(agent, num_episodes=3)
        
        # Print summary
        print("\n=== Episode Summary ===")
        for stats in episode_stats:
            print(f"Episode {stats['episode']}: "
                  f"Total Reward={stats['total_reward']:.3f}, "
                  f"Length={stats['episode_length']}, "
                  f"Mean Reward={stats['mean_reward']:.3f}")
    except Exception as e:
        print(f"Error rendering multiple episodes: {e}")
        return
    
    print(f"\nAll outputs saved to RL/video/outputs/")
    print("- Debug episode: RL/video/outputs/debug/")
    print("- Batch episodes: RL/video/outputs/batch/")

if __name__ == "__main__":
    main() 