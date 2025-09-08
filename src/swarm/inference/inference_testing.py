#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import os

# Add src directory to path for clean imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
sys.path.insert(0, str(src_dir))

from swarm.inference.model_loader import load_model, create_env


def create_output_dir():
    """Create output directory for GIFs."""
    output_dir = Path(__file__).parent / "inference_outputs"
    output_dir.mkdir(exist_ok=True)
    return output_dir


def save_scan_frame(scan_data, channel_idx, frame_idx, temp_dir):
    """Save a single frame of scan data as an image."""
    plt.figure(figsize=(8, 6))
    plt.imshow(scan_data[:, :, channel_idx], cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Charge sensor signal')
    plt.title(f'Channel {channel_idx} - Frame {frame_idx}')
    plt.xlabel('Gate voltage')
    plt.ylabel('Gate voltage')
    
    frame_path = temp_dir / f"channel_{channel_idx}_frame_{frame_idx:03d}.png"
    plt.savefig(frame_path, dpi=100, bbox_inches='tight')
    plt.close()
    return frame_path


def create_gif_from_frames(frame_paths, output_path, duration=500):
    """Create GIF from a list of image frame paths."""
    images = []
    for frame_path in sorted(frame_paths):
        images.append(Image.open(frame_path))
    
    if images:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0
        )
        print(f"Saved GIF: {output_path}")
    
    # Clean up temporary frames
    for frame_path in frame_paths:
        frame_path.unlink()


def run_inference_episode(algo, max_steps=100):
    """
    Run a single inference episode and collect scan data for each channel.
    
    Args:
        algo: Loaded RLlib algorithm
        max_steps: Maximum number of steps per episode
    
    Returns:
        List of scan data arrays for each step
    """
    env = create_env()
    scan_history = []
    
    try:
        obs, info = env.reset()
        scan_history.append(obs['image'])
        
        for step in range(max_steps):
            # Compute actions for all agents
            if isinstance(obs, dict) and any(agent_id in obs for agent_id in env.all_agent_ids):
                # Multi-agent case
                actions = {}
                for agent_id, agent_obs in obs.items():
                    if agent_id in env.all_agent_ids:
                        policy_id = f"{agent_id.split('_')[0]}_policy"
                        action = algo.compute_single_action(agent_obs, policy_id=policy_id)
                        actions[agent_id] = action
            else:
                # Single agent case
                actions = algo.compute_single_action(obs)
            
            # Take step in environment
            obs, reward, terminated, truncated, info = env.step(actions)
            
            # Store scan data (image observations)
            if isinstance(obs, dict):
                # For multi-agent, get base environment observation
                base_obs = env.base_env.get_observation()
                scan_history.append(base_obs['image'])
            else:
                scan_history.append(obs['image'])
            
            # Check if episode is done
            if terminated or truncated:
                print(f"Episode ended at step {step + 1}")
                break
        
        print(f"Episode completed with {len(scan_history)} frames")
        return scan_history
        
    except Exception as e:
        print(f"Error during episode: {e}")
        return scan_history
    finally:
        env.close()


def create_channel_gifs(scan_history, output_dir):
    """
    Create separate GIFs for each scan channel.
    
    Args:
        scan_history: List of scan data arrays (height, width, channels)
        output_dir: Directory to save GIFs
    """
    if not scan_history:
        print("No scan history to process")
        return
    
    # Get number of channels from first frame
    num_channels = scan_history[0].shape[2]
    print(f"Creating GIFs for {num_channels} channels with {len(scan_history)} frames")
    
    # Create temporary directory for frame images
    temp_dir = output_dir / "temp_frames"
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Process each channel separately
        for channel_idx in range(num_channels):
            print(f"Processing channel {channel_idx}...")
            frame_paths = []
            
            # Create frame images for this channel
            for frame_idx, scan_data in enumerate(scan_history):
                frame_path = save_scan_frame(scan_data, channel_idx, frame_idx, temp_dir)
                frame_paths.append(frame_path)
            
            # Create GIF for this channel
            gif_path = output_dir / f"channel_{channel_idx}_scan_evolution.gif"
            create_gif_from_frames(frame_paths, gif_path)
        
    finally:
        # Clean up temporary directory
        if temp_dir.exists():
            for file in temp_dir.iterdir():
                file.unlink()
            temp_dir.rmdir()


def run_inference_test(checkpoint_path=None, num_episodes=1, max_steps_per_episode=50):
    """
    Run inference test with GIF generation.
    
    Args:
        checkpoint_path: Path to checkpoint (None for latest)
        num_episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
    """
    print("Loading model...")
    algo = load_model(checkpoint_path)
    
    output_dir = create_output_dir()
    print(f"Output directory: {output_dir}")
    
    for episode in range(num_episodes):
        print(f"\nRunning episode {episode + 1}/{num_episodes}")
        
        # Run inference episode
        scan_history = run_inference_episode(algo, max_steps_per_episode)
        
        if scan_history:
            # Create episode-specific output directory
            episode_dir = output_dir / f"episode_{episode + 1}"
            episode_dir.mkdir(exist_ok=True)
            
            # Create GIFs for each channel
            create_channel_gifs(scan_history, episode_dir)
            
            print(f"Episode {episode + 1} GIFs saved to: {episode_dir}")
        else:
            print(f"No scan data collected for episode {episode + 1}")
    
    print(f"\nInference testing complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference testing with GIF generation")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint directory (default: latest)")
    parser.add_argument("--episodes", type=int, default=1,
                       help="Number of episodes to run (default: 1)")
    parser.add_argument("--max-steps", type=int, default=50,
                       help="Maximum steps per episode (default: 50)")
    
    args = parser.parse_args()
    
    try:
        run_inference_test(
            checkpoint_path=args.checkpoint,
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps
        )
    except KeyboardInterrupt:
        print("\nInference testing interrupted by user")
    except Exception as e:
        print(f"Error during inference testing: {e}")
        import traceback
        traceback.print_exc()
    finally:
        import ray
        if ray.is_initialized():
            ray.shutdown()