import os
import torch
import numpy as np

# Set matplotlib backend before importing pyplot to avoid GUI issues
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional, Callable, Dict, Any, Tuple, List, Union, Sequence
import yaml
from pathlib import Path

from RL.environment import QuantumDeviceEnv
from RL.agent.ppo_agent import PPOAgent

class VideoRenderer:
    """
    Renders episodes of a trained RL agent in the environment, saving video and logs.
    Supports overlays and hooks for extensibility.
    """
    def __init__(self, config_path: str, render_mode: str = "rgb_array", fps: Optional[int] = None):
        self.config_path = config_path
        self.render_mode = render_mode
        self.config = self._load_config(config_path)
        self.env = QuantumDeviceEnv(config_path=config_path, render_mode=render_mode)
        self.fps = fps or self.config.get('training', {}).get('render_fps', 10)
        self.frame_delay = 1000 // self.fps
        
        # Set up default output directories within RL/video/
        self.video_dir = Path(__file__).parent
        self.default_output_dir = self.video_dir / "outputs"
        self.default_output_dir.mkdir(exist_ok=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_trained_model(self, model_path: str, config_path: str = "../agent/config/ppo_config.yaml") -> PPOAgent:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Create PPOAgent with the environment
        agent = PPOAgent(self.env, config_path)
        
        # Load the trained model
        agent.load_model(model_path)
        
        return agent

    def _convert_obs_to_tensor(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Convert numpy observation to torch tensors for the agent.
        
        Args:
            obs: Observation from environment (numpy arrays or other types)
            
        Returns:
            dict: Observation as torch tensors
        """
        tensor_obs = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                tensor_obs[key] = torch.from_numpy(value).float()
            elif isinstance(value, torch.Tensor):
                # If already a tensor, just ensure it's the right dtype
                tensor_obs[key] = value.float()
            else:
                # Use clone().detach() to avoid the warning
                tensor_obs[key] = torch.tensor(value, dtype=torch.float32)
        return tensor_obs

    def render_episode(self, agent: PPOAgent, max_steps: Optional[int] = None, save_path: Optional[str] = None, log_path: Optional[str] = None, overlay_fn: Optional[Callable] = None) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        frames, actions, rewards = [], [], []
        obs, info = self.env.reset()
        step_count = 0
        max_steps = int(max_steps) if max_steps is not None else int(self.env.max_steps)
        
        while step_count < max_steps:
            # Convert observation to tensor format expected by PPOAgent
            obs_tensor = self._convert_obs_to_tensor(obs)
            
            # Get action from agent
            action, log_prob, value = agent.get_action(obs_tensor)
            
            # Convert action to numpy if it's a tensor
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            
            obs, reward, terminated, truncated, info = self.env.step(action)
            frame = self.env.render()
            if overlay_fn and frame is not None:
                frame = overlay_fn(frame, step_count, action, reward, info)
            if frame is not None:
                frames.append(frame)
            actions.append(np.array(action))
            rewards.append(reward)
            step_count += 1
            if terminated or truncated:
                break
        if save_path and frames:
            self._save_video(frames, save_path)
        if log_path:
            np.savez(log_path, actions=np.array(actions), rewards=np.array(rewards))
        return frames, actions, rewards

    def _save_video(self, frames: Sequence[np.ndarray], save_path: str):
        """
        Save frames as a video file (MP4 or GIF) or as individual images if video creation fails.
        Uses imageio for GIFs (robust for numpy arrays), and matplotlib animation for MP4s (if ffmpeg is available).
        """
        import imageio
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Debug: Check frame information
        print(f"[VideoRenderer] Number of frames: {len(frames)}")
        if frames:
            print(f"[VideoRenderer] First frame shape: {frames[0].shape}")
            print(f"[VideoRenderer] First frame dtype: {frames[0].dtype}")
            print(f"[VideoRenderer] First frame range: [{frames[0].min()}, {frames[0].max()}]")
            print(f"[VideoRenderer] First frame non-zero pixels: {np.count_nonzero(frames[0])}")

        # Ensure all frames are uint8 RGB
        def to_rgb_uint8(frame):
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            if frame.ndim == 2:
                frame = np.stack([frame]*3, axis=-1)
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            return frame
        frames = [to_rgb_uint8(f) for f in frames]

        ext = os.path.splitext(save_path)[1].lower()
        try:
            if ext == '.gif':
                # Use imageio for GIFs (no 'format' arg needed, inferred from extension)
                print(f"[VideoRenderer] Saving GIF with {len(frames)} frames at {self.fps} fps to {save_path}")
                duration = 1.0 / self.fps
                imageio.mimsave(save_path, list(frames), duration=duration)
                print(f"[VideoRenderer] GIF saved: {save_path}")
            elif ext == '.mp4':
                # Use matplotlib animation for MP4s (ffmpeg required)
                print(f"[VideoRenderer] Saving MP4 with {len(frames)} frames at {self.fps} fps to {save_path}")
                fig, ax = plt.subplots(figsize=(8, 6))
                def animate(i):
                    ax.clear()
                    ax.imshow(frames[i])
                    ax.set_title(f"Step {i+1}")
                    ax.axis('off')
                    return ax,
                from matplotlib import animation
                anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=self.frame_delay, blit=True)
                try:
                    anim.save(save_path, writer='ffmpeg', fps=self.fps)
                    print(f"[VideoRenderer] MP4 saved: {save_path}")
                except Exception as e:
                    print(f"[VideoRenderer] Failed to save MP4 with ffmpeg: {e}")
                    # Fallback: save as GIF instead
                    gif_path = save_path.replace('.mp4', '.gif')
                    print(f"[VideoRenderer] Falling back to GIF: {gif_path}")
                    duration = 1.0 / self.fps
                    imageio.mimsave(gif_path, list(frames), duration=duration)
                    print(f"[VideoRenderer] GIF saved: {gif_path}")
                finally:
                    plt.close('all')
            else:
                # Unknown extension, fallback to GIF
                gif_path = save_path + '.gif'
                print(f"[VideoRenderer] Unknown extension, saving as GIF: {gif_path}")
                duration = 1.0 / self.fps
                imageio.mimsave(gif_path, list(frames), duration=duration)
                print(f"[VideoRenderer] GIF saved: {gif_path}")
        except Exception as e:
            print(f"[VideoRenderer] Error saving video: {e}")
            # Fallback: save individual frames as PNGs
            frames_dir = save_path.replace(ext, '_frames')
            os.makedirs(frames_dir, exist_ok=True)
            for i, frame in enumerate(frames):
                plt.figure(figsize=(8, 6))
                plt.imshow(frame)
                plt.title(f"Step {i+1}")
                plt.axis('off')
                frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
                plt.savefig(frame_path, dpi=150, bbox_inches='tight')
                plt.close()
            print(f"[VideoRenderer] Frames saved to: {frames_dir}")
        finally:
            plt.close('all')

    def render_debug_episode(self, agent: PPOAgent, save_dir: Optional[Union[str, Path]] = None) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        """
        Render episode with detailed debugging information.
        Saves video, logs, and summary plots to RL/video/outputs/debug/
        """
        if save_dir is None:
            save_dir = self.default_output_dir / "debug"
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = save_dir / "episode.mp4"
        log_path = save_dir / "episode.npz"
        
        frames, actions, rewards = self.render_episode(
            agent, 
            save_path=str(video_path), 
            log_path=str(log_path)
        )
        
        # Create summary plot
        self._create_summary_plot(actions, rewards, save_dir)
        
        print(f"Debug episode saved to {save_dir}")
        return frames, actions, rewards

    def render_multiple_episodes(self, agent: PPOAgent, num_episodes: int = 5, output_dir: Optional[Union[str, Path]] = None) -> List[Dict]:
        """
        Render multiple episodes for analysis.
        Saves all videos and logs to RL/video/outputs/batch/
        """
        if output_dir is None:
            output_dir = self.default_output_dir / "batch"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        episode_stats = []
        
        for episode in range(num_episodes):
            print(f"\nRendering episode {episode + 1}/{num_episodes}")
            
            video_path = output_dir / f"episode_{episode+1}.mp4"
            log_path = output_dir / f"episode_{episode+1}.npz"
            
            frames, actions, rewards = self.render_episode(
                agent,
                save_path=str(video_path),
                log_path=str(log_path)
            )
            
            episode_stats.append({
                'episode': episode + 1,
                'total_reward': sum(rewards),
                'episode_length': len(actions),
                'mean_reward': np.mean(rewards),
                'final_reward': rewards[-1] if rewards else 0
            })
        
        # Save summary
        summary_path = output_dir / "episode_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("Episode Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for stats in episode_stats:
                f.write(f"Episode {stats['episode']}:\n")
                f.write(f"  Total Reward: {stats['total_reward']:.3f}\n")
                f.write(f"  Episode Length: {stats['episode_length']}\n")
                f.write(f"  Mean Reward: {stats['mean_reward']:.3f}\n")
                f.write(f"  Final Reward: {stats['final_reward']:.3f}\n\n")
        
        print(f"\nRendered {num_episodes} episodes to {output_dir}")
        print(f"Summary saved to {summary_path}")
        return episode_stats

    def _create_summary_plot(self, actions: List[np.ndarray], rewards: List[float], save_dir: Path):
        """Create summary plots for debugging."""
        # Use a non-interactive backend to avoid GUI issues
        import matplotlib
        matplotlib.use('Agg')
        
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot actions
            actions_array = np.array(actions)
            for i in range(actions_array.shape[1]):
                ax1.plot(actions_array[:, i], label=f'Action {i+1}')
            ax1.set_title('Actions Over Time')
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Voltage')
            ax1.legend()
            ax1.grid(True)
            
            # Plot rewards
            ax2.plot(rewards, 'b-', label='Reward')
            ax2.set_title('Rewards Over Time')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Reward')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(save_dir / "episode_summary.png", dpi=150, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error creating summary plot: {e}")
            # Try alternative approach
            try:
                plt.figure(figsize=(12, 8))
                plt.subplot(2, 1, 1)
                actions_array = np.array(actions)
                for i in range(actions_array.shape[1]):
                    plt.plot(actions_array[:, i], label=f'Action {i+1}')
                plt.title('Actions Over Time')
                plt.xlabel('Step')
                plt.ylabel('Voltage')
                plt.legend()
                plt.grid(True)
                
                plt.subplot(2, 1, 2)
                plt.plot(rewards, 'b-', label='Reward')
                plt.title('Rewards Over Time')
                plt.xlabel('Step')
                plt.ylabel('Reward')
                plt.legend()
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(save_dir / "episode_summary.png", dpi=150, bbox_inches='tight')
                plt.close()
            except Exception as e2:
                print(f"Failed to create summary plot: {e2}")
        finally:
            plt.close('all') 