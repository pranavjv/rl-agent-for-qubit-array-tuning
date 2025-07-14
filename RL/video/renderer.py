import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Optional, Callable, Dict, Any, Tuple, List, Union
import yaml
from pathlib import Path

from RL.environment import QuantumDeviceEnv
from RL.basic_ppo import PPO

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

    def load_trained_model(self, model_path: str, network_class) -> PPO:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        agent = PPO(network_class, self.env)
        checkpoint = torch.load(model_path, map_location='cpu')
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        if 'log_std' in checkpoint:
            agent.log_std.data = checkpoint['log_std']
        agent.actor.eval()
        agent.critic.eval()
        return agent

    def render_episode(self, agent: PPO, max_steps: Optional[int] = None, save_path: Optional[str] = None, log_path: Optional[str] = None, overlay_fn: Optional[Callable] = None) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
        frames, actions, rewards = [], [], []
        obs, info = self.env.reset()
        step_count = 0
        max_steps = int(max_steps) if max_steps is not None else int(self.env.max_steps)
        while step_count < max_steps:
            action, _ = agent.get_action(obs)
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

    def _save_video(self, frames: List[np.ndarray], save_path: str):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        def animate(i):
            ax.clear()
            ax.imshow(frames[i])
            ax.set_title(f"Step {i+1}")
            ax.axis('off')
            return ax,
        anim = animation.FuncAnimation(fig, animate, frames=len(frames), interval=self.frame_delay, blit=True)
        anim.save(save_path, writer='pillow', fps=self.fps)
        plt.close()

    def render_debug_episode(self, agent: PPO, save_dir: Optional[Union[str, Path]] = None) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
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

    def render_multiple_episodes(self, agent: PPO, num_episodes: int = 5, output_dir: Optional[Union[str, Path]] = None) -> List[Dict]:
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
        plt.savefig(save_dir / "episode_summary.png", dpi=150)
        plt.close() 