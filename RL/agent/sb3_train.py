from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))


import os
import sys
import yaml
import argparse
import torch.nn as nn
from typing import Optional
from environment import QuantumDeviceEnv
from sb3_policy import QuantumMultiModalPolicy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed



def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_env(config_path: str, render_mode: Optional[str] = None):
    """Create and wrap the quantum device environment."""
    env = QuantumDeviceEnv(config_path=config_path, render_mode=render_mode)
    env = Monitor(env)
    return env


def train_sb3_agent(
    env_config_path: str = "env_config.yaml",
    ppo_config_path: str = "config/ppo_config.yaml",
    total_timesteps: int = 10000,
    save_interval: int = 1000,
    eval_interval: int = 500,
    tensorboard_log: str = "./logs/sb3_quantum",
    model_save_path: str = "./models/sb3_quantum_model",
    verbose: int = 1
):
    """
    Train a PPO agent using Stable Baselines 3.
    
    Args:
        env_config_path: Path to environment configuration
        ppo_config_path: Path to PPO configuration
        total_timesteps: Total timesteps for training
        save_interval: Interval for saving checkpoints
        eval_interval: Interval for evaluation
        tensorboard_log: Path for tensorboard logs
        model_save_path: Path to save the final model
        verbose: Verbosity level
    """
    
    print("=== SB3 Quantum Device Training ===")
    
    # Load configurations
    print(f"Loading environment config from: {env_config_path}")
    env_config = load_config(env_config_path)
    
    print(f"Loading PPO config from: {ppo_config_path}")
    ppo_config = load_config(ppo_config_path)
    
    # Set random seed
    seed = env_config.get('training', {}).get('seed', 42)
    set_random_seed(seed)
    print(f"Random seed set to: {seed}")
    
    # Create environment
    print("Creating quantum device environment...")
    env = create_env(env_config_path)
    
    # Create vectorized environment
    env = DummyVecEnv([lambda: env])
    
    # Create callbacks
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_interval,
        save_path="./models/checkpoints/",
        name_prefix="sb3_quantum_checkpoint"
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_env = create_env(env_config_path)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best/",
        log_path="./logs/eval/",
        eval_freq=eval_interval,
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)
    
    # Extract PPO parameters from config
    ppo_params = ppo_config.get('ppo', {})
    optimization_params = ppo_config.get('optimization', {})
    
    # Create PPO model with custom policy
    print("Creating PPO model with custom multi-modal policy...")
    model = PPO(
        policy=QuantumMultiModalPolicy,
        env=env,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        
        # PPO parameters
        learning_rate=ppo_params.get('actor_lr', 3e-4),
        n_steps=ppo_params.get('timesteps_per_batch', 1024),
        batch_size=64,  # SB3 default
        n_epochs=ppo_params.get('updates_per_iteration', 5),
        gamma=ppo_params.get('gamma', 0.99),
        gae_lambda=ppo_params.get('gae_lambda', 0.95),
        clip_range=ppo_params.get('epsilon', 0.2),
        ent_coef=ppo_params.get('entropy_coef', 0.01),
        max_grad_norm=optimization_params.get('max_grad_norm', 0.5),
        
        # Device settings
        device='auto',  # Let SB3 choose best device
        
        # Policy parameters
        policy_kwargs={
            'features_extractor_class': None,  # We use custom policy
            'features_extractor_kwargs': {},
            'net_arch': [],  # Not used with custom policy
            'activation_fn': nn.ReLU,
        }
    )
    
    # Print model info
    print(f"Model created successfully!")
    print(f"Policy: {type(model.policy).__name__}")
    print(f"Environment: {type(env.envs[0]).__name__}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Train the model
    print(f"\nStarting training for {total_timesteps} timesteps...")
    print(f"Checkpoints will be saved every {save_interval} timesteps")
    print(f"Evaluation will be performed every {eval_interval} timesteps")
    print(f"Tensorboard logs: {tensorboard_log}")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True
        )
        
        # Save final model
        print(f"\nTraining completed! Saving final model to: {model_save_path}")
        model.save(model_save_path)
        
        # Save training info
        training_info = {
            'total_timesteps': total_timesteps,
            'final_model_path': model_save_path,
            'config_files': {
                'env_config': env_config_path,
                'ppo_config': ppo_config_path
            },
            'hyperparameters': {
                'learning_rate': ppo_params.get('actor_lr', 3e-4),
                'n_steps': ppo_params.get('timesteps_per_batch', 1024),
                'n_epochs': ppo_params.get('updates_per_iteration', 5),
                'gamma': ppo_params.get('gamma', 0.99),
                'gae_lambda': ppo_params.get('gae_lambda', 0.95),
                'clip_range': ppo_params.get('epsilon', 0.2),
                'ent_coef': ppo_params.get('entropy_coef', 0.01),
            }
        }
        
        info_path = f"{model_save_path}_info.yaml"
        with open(info_path, 'w') as f:
            yaml.dump(training_info, f, default_flow_style=False)
        
        print(f"Training info saved to: {info_path}")
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    
    finally:
        # Clean up
        env.close()
        eval_env.close()


def evaluate_sb3_model(
    model_path: str,
    env_config_path: str = "env_config.yaml",
    num_episodes: int = 10,
    render: bool = False
):
    """
    Evaluate a trained SB3 model.
    
    Args:
        model_path: Path to the trained model
        env_config_path: Path to environment configuration
        num_episodes: Number of episodes to evaluate
        render: Whether to render episodes
    """
    
    print(f"=== Evaluating SB3 Model: {model_path} ===")
    
    # Load model
    print("Loading model...")
    model = PPO.load(model_path)
    
    # Create environment
    render_mode = "human" if render else None
    env = create_env(env_config_path, render_mode=render_mode)
    
    # Evaluation
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"Running {num_episodes} evaluation episodes...")
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if terminated:  # Success if terminated (not truncated)
            success_count += 1
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.3f}, Length = {episode_length}")
    
    # Print results
    mean_reward = sum(episode_rewards) / len(episode_rewards)
    std_reward = (sum((r - mean_reward) ** 2 for r in episode_rewards) / len(episode_rewards)) ** 0.5
    mean_length = sum(episode_lengths) / len(episode_lengths)
    success_rate = success_count / num_episodes
    
    print(f"\n=== Evaluation Results ===")
    print(f"Mean Reward: {mean_reward:.3f} ± {std_reward:.3f}")
    print(f"Mean Episode Length: {mean_length:.1f}")
    print(f"Success Rate: {success_rate:.1%} ({success_count}/{num_episodes})")
    print(f"Min Reward: {min(episode_rewards):.3f}")
    print(f"Max Reward: {max(episode_rewards):.3f}")
    
    env.close()


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description="Train or evaluate SB3 agent on quantum device environment")
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train",
                       help="Mode: train or evaluate")
    parser.add_argument("--env-config", default="env_config.yaml",
                       help="Path to environment configuration file")
    parser.add_argument("--ppo-config", default="config/ppo_config.yaml",
                       help="Path to PPO configuration file")
    parser.add_argument("--model-path", default="./models/sb3_quantum_model",
                       help="Path to model (for evaluation) or save path (for training)")
    parser.add_argument("--total-timesteps", type=int, default=10000,
                       help="Total timesteps for training")
    parser.add_argument("--save-interval", type=int, default=1000,
                       help="Interval for saving checkpoints")
    parser.add_argument("--eval-interval", type=int, default=500,
                       help="Interval for evaluation during training")
    parser.add_argument("--num-episodes", type=int, default=10,
                       help="Number of episodes for evaluation")
    parser.add_argument("--render", action="store_true",
                       help="Render episodes during evaluation")
    parser.add_argument("--verbose", type=int, default=1,
                       help="Verbosity level")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./models/checkpoints", exist_ok=True)
    os.makedirs("./models/best", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./logs/eval", exist_ok=True)
    
    if args.mode == "train":
        train_sb3_agent(
            env_config_path=args.env_config,
            ppo_config_path=args.ppo_config,
            total_timesteps=args.total_timesteps,
            save_interval=args.save_interval,
            eval_interval=args.eval_interval,
            model_save_path=args.model_path,
            verbose=args.verbose
        )
    elif args.mode == "evaluate":
        evaluate_sb3_model(
            model_path=args.model_path,
            env_config_path=args.env_config,
            num_episodes=args.num_episodes,
            render=args.render
        )


if __name__ == "__main__":
    main() 