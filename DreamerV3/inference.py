import os
import sys
import pathlib
import argparse
import numpy as np
from functools import partial as bind

# Add DreamerV3 paths
folder = pathlib.Path(__file__).parent
sys.path.insert(0, str(folder))
sys.path.insert(1, str(folder / 'dreamerv3'))

import elements
import embodied
import ruamel.yaml as yaml
from dreamerv3.agent import Agent

from qarray_env import QuantumDeviceEnv


def load_config(config_path='dreamerv3/configs.yaml'):
    """Load DreamerV3 configuration."""
    configs = elements.Path(config_path).read()
    configs = yaml.YAML(typ='safe').load(configs)
    config = elements.Config(configs['defaults'])
    return config


def make_env(config, render_mode='rgb_array'):
    """Create the quantum environment."""
    from embodied.envs import from_gym
    
    # Create environment with rgb_array render mode for image capture
    env = from_gym.FromGym(QuantumDeviceEnv(render_mode=render_mode))
    
    # Apply wrappers
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.UnifyDtypes(env)
    env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    
    return env


def load_agent(config, checkpoint_path):
    """Load the trained DreamerV3 agent."""
    # Create a temporary environment to get observation and action spaces
    env = make_env(config)
    notlog = lambda k: not k.startswith('log/')
    obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
    env.close()
    
    # Create agent
    agent = Agent(obs_space, act_space, elements.Config(
        **config.agent,
        logdir=config.logdir,
        seed=config.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
    ))
    
    # Load checkpoint
    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(checkpoint_path, keys=['agent'])
    
    return agent


def run_inference(checkpoint_path, num_episodes, max_steps_per_episode, 
                 output_dir='inference_frames', config_path='dreamerv3/configs.yaml'):
    """Run inference and save frames."""
    
    print(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Override some config settings for inference
    config = config.update({
        'task': 'custom_qarray',
        'jax': {'platform': 'cpu', 'debug': False, 'prealloc': False},
        'logdir': '/tmp/inference_logdir',
    })
    
    print(f"Loading agent from checkpoint: {checkpoint_path}")
    agent = load_agent(config, checkpoint_path)
    
    print("Creating environment...")
    
    # Create driver for proper environment interaction
    make_env_fn = lambda: make_env(config, render_mode='rgb_array')
    driver = embodied.Driver([make_env_fn], parallel=False)
    
    print(f"Starting inference for {num_episodes} episodes...")
    print(f"Frames will be saved to: {output_dir}")
    
    total_steps = 0
    episodes_completed = 0
    
    # Storage for frame saving
    frame_data = {'step': 0, 'output_dir': output_dir}
    
    def save_frame_callback(tran, worker):
        """Callback to save frames during episode"""
        if worker == 0:  # Only save from first worker
            # Save frame from current environment state
            env = driver.envs[0] if not driver.parallel else None
            if env:
                # Traverse the wrapper chain to find the underlying QuantumDeviceEnv
                current_env = env
                quantum_env = None
                
                # Look for the QuantumDeviceEnv through the wrapper chain
                while current_env is not None:
                    if hasattr(current_env, '_env') and hasattr(current_env._env, '_render_frame'):
                        quantum_env = current_env._env
                        break
                    elif hasattr(current_env, 'env') and hasattr(current_env.env, '_render_frame'):
                        quantum_env = current_env.env
                        break
                    elif hasattr(current_env, '_render_frame'):
                        quantum_env = current_env
                        break
                    
                    # Move to next level in wrapper chain
                    if hasattr(current_env, 'env'):
                        current_env = current_env.env
                    elif hasattr(current_env, '_env'):
                        current_env = current_env._env
                    else:
                        break
                
                if quantum_env and hasattr(quantum_env, '_render_frame'):
                    frame = quantum_env._render_frame(inference_plot=True)
                    if frame is not None:
                        import matplotlib.pyplot as plt
                        import matplotlib.image as mpimg
                        
                        os.makedirs(frame_data['output_dir'], exist_ok=True)
                        frame_path = os.path.join(frame_data['output_dir'], f'step_{frame_data["step"]:04d}.png')
                        
                        if frame.ndim == 3:  # RGB image
                            mpimg.imsave(frame_path, frame)
                        else:  # Grayscale or other format
                            plt.imsave(frame_path, frame, cmap='viridis')
                        
                        print(f"Saved frame: {frame_path}")
                        frame_data['step'] += 1
                    else:
                        print(f"Warning: No frame available for step {frame_data['step']}")
                else:
                    print(f"Warning: Could not find QuantumDeviceEnv with _render_frame method")
    
    # Set up driver callbacks
    driver.on_step(save_frame_callback)
    
    # Use same policy pattern as eval_only script - direct agent.policy call
    policy = lambda *args: agent.policy(*args, mode='eval')
    
    # Reset driver
    driver.reset(agent.init_policy)
    
    # Run episodes
    episode_count = 0
    step_count = 0
    
    while episode_count < num_episodes:
        print(f"\n--- Episode {episode_count + 1}/{num_episodes} ---")
        new_dir = f"{output_dir}/episode_{episode_count+1}"
        os.makedirs(new_dir, exist_ok=True)
        frame_data['output_dir'] = new_dir
        frame_data['step'] = 0
        
        # Run until episode ends or max steps reached
        initial_step = step_count
        driver(policy, steps=max_steps_per_episode)
        
        episode_count += 1
        steps_this_episode = step_count - initial_step
        
        print(f"Episode {episode_count} completed:")
        print(f"  Steps taken: {steps_this_episode}")
    
    driver.close()
    print(f"\nInference completed! Total frames saved: {frame_data['step']}")
    print(f"Frames are located in: {os.path.abspath(output_dir)}")


def main():
    parser = argparse.ArgumentParser(description='Run DreamerV3 inference on QuantumDeviceEnv')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to the checkpoint directory (e.g., /root/logdir/20250721T102619/ckpt/20250721T112751F343270)')
    parser.add_argument('--episodes', type=int, default=1,
                       help='Number of episodes to run (default: 1)')
    parser.add_argument('--max-steps', type=int, default=50,
                       help='Maximum steps per episode (default: 50)')
    parser.add_argument('--output-dir', type=str, default='inference_frames',
                       help='Directory to save frames (default: inference_frames)')
    parser.add_argument('--config', type=str, default='dreamerv3/configs.yaml',
                       help='Path to DreamerV3 config file (default: dreamerv3/configs.yaml)')
    
    args = parser.parse_args()
    
    # Validate checkpoint path
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint path does not exist: {args.checkpoint}")
        sys.exit(1)
    
    # Validate config path
    if not os.path.exists(args.config):
        print(f"Error: Config file does not exist: {args.config}")
        sys.exit(1)
    
    try:
        run_inference(
            checkpoint_path=args.checkpoint,
            num_episodes=args.episodes,
            max_steps_per_episode=args.max_steps,
            output_dir=args.output_dir,
            config_path=args.config
        )
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
