#!/usr/bin/env python3

import sys
import numpy as np
import torch
from pathlib import Path

# Add src directory to path for clean imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
sys.path.insert(0, str(src_dir))

from swarm.inference.model_loader import load_model, create_env


def run_inference_with_critic(algo, max_steps=50, deterministic=True):
    """
    Run inference episode and print both policy actions and critic values.
    
    Args:
        algo: Loaded RLlib algorithm
        max_steps: Maximum number of steps per episode
        deterministic: Whether to use deterministic policy actions
    """
    env = create_env()
    
    try:
        obs, info = env.reset()
        print(f"Initial observation keys: {list(obs.keys())}")
        print("="*80)
        
        for step in range(max_steps):
            print(f"\n--- STEP {step + 1} ---")
            
            # Compute actions and critic values for all agents
            actions = {}
            for agent_id, agent_obs in obs.items():
                if agent_id in env.all_agent_ids:
                    policy_id = f"{agent_id.split('_')[0]}_policy"
                    print(f"\nAgent: {agent_id} (Policy: {policy_id})")
                    
                    # Get RLModule and compute forward pass
                    rl_module = algo.get_module(policy_id)
                    obs_tensor = torch.from_numpy(agent_obs).unsqueeze(0).float()
                    
                    # Forward inference for policy
                    result = rl_module.forward_inference({"obs": obs_tensor})
                    
                    # Extract action from distribution inputs
                    action_dist_inputs = result["action_dist_inputs"][0]
                    action_dim = action_dist_inputs.shape[0] // 2
                    mean = action_dist_inputs[:action_dim]
                    log_std = action_dist_inputs[action_dim:]

                    if deterministic:
                        action = mean
                    else:
                        std = torch.exp(log_std)
                        action = torch.normal(mean, std)

                    action = torch.clamp(action, -1.0, 1.0)
                    actions[agent_id] = action.item()
                    
                    print(f"  Policy action mean: {mean.item():.4f}")
                    print(f"  Policy action log_std: {log_std.item():.4f}")
                    print(f"  Final action: {action.item():.4f}")
                    
                    # Get critic value using proper RLModule API
                    try:
                        # Import required constants
                        from ray.rllib.core.columns import Columns
                        
                        # Prepare input dict with proper column name
                        input_dict = {Columns.OBS: obs_tensor}
                        
                        # Method 1: Direct value computation using compute_values()
                        if hasattr(rl_module, 'compute_values'):
                            value = rl_module.compute_values(input_dict)
                            print(f"  Critic value: {value.item():.4f}")
                        else:
                            # Method 2: Try forward_train which includes embeddings for value computation
                            train_result = rl_module.forward_train(input_dict)
                            if Columns.EMBEDDINGS in train_result and hasattr(rl_module, 'compute_values'):
                                value = rl_module.compute_values(
                                    input_dict, 
                                    embeddings=train_result[Columns.EMBEDDINGS]
                                )
                                print(f"  Critic value (with embeddings): {value.item():.4f}")
                            else:
                                print("  Critic value: Not available (use_critic=False or missing compute_values)")
                            
                    except Exception as e:
                        print(f"  Critic value: Error accessing - {e}")
                        # Fallback: check if model was loaded with inference_only=True
                        print("  Note: If using inference_only=True, value function may be excluded")
                        
                    # Print observation statistics
                    print(f"  Obs shape: {agent_obs.shape}")
                    print(f"  Obs mean: {agent_obs.mean():.4f}, std: {agent_obs.std():.4f}")
                    print(f"  Obs min: {agent_obs.min():.4f}, max: {agent_obs.max():.4f}")
            
            # Take step in environment
            print(f"\nActions taken: {actions}")
            obs, reward, terminated, truncated, info = env.step(actions)
            
            print(f"Rewards: {reward}")
            print(f"Terminated: {terminated}")
            print(f"Truncated: {truncated}")
            
            # Check if episode is done
            if terminated or truncated:
                print(f"\nEpisode ended at step {step + 1}")
                break
        
        print(f"\nEpisode completed after {step + 1} steps")
        
    except Exception as e:
        print(f"Error during episode: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()


def run_critic_inference_test(checkpoint_path=None, num_episodes=1, max_steps_per_episode=50):
    """
    Run inference test with critic value logging.
    
    Args:
        checkpoint_path: Path to checkpoint (None for latest)
        num_episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
    """
    print("Loading model...")
    algo = load_model(checkpoint_path)
    
    for episode in range(num_episodes):
        print(f"\n{'='*80}")
        print(f"EPISODE {episode + 1}/{num_episodes}")
        print(f"{'='*80}")
        
        # Run inference episode with critic logging
        run_inference_with_critic(algo, max_steps_per_episode)
    
    print(f"\nCritic inference testing complete.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run inference with critic value logging")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to checkpoint directory (default: latest)")
    parser.add_argument("--episodes", type=int, default=1,
                       help="Number of episodes to run (default: 1)")
    parser.add_argument("--max-steps", type=int, default=50,
                       help="Maximum steps per episode (default: 50)")
    
    args = parser.parse_args()
    
    try:
        run_critic_inference_test(
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