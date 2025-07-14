import sys
import os
import argparse
import yaml

# Add parent directory to path to import environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import QuantumDeviceEnv
try:
    from ppo_agent import PPOAgent
except ImportError:
    # Fallback for direct script execution
    from agent.ppo_agent import PPOAgent


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on quantum device environment")
    parser.add_argument("--config", type=str, default="config/ppo_config.yaml",
                       help="Path to PPO configuration file")
    parser.add_argument("--env-config", type=str, default="../env_config.yaml",
                       help="Path to environment configuration file")
    parser.add_argument("--render", action="store_true",
                       help="Render environment during training")
    parser.add_argument("--eval-only", action="store_true",
                       help="Only evaluate a trained model")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to model to load for evaluation")
    parser.add_argument("--num-eval-episodes", type=int, default=10,
                       help="Number of episodes for evaluation")
    
    args = parser.parse_args()
    
    # Load environment
    print("Initializing quantum device environment...")
    env = QuantumDeviceEnv(config_path=args.env_config, render_mode="human" if args.render else None)
    
    # Initialize agent
    print("Initializing PPO agent...")
    agent = PPOAgent(env, config_path=args.config)
    
    if args.eval_only:
        # Evaluation mode
        if args.model_path:
            print(f"Loading model from {args.model_path}...")
            agent.load_model(args.model_path)
        
        print(f"Evaluating agent for {args.num_eval_episodes} episodes...")
        results = agent.evaluate(num_episodes=args.num_eval_episodes)
        
        print("Evaluation Results:")
        print(f"  Mean reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
        print(f"  Mean episode length: {results['mean_length']:.1f}")
        print(f"  Individual episode rewards: {results['episode_rewards']}")
        
    else:
        # Training mode
        print("Starting training...")
        try:
            agent.train()
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            print("Saving current model...")
            agent.save_model("interrupted_model.pth")
        
        # Final evaluation
        print("\nFinal evaluation...")
        results = agent.evaluate(num_episodes=args.num_eval_episodes)
        print(f"Final mean reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    
    env.close()
    print("Done!")


if __name__ == "__main__":
    main() 