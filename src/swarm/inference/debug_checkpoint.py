#!/usr/bin/env python3
"""
Debug checkpoint loading issues.
"""
import sys
from pathlib import Path
import ray
from ray.tune.registry import register_env

# Add src directory to path for clean imports
current_dir = Path(__file__).parent
src_dir = current_dir.parent.parent
sys.path.insert(0, str(src_dir))

def create_env(config=None):
    """Create multi-agent quantum environment."""
    from swarm.environment.multi_agent_wrapper import MultiAgentQuantumWrapper
    return MultiAgentQuantumWrapper(training=False)

def debug_checkpoint_loading():
    """Debug checkpoint loading step by step."""
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(include_dashboard=False, log_to_driver=False)
    
    # Register environment
    register_env("qarray_multiagent_env", create_env)
    
    # Try to understand checkpoint structure
    checkpoint_path = Path(__file__).parent.parent / "training" / "checkpoints" / "iteration_100"
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Checkpoint exists: {checkpoint_path.exists()}")
    
    try:
        # Import necessary modules first
        from swarm.voltage_model import create_rl_module_spec
        
        # Try loading with debug info
        from ray.rllib.algorithms.algorithm import Algorithm
        print("Attempting to load checkpoint...")
        
        algo = Algorithm.from_checkpoint(str(checkpoint_path.absolute()))
        print("Successfully loaded checkpoint!")
        return algo
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    debug_checkpoint_loading()