#!/usr/bin/env python3
"""
Test script for the QDARTS environment implementation.
"""

import numpy as np
import os
import sys

# Add the current directory to the path
sys.path.append(os.path.dirname(__file__))

from qdarts_env import QDARTSEnv

def test_qdarts_env():
    """
    Test the QDARTS environment implementation.
    """
    print("Testing QDARTS Environment...")
    
    try:
        # Create the environment
        env = QDARTSEnv(config_path='qdarts_env_config.yaml', render_mode=None)
        print("✅ Environment created successfully")
        
        # Test reset
        observation, info = env.reset()
        print("✅ Environment reset successfully")
        print(f"   Observation keys: {list(observation.keys())}")
        print(f"   Image shape: {observation['image'].shape}")
        print(f"   Voltage shape: {observation['voltages'].shape}")
        
        # Test action space
        print(f"✅ Action space: {env.action_space}")
        print(f"   Action space shape: {env.action_space.shape}")
        print(f"   Action space low: {env.action_space.low}")
        print(f"   Action space high: {env.action_space.high}")
        
        # Test observation space
        print(f"✅ Observation space: {env.observation_space}")
        
        # Test step
        action = np.array([0.1, 0.2, 0.0, 0.0, 0.0])  # Small action
        observation, reward, terminated, truncated, info = env.step(action)
        print("✅ Environment step successful")
        print(f"   Reward: {reward}")
        print(f"   Terminated: {terminated}")
        print(f"   Truncated: {truncated}")
        
        # Test multiple steps
        for i in range(5):
            action = env.action_space.sample()  # Random action
            observation, reward, terminated, truncated, info = env.step(action)
            print(f"   Step {i+1}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
            
            if terminated or truncated:
                break
        
        # Test render
        try:
            rendered = env.render()
            print("✅ Environment render successful")
        except Exception as e:
            print(f"⚠️  Render failed: {e}")
        
        # Test close
        env.close()
        print("✅ Environment closed successfully")
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_qdarts_env()
    sys.exit(0 if success else 1) 