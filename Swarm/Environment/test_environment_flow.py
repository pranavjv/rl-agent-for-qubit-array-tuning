#!/usr/bin/env python3
"""
Comprehensive test for the environment flow between env.py and qarray_base_class.py
Tests the complete interaction workflow from initialization to step execution.
"""

import numpy as np
import sys
import os
import traceback
from typing import Dict, Any

# Add the parent directory to the path to enable proper imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the environment
from Environment.env import QuantumDeviceEnv


def test_environment_initialization():
    """Test environment initialization and configuration loading"""
    print("=" * 60)
    print("TESTING ENVIRONMENT INITIALIZATION")
    print("=" * 60)
    
    try:
        # Initialize environment with training mode
        env = QuantumDeviceEnv(training=True)
        
        print(f"✓ Environment initialized successfully")
        print(f"  - Number of dots: {env.num_dots}")
        print(f"  - Max steps: {env.max_steps}")
        print(f"  - Tolerance: {env.tolerance}")
        print(f"  - Debug mode: {env.debug}")
        print(f"  - Training mode: {env.training}")
        
        # Check array initialization
        print(f"  - Array voltage range: [{env.obs_voltage_min}, {env.obs_voltage_max}]")
        print(f"  - Array image size: {env.obs_image_size}")
        
        # Check action/observation spaces
        print(f"  - Gate voltage action space: {env.action_space['action_gate_voltages'].shape}")
        print(f"  - Barrier voltage action space: {env.action_space['action_barrier_voltages'].shape}")
        print(f"  - Image observation space: {env.observation_space['image'].shape}")
        
        return env
        
    except Exception as e:
        print(f"✗ Environment initialization failed: {e}")
        traceback.print_exc()
        return None


def test_reset_functionality(env: QuantumDeviceEnv):
    """Test environment reset functionality"""
    print("\n" + "=" * 60)
    print("TESTING RESET FUNCTIONALITY")
    print("=" * 60)
    
    try:
        # Test reset with seed
        obs, info = env.reset(seed=42)
        
        print(f"✓ Reset completed successfully")
        print(f"  - Current step after reset: {env.current_step}")
        print(f"  - Observation keys: {list(obs.keys())}")
        print(f"  - Image shape: {obs['image'].shape}")
        print(f"  - Gate voltages shape: {obs['obs_gate_voltages'].shape}")
        print(f"  - Barrier voltages shape: {obs['obs_barrier_voltages'].shape}")
        print(f"  - Image value range: [{obs['image'].min():.4f}, {obs['image'].max():.4f}]")
        
        # Check device state
        print(f"  - Ground truth gates shape: {env.device_state['gate_ground_truth'].shape}")
        print(f"  - Current gate voltages: {env.device_state['current_gate_voltages']}")
        print(f"  - Virtual gate matrix shape: {env.device_state['virtual_gate_matrix'].shape}")
        
        # Test reset without seed
        obs2, info2 = env.reset()
        print(f"✓ Reset without seed completed")
        
        # Check if observations are different (should be due to randomization)
        gate_diff = np.allclose(obs['obs_gate_voltages'], obs2['obs_gate_voltages'])
        print(f"  - Gate voltages identical after reset: {gate_diff}")
        
        return obs, info
        
    except Exception as e:
        print(f"✗ Reset functionality failed: {e}")
        traceback.print_exc()
        return None, None


def test_qarray_integration(env: QuantumDeviceEnv):
    """Test the qarray_base_class integration"""
    print("\n" + "=" * 60)
    print("TESTING QARRAY BASE CLASS INTEGRATION")
    print("=" * 60)
    
    try:
        # Test direct qarray method calls
        print(f"✓ QarrayBaseClass instance exists: {env.array is not None}")
        print(f"  - Array num_dots: {env.array.num_dots}")
        print(f"  - Array image size: {env.array.obs_image_size}")
        print(f"  - Array channels: {env.array.obs_channels}")
        
        # Test ground truth calculation
        gt = env.array.calculate_ground_truth()
        print(f"  - Ground truth shape: {gt.shape}")
        print(f"  - Ground truth values: {gt}")
        
        # Test observation generation
        test_gates = np.zeros(env.num_dots, dtype=np.float32)
        test_barriers = np.zeros(env.num_dots - 1, dtype=np.float32)
        raw_obs = env.array._get_obs(test_gates, test_barriers)
        
        print(f"  - Raw observation keys: {list(raw_obs.keys())}")
        print(f"  - Raw image shape: {raw_obs['image'].shape}")
        print(f"  - Raw image value range: [{raw_obs['image'].min():.4f}, {raw_obs['image'].max():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"✗ QArray integration test failed: {e}")
        traceback.print_exc()
        return False


def test_step_functionality(env: QuantumDeviceEnv):
    """Test environment step functionality"""
    print("\n" + "=" * 60)
    print("TESTING STEP FUNCTIONALITY")
    print("=" * 60)
    
    try:
        # Reset environment first
        obs, info = env.reset(seed=123)
        
        # Create a valid action
        action = {
            'action_gate_voltages': np.random.uniform(
                env.gate_voltage_min, env.gate_voltage_max, 
                size=(env.num_plunger_voltages,)
            ).astype(np.float32),
            'action_barrier_voltages': np.random.uniform(
                env.barrier_voltage_min, env.barrier_voltage_max,
                size=(env.num_barrier_voltages,)
            ).astype(np.float32)
        }
        
        print(f"  - Action gate voltages: {action['action_gate_voltages']}")
        print(f"  - Action barrier voltages: {action['action_barrier_voltages']}")
        
        # Take a step
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"✓ Step completed successfully")
        print(f"  - Step count: {env.current_step}")
        print(f"  - Reward keys: {list(reward.keys())}")
        print(f"  - Gate rewards: {reward['gates']}")
        print(f"  - Barrier rewards: {reward['barriers']}")
        print(f"  - Terminated: {terminated}")
        print(f"  - Truncated: {truncated}")
        print(f"  - Next observation shape: {next_obs['image'].shape}")
        
        # Test multiple steps
        step_count = 0
        total_rewards = {'gates': [], 'barriers': []}
        
        while not terminated and not truncated and step_count < 5:
            # Generate random action
            action = {
                'action_gate_voltages': np.random.uniform(
                    env.gate_voltage_min, env.gate_voltage_max, 
                    size=(env.num_plunger_voltages,)
                ).astype(np.float32),
                'action_barrier_voltages': np.random.uniform(
                    env.barrier_voltage_min, env.barrier_voltage_max,
                    size=(env.num_barrier_voltages,)
                ).astype(np.float32)
            }
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_rewards['gates'].append(reward['gates'])
            total_rewards['barriers'].append(reward['barriers'])
            step_count += 1
        
        print(f"  - Completed {step_count + 1} total steps")
        print(f"  - Final terminated state: {terminated}")
        print(f"  - Final truncated state: {truncated}")
        
        return True
        
    except Exception as e:
        print(f"✗ Step functionality test failed: {e}")
        traceback.print_exc()
        return False


def test_reward_computation(env: QuantumDeviceEnv):
    """Test reward computation logic"""
    print("\n" + "=" * 60)
    print("TESTING REWARD COMPUTATION")
    print("=" * 60)
    
    try:
        # Reset and get initial state
        obs, info = env.reset(seed=456)
        
        # Get ground truth
        gate_gt = env.device_state['gate_ground_truth']
        barrier_gt = env.device_state['barrier_ground_truth']
        
        print(f"  - Gate ground truth: {gate_gt}")
        print(f"  - Barrier ground truth: {barrier_gt}")
        
        # Test reward at different distances from ground truth
        test_cases = [
            ("At ground truth", gate_gt, barrier_gt if barrier_gt is not None else np.zeros(env.num_barrier_voltages)),
            ("Far from ground truth", 
             gate_gt + 0.5, 
             (barrier_gt + 0.1) if barrier_gt is not None else np.zeros(env.num_barrier_voltages)),
            ("Close to ground truth", 
             gate_gt + 0.01, 
             (barrier_gt + 0.01) if barrier_gt is not None else np.zeros(env.num_barrier_voltages))
        ]
        
        for case_name, test_gates, test_barriers in test_cases:
            # Set device state manually
            env.device_state['current_gate_voltages'] = test_gates
            env.device_state['current_barrier_voltages'] = test_barriers
            
            reward, at_target = env._get_reward()
            
            print(f"  - {case_name}:")
            print(f"    Gate rewards: {reward['gates']}")
            print(f"    Barrier rewards: {reward['barriers']}")
            print(f"    At target: {at_target}")
        
        return True
        
    except Exception as e:
        print(f"✗ Reward computation test failed: {e}")
        traceback.print_exc()
        return False


def test_observation_normalization(env: QuantumDeviceEnv):
    """Test observation normalization"""
    print("\n" + "=" * 60)
    print("TESTING OBSERVATION NORMALIZATION")
    print("=" * 60)
    
    try:
        # Reset environment
        obs, info = env.reset(seed=789)
        
        # Get raw observation
        test_gates = env.device_state['current_gate_voltages']
        test_barriers = env.device_state['current_barrier_voltages']
        raw_obs = env.array._get_obs(test_gates, test_barriers)
        
        print(f"  - Raw image range: [{raw_obs['image'].min():.4f}, {raw_obs['image'].max():.4f}]")
        print(f"  - Raw image mean: {raw_obs['image'].mean():.4f}")
        print(f"  - Raw image std: {raw_obs['image'].std():.4f}")
        
        # Apply normalization
        normalized_obs = env._normalise_obs(raw_obs)
        
        print(f"  - Normalized image range: [{normalized_obs['image'].min():.4f}, {normalized_obs['image'].max():.4f}]")
        print(f"  - Normalized image mean: {normalized_obs['image'].mean():.4f}")
        print(f"  - Normalized image std: {normalized_obs['image'].std():.4f}")
        print(f"  - Normalization successful: {0.0 <= normalized_obs['image'].min() and normalized_obs['image'].max() <= 1.0}")
        
        return True
        
    except Exception as e:
        print(f"✗ Observation normalization test failed: {e}")
        traceback.print_exc()
        return False


def run_complete_episode(env: QuantumDeviceEnv):
    """Run a complete episode to test the full workflow"""
    print("\n" + "=" * 60)
    print("RUNNING COMPLETE EPISODE")
    print("=" * 60)
    
    try:
        obs, info = env.reset(seed=999)
        episode_rewards = {'gates': [], 'barriers': []}
        step_count = 0
        
        print(f"  - Episode started")
        print(f"  - Initial ground truth: {env.device_state['gate_ground_truth']}")
        
        while True:
            # Generate action (try to move towards ground truth)
            gt = env.device_state['gate_ground_truth']
            current = env.device_state['current_gate_voltages']
            
            # Simple policy: move 10% towards ground truth
            direction = gt - current
            action_gates = current + 0.1 * direction
            
            # Clip to action space bounds
            action_gates = np.clip(action_gates, env.gate_voltage_min, env.gate_voltage_max)
            
            action = {
                'action_gate_voltages': action_gates.astype(np.float32),
                'action_barrier_voltages': np.random.uniform(
                    env.barrier_voltage_min, env.barrier_voltage_max,
                    size=(env.num_barrier_voltages,)
                ).astype(np.float32)
            }
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_rewards['gates'].append(reward['gates'])
            episode_rewards['barriers'].append(reward['barriers'])
            step_count += 1
            
            if step_count % 10 == 0 or terminated or truncated:
                distance = np.linalg.norm(env.device_state['gate_ground_truth'] - 
                                        env.device_state['current_gate_voltages'])
                print(f"  - Step {step_count}: distance to GT = {distance:.4f}")
            
            if terminated or truncated:
                break
                
            if step_count >= env.max_steps:
                break
        
        print(f"✓ Episode completed")
        print(f"  - Total steps: {step_count}")
        print(f"  - Terminated: {terminated}")
        print(f"  - Truncated: {truncated}")
        print(f"  - Final distance to ground truth: {np.linalg.norm(env.device_state['gate_ground_truth'] - env.device_state['current_gate_voltages']):.4f}")
        print(f"  - Final gate rewards: {reward['gates']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Complete episode test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("COMPREHENSIVE ENVIRONMENT FLOW TEST")
    print("Testing interaction between env.py and qarray_base_class.py")
    print("=" * 80)
    
    # Test results tracking
    results = {}
    
    # Test environment initialization
    env = test_environment_initialization()
    results['initialization'] = env is not None
    
    if env is None:
        print("\n✗ Cannot proceed with further tests - initialization failed")
        return
    
    # Test reset functionality
    obs, info = test_reset_functionality(env)
    results['reset'] = obs is not None
    
    # Test qarray integration
    results['qarray_integration'] = test_qarray_integration(env)
    
    # Test step functionality
    results['step'] = test_step_functionality(env)
    
    # Test reward computation
    results['reward'] = test_reward_computation(env)
    
    # Test observation normalization
    results['normalization'] = test_observation_normalization(env)
    
    # Run complete episode
    results['complete_episode'] = run_complete_episode(env)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"  {test_name:20}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All environment flow tests passed!")
    else:
        print("⚠️  Some tests failed - check output above for details")


if __name__ == "__main__":
    main()