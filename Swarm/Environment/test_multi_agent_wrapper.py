#!/usr/bin/env python3
"""
Comprehensive test script for MultiAgentQuantumWrapper.

This script tests the multi-agent wrapper by simulating multiple agents
interacting with the quantum device environment, including reset, step,
and observation extraction functionality.
"""

import sys
import os
import numpy as np
import traceback
from typing import Dict, Any

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from multi_agent_wrapper import MultiAgentQuantumWrapper
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running this from the Environment directory")
    sys.exit(1)


class MultiAgentTester:
    """Test class for multi-agent wrapper functionality."""
    
    def __init__(self, num_quantum_dots: int = 4):
        """Initialize tester with specified number of quantum dots."""
        self.num_quantum_dots = num_quantum_dots
        self.wrapper = None
        
    def setup_environment(self):
        """Create multi-agent wrapper (which creates base environment internally)."""
        print(f"🔧 Setting up environment with {self.num_quantum_dots} quantum dots...")
        
        try:            
            # Create multi-agent wrapper (instantiates base env internally)
            self.wrapper = MultiAgentQuantumWrapper(
                num_quantum_dots=self.num_quantum_dots,
                training=True
            )
            print("✓ Created MultiAgentQuantumWrapper with internal QuantumDeviceEnv")
            
            return True
            
        except Exception as e:
            print(f"✗ Environment setup failed: {e}")
            traceback.print_exc()
            return False
    
    def test_agent_configuration(self):
        """Test agent ID configuration and space setup."""
        print("\n📋 Testing agent configuration...")
        
        try:
            # Check agent IDs
            agent_ids = self.wrapper.get_agent_ids()
            print(f"✓ Total agents: {len(agent_ids)}")
            print(f"  Gate agents: {self.wrapper.gate_agent_ids}")
            print(f"  Barrier agents: {self.wrapper.barrier_agent_ids}")
            
            # Verify agent counts
            expected_gates = self.num_quantum_dots
            expected_barriers = self.num_quantum_dots - 1
            
            assert len(self.wrapper.gate_agent_ids) == expected_gates, \
                f"Expected {expected_gates} gate agents, got {len(self.wrapper.gate_agent_ids)}"
            assert len(self.wrapper.barrier_agent_ids) == expected_barriers, \
                f"Expected {expected_barriers} barrier agents, got {len(self.wrapper.barrier_agent_ids)}"
            
            print("✓ Agent counts verified")
            
            # Check observation and action spaces
            print("\n🔍 Observation and action spaces:")
            for agent_id in agent_ids[:3]:  # Show first 3 agents
                obs_space = self.wrapper.observation_spaces[agent_id]
                action_space = self.wrapper.action_spaces[agent_id]
                
                print(f"  {agent_id}:")
                print(f"    Obs space: {obs_space}")
                print(f"    Action space: {action_space}")
                
            return True
            
        except Exception as e:
            print(f"✗ Agent configuration test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_environment_reset(self):
        """Test environment reset and observation extraction."""
        print("\n🔄 Testing environment reset...")
        
        try:
            # Reset environment
            observations, infos = self.wrapper.reset(seed=42)
            print(f"✓ Reset successful - got observations for {len(observations)} agents")
            
            # Verify all agents have observations
            agent_ids = self.wrapper.get_agent_ids()
            missing_agents = set(agent_ids) - set(observations.keys())
            if missing_agents:
                print(f"⚠️  Missing observations for agents: {missing_agents}")
            else:
                print("✓ All agents have observations")
            
            # Check observation shapes and values
            print("\n🔬 Observation details:")
            for agent_id in agent_ids[:4]:  # Check first 4 agents
                if agent_id in observations:
                    obs = observations[agent_id]
                    print(f"  {agent_id}:")
                    print(f"    Shape: {obs.shape}")
                    print(f"    Type: {obs.dtype}")
                    print(f"    Range: [{obs.min():.3f}, {obs.max():.3f}]")
                    print(f"    Channels: {obs.shape[2] if len(obs.shape) > 2 else 'N/A'}")
                    
                    # Verify channel assignments
                    channels = self.wrapper.agent_channel_map.get(agent_id, [])
                    print(f"    Assigned channels: {channels}")
            
            # Store observations for next test
            self.last_observations = observations
            self.last_infos = infos
            
            return True
            
        except Exception as e:
            print(f"✗ Environment reset test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_agent_actions(self):
        """Test agent action sampling and combination."""
        print("\n🎯 Testing agent actions...")
        
        try:
            agent_ids = self.wrapper.get_agent_ids()
            
            # Sample random actions for all agents
            agent_actions = {}
            print("🎲 Sampling random actions:")
            
            for agent_id in agent_ids:
                action_space = self.wrapper.action_spaces[agent_id]
                action = action_space.sample()
                agent_actions[agent_id] = action
                
                print(f"  {agent_id}: {action} (bounds: [{action_space.low[0]:.2f}, {action_space.high[0]:.2f}])")
            
            # Test action combination
            combined_action = self.wrapper._combine_agent_actions(agent_actions)
            print(f"\n🔗 Combined global action:")
            print(f"  Gate voltages: {combined_action['action_gate_voltages']}")
            print(f"  Barrier voltages: {combined_action['action_barrier_voltages']}")
            
            # Verify action dimensions
            expected_gates = self.num_quantum_dots
            expected_barriers = self.num_quantum_dots - 1
            
            assert len(combined_action['action_gate_voltages']) == expected_gates, \
                f"Expected {expected_gates} gate voltages, got {len(combined_action['action_gate_voltages'])}"
            assert len(combined_action['action_barrier_voltages']) == expected_barriers, \
                f"Expected {expected_barriers} barrier voltages, got {len(combined_action['action_barrier_voltages'])}"
            
            print("✓ Action combination verified")
            
            # Store actions for next test
            self.test_actions = agent_actions
            
            return True
            
        except Exception as e:
            print(f"✗ Agent actions test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_environment_step(self):
        """Test environment step with multi-agent actions."""
        print("\n👟 Testing environment step...")
        
        try:
            if not hasattr(self, 'test_actions'):
                print("⚠️  No test actions available, skipping step test")
                return False
            
            # Perform environment step
            obs, rewards, terminated, truncated, infos = self.wrapper.step(self.test_actions)
            
            print(f"✓ Step successful - got observations for {len(obs)} agents")
            print(f"✓ Got rewards for {len(rewards)} agents")
            
            # Check reward distribution
            print("\n💰 Reward details:")
            agent_ids = self.wrapper.get_agent_ids()
            
            gate_rewards = []
            barrier_rewards = []
            
            for agent_id in agent_ids:
                if agent_id in rewards:
                    reward = rewards[agent_id]
                    agent_type = "Gate" if "plunger" in agent_id else "Barrier"
                    print(f"  {agent_id} ({agent_type}): {reward:.4f}")
                    
                    if "plunger" in agent_id:
                        gate_rewards.append(reward)
                    else:
                        barrier_rewards.append(reward)
            
            if gate_rewards:
                print(f"  Gate reward stats: mean={np.mean(gate_rewards):.4f}, std={np.std(gate_rewards):.4f}")
            if barrier_rewards:
                print(f"  Barrier reward stats: mean={np.mean(barrier_rewards):.4f}, std={np.std(barrier_rewards):.4f}")
            
            # Check termination flags
            print(f"\n🏁 Termination status:")
            print(f"  Terminated: {terminated}")
            print(f"  Truncated: {truncated}")
            print(f"  __all__ terminated: {terminated.get('__all__', 'Missing!')}")
            print(f"  __all__ truncated: {truncated.get('__all__', 'Missing!')}")
            
            # Verify observation consistency after step
            print("\n🔄 Post-step observation verification:")
            for agent_id in agent_ids[:2]:  # Check first 2 agents
                if agent_id in obs:
                    new_obs = obs[agent_id]
                    old_obs = self.last_observations.get(agent_id)
                    
                    print(f"  {agent_id}:")
                    print(f"    New shape: {new_obs.shape}")
                    print(f"    Old shape: {old_obs.shape if old_obs is not None else 'N/A'}")
                    
                    if old_obs is not None:
                        # Check if observations changed (they should after environment step)
                        obs_changed = not np.allclose(new_obs, old_obs, atol=1e-6)
                        print(f"    Observation changed: {obs_changed}")
            
            return True
            
        except Exception as e:
            print(f"✗ Environment step test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_channel_assignments(self):
        """Test channel assignment logic for different agent types."""
        print("\n📺 Testing channel assignments...")
        
        try:
            print("Channel assignment mapping:")
            for agent_id in self.wrapper.all_agent_ids:
                channels = self.wrapper.agent_channel_map[agent_id]
                agent_type = "Gate" if "plunger" in agent_id else "Barrier"
                print(f"  {agent_id} ({agent_type}): channels {channels}")
            
            # Verify channel assignment logic
            for i, agent_id in enumerate(self.wrapper.gate_agent_ids):
                expected_channels = []
                if i == 0:
                    # First gate: [0, 0]
                    expected_channels = [0, 0]
                elif i == len(self.wrapper.gate_agent_ids) - 1:
                    # Last gate: [N-2, N-2]
                    last_channel = self.num_quantum_dots - 2
                    expected_channels = [last_channel, last_channel]
                else:
                    # Middle gates: [i-1, i]
                    expected_channels = [i-1, i]
                
                actual_channels = self.wrapper.agent_channel_map[agent_id]
                assert actual_channels == expected_channels, \
                    f"Gate {agent_id}: expected {expected_channels}, got {actual_channels}"
            
            print("✓ Gate channel assignments verified")
            
            # Verify barrier channel assignments
            for i, agent_id in enumerate(self.wrapper.barrier_agent_ids):
                expected_channels = [i]
                actual_channels = self.wrapper.agent_channel_map[agent_id]
                assert actual_channels == expected_channels, \
                    f"Barrier {agent_id}: expected {expected_channels}, got {actual_channels}"
            
            print("✓ Barrier channel assignments verified")
            
            return True
            
        except Exception as e:
            print(f"✗ Channel assignment test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_multiple_episodes(self, num_episodes: int = 3):
        """Test multiple episode runs to check for consistency."""
        print(f"\n🔁 Testing {num_episodes} episodes...")
        
        try:
            for episode in range(num_episodes):
                print(f"\n--- Episode {episode + 1} ---")
                
                # Reset for new episode
                obs, infos = self.wrapper.reset(seed=episode * 100)
                print(f"Reset: {len(obs)} agent observations")
                
                # Run a few steps
                for step in range(3):
                    # Sample new actions
                    actions = {}
                    for agent_id in self.wrapper.get_agent_ids():
                        actions[agent_id] = self.wrapper.action_spaces[agent_id].sample()
                    
                    # Step environment
                    obs, rewards, terminated, truncated, infos = self.wrapper.step(actions)
                    
                    total_reward = sum(rewards.values())
                    print(f"  Step {step + 1}: Total reward = {total_reward:.4f}")
                    
                    # Break if episode ended
                    if terminated.get("__all__", False) or truncated.get("__all__", False):
                        print(f"    Episode ended at step {step + 1}")
                        break
            
            print("✓ Multiple episodes completed successfully")
            return True
            
        except Exception as e:
            print(f"✗ Multiple episodes test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_observation_extraction_details(self):
        """Test detailed observation extraction for debugging."""
        print("\n🔍 Testing detailed observation extraction...")
        
        try:
            # Get fresh observations from base environment
            global_obs, _ = self.wrapper.base_env.reset(seed=123)
            print("Base environment global observation structure:")
            for key, value in global_obs.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}, range=[{value.min():.3f}, {value.max():.3f}]")
                else:
                    print(f"  {key}: {type(value)} = {value}")
            
            print("\nTesting agent observation extraction:")
            
            # Test each agent type
            for agent_id in self.wrapper.all_agent_ids[:4]:  # Test first 4 agents
                try:
                    agent_obs = self.wrapper._extract_agent_observation(global_obs, agent_id)
                    agent_type = "Gate" if "plunger" in agent_id else "Barrier"
                    channels = self.wrapper.agent_channel_map[agent_id]
                    
                    print(f"\n  {agent_id} ({agent_type}):")
                    print(f"    Assigned channels: {channels}")
                    print(f"    Observation shape: {agent_obs.shape}")
                    print(f"    Observation range: [{agent_obs.min():.3f}, {agent_obs.max():.3f}]")
                    
                    # Verify expected shapes
                    expected_height, expected_width = global_obs['image'].shape[:2]
                    expected_channels = len(channels) if len(channels) > 1 else 1
                    
                    assert agent_obs.shape == (expected_height, expected_width, expected_channels), \
                        f"Expected shape ({expected_height}, {expected_width}, {expected_channels}), got {agent_obs.shape}"
                    
                    print(f"    ✓ Shape verified")
                    
                except Exception as e:
                    print(f"    ✗ Failed to extract observation for {agent_id}: {e}")
                    return False
            
            print("\n✓ All observation extractions successful")
            return True
            
        except Exception as e:
            print(f"✗ Observation extraction test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_error_scenarios(self):
        """Test error handling and edge cases."""
        print("\n⚠️  Testing error scenarios...")
        
        try:
            # Test with incomplete agent actions
            print("Testing with incomplete agent actions...")
            incomplete_actions = {
                self.wrapper.gate_agent_ids[0]: np.array([0.1])  # Only one agent
            }
            
            try:
                obs, rewards, terminated, truncated, infos = self.wrapper.step(incomplete_actions)
                print("  ⚠️  Incomplete actions accepted (may be intentional)")
            except Exception as e:
                print(f"  ✓ Incomplete actions rejected: {e}")
            
            # Test with invalid agent ID
            print("Testing with invalid agent ID...")
            invalid_actions = {
                "invalid_agent": np.array([0.1])
            }
            
            try:
                obs, rewards, terminated, truncated, infos = self.wrapper.step(invalid_actions)
                print("  ⚠️  Invalid agent ID accepted")
            except Exception as e:
                print(f"  ✓ Invalid agent ID rejected: {e}")
            
            # Test with out-of-bounds actions
            print("Testing with out-of-bounds actions...")
            oob_actions = {}
            for agent_id in self.wrapper.get_agent_ids():
                action_space = self.wrapper.action_spaces[agent_id]
                # Action way outside bounds
                oob_actions[agent_id] = np.array([action_space.high[0] * 10])
            
            try:
                obs, rewards, terminated, truncated, infos = self.wrapper.step(oob_actions)
                print("  ⚠️  Out-of-bounds actions accepted (may be clipped)")
            except Exception as e:
                print(f"  ✓ Out-of-bounds actions rejected: {e}")
            
            return True
            
        except Exception as e:
            print(f"✗ Error scenario test failed: {e}")
            traceback.print_exc()
            return False
    
    def test_reward_distribution(self):
        """Test reward distribution logic in detail."""
        print("\n💎 Testing reward distribution...")
        
        try:
            # Create mock global rewards
            mock_global_rewards = {
                'gates': np.random.uniform(-1, 1, self.num_quantum_dots),
                'barriers': np.random.uniform(-1, 1, self.num_quantum_dots - 1)
            }
            
            print("Mock global rewards:")
            print(f"  Gates: {mock_global_rewards['gates']}")
            print(f"  Barriers: {mock_global_rewards['barriers']}")
            
            # Test reward distribution
            agent_rewards = self.wrapper._distribute_rewards(mock_global_rewards)
            
            print("\nDistributed agent rewards:")
            gate_total = 0
            barrier_total = 0
            
            for agent_id, reward in agent_rewards.items():
                agent_type = "Gate" if "plunger" in agent_id else "Barrier"
                print(f"  {agent_id} ({agent_type}): {reward:.4f}")
                
                if "plunger" in agent_id:
                    gate_total += reward
                else:
                    barrier_total += reward
            
            # Verify totals match
            expected_gate_total = np.sum(mock_global_rewards['gates'])
            expected_barrier_total = np.sum(mock_global_rewards['barriers'])
            
            print(f"\nReward totals verification:")
            print(f"  Gate total: {gate_total:.4f} (expected: {expected_gate_total:.4f})")
            print(f"  Barrier total: {barrier_total:.4f} (expected: {expected_barrier_total:.4f})")
            
            assert abs(gate_total - expected_gate_total) < 1e-6, \
                f"Gate reward total mismatch: {gate_total} vs {expected_gate_total}"
            assert abs(barrier_total - expected_barrier_total) < 1e-6, \
                f"Barrier reward total mismatch: {barrier_total} vs {expected_barrier_total}"
            
            print("✓ Reward distribution verified")
            return True
            
        except Exception as e:
            print(f"✗ Reward distribution test failed: {e}")
            traceback.print_exc()
            return False
    
    def cleanup(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up...")
        try:
            if self.wrapper:
                self.wrapper.close()
            print("✓ Cleanup completed")
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")


def main():
    """Run all tests for the multi-agent wrapper."""
    print("=" * 60)
    print("🧪 MULTI-AGENT WRAPPER COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    # Test with different quantum dot configurations
    test_configs = [6]  # Different numbers of quantum dots
    
    all_passed = True
    
    for num_dots in test_configs:
        print(f"\n🎯 TESTING WITH {num_dots} QUANTUM DOTS")
        print("-" * 40)
        
        tester = MultiAgentTester(num_quantum_dots=num_dots)
        
        # Run test suite
        tests = [
            tester.setup_environment,
            tester.test_agent_configuration,
            tester.test_channel_assignments,
            tester.test_environment_reset,
            tester.test_observation_extraction_details,
            tester.test_agent_actions,
            tester.test_environment_step,
            tester.test_reward_distribution,
            tester.test_error_scenarios,
            tester.test_multiple_episodes,
        ]
        
        config_passed = True
        for test in tests:
            if not test():
                config_passed = False
                all_passed = False
                break
        
        if config_passed:
            print(f"\n✅ ALL TESTS PASSED for {num_dots} quantum dots!")
        else:
            print(f"\n❌ TESTS FAILED for {num_dots} quantum dots!")
        
        # Cleanup
        tester.cleanup()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL CONFIGURATIONS PASSED!")
    else:
        print("💥 SOME TESTS FAILED - CHECK OUTPUT ABOVE")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)