"""
Multi-agent wrapper for QuantumDeviceEnv.

This wrapper converts the global observation/action spaces into individual agent spaces
and handles the conversion between single-agent actions and global environment actions.
"""

import os
import sys
from typing import Dict

import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# Add src directory to path for clean imports
from pathlib import Path
src_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_dir))

from swarm.environment.env import QuantumDeviceEnv


class MultiAgentEnvWrapper(MultiAgentEnv):
    """
    Multi-agent wrapper that converts global env to individual agent interactions.

    Each agent sees:
    - Gate agents: 2-channel image (corresponding to their dot pairs) + single voltage
    - Barrier agents: 1-channel image (corresponding to adjacent dots) + single voltage

    Each agent outputs:
    - A single voltage value (gate or barrier)

    The wrapper combines individual agent actions into global environment actions.
    """

    def __init__(
        self, training: bool = True, capacitance_model=None
    ):  # "fake"):
        """
        Initialize multi-agent wrapper.

        Automatically infers the array size from the underlying base env

        Args:
            base_env: Base QuantumDeviceEnv instance
        """
        super().__init__()

        self.base_env = QuantumDeviceEnv(
            training=training, capacitance_model=capacitance_model
        )

        self.num_gates = self.base_env.num_dots
        self.use_barriers = self.base_env.use_barriers
        self.num_barriers = self.base_env.num_dots - 1
        self.num_image_channels = self.base_env.num_dots - 1  # N-1 charge stability diagrams

        # Create agent IDs (0-indexed to match expected format)
        self.gate_agent_ids = [f"plunger_{i}" for i in range(self.num_gates)]
        self.barrier_agent_ids = [f"barrier_{i}" for i in range(self.num_barriers)]
        self.all_agent_ids = self.gate_agent_ids + self.barrier_agent_ids

        # Setup channel assignments for agents
        self._setup_channel_assignments()

        # Preserve original spaces for policy mapping
        base_obs = self.base_env.observation_space
        base_action = self.base_env.action_space
        self.base_observation_space = base_obs
        self.base_action_space = base_action

        # Create individual agent spaces
        self._create_agent_spaces(base_obs, base_action)

    def _setup_channel_assignments(self):
        """
        Assign image channels to individual agents.

        For N quantum dots, we have N-1 image channels (charge stability diagrams).
        Gate channel assignment strategy:
        - First gate (plunger_0): Gets [0, 0] (first channel twice)
        - Middle gates: Get adjacent pairs [i-1, i]
        - Last gate (plunger_N-1): Gets [N-2, N-2] (last channel twice)
        - Barrier agents get 1 channel: the channel for dots they separate
        """
        self.agent_channel_map = {}

        # Gate agents: special assignment for ends, pairs for middle
        for i, agent_id in enumerate(self.gate_agent_ids):
            if i == 0:
                # First gate agent: first channel twice
                self.agent_channel_map[agent_id] = [0, 0]
            elif i == self.num_gates - 1:
                # Last gate agent: last channel twice
                last_channel = self.num_gates - 2  # N-1 channels, so index N-2
                self.agent_channel_map[agent_id] = [last_channel, last_channel]
            else:
                # Middle gate agents: adjacent channel pairs [i-1, i]
                # Gate 1 gets [0, 1], Gate 2 gets [1, 2], etc.
                self.agent_channel_map[agent_id] = [i - 1, i]

        # Barrier agents: each gets 1 channel for the dots they separate
        for i, agent_id in enumerate(self.barrier_agent_ids):
            self.agent_channel_map[agent_id] = [i]  # Barrier i separates dots i and i+1

    def _create_agent_spaces(self, base_obs, base_action):
        """Create observation and action spaces for individual agents."""
        image_shape = base_obs["image"].shape  # (H, W, N-1)

        # Voltage ranges
        gate_low = base_action["action_gate_voltages"].low[0]
        gate_high = base_action["action_gate_voltages"].high[0]
        barrier_low = base_action["action_barrier_voltages"].low[0]
        barrier_high = base_action["action_barrier_voltages"].high[0]

        # Create spaces for each agent
        self.observation_spaces = {}
        self.action_spaces = {}

        # Gate agents: 2-channel images + single voltage
        for agent_id in self.gate_agent_ids:
            self.observation_spaces[agent_id] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(image_shape[0], image_shape[1], 2),  # 2 channels
                dtype=np.float32,
            )

            self.action_spaces[agent_id] = spaces.Box(
                low=gate_low,
                high=gate_high,
                shape=(1,),  # Single voltage output
                dtype=np.float32,
            )

        # Barrier agents: 1-channel images + single voltage
        for agent_id in self.barrier_agent_ids:
            self.observation_spaces[agent_id] = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(image_shape[0], image_shape[1], 1),  # 1 channel
                dtype=np.float32,
            )

            self.action_spaces[agent_id] = spaces.Box(
                low=barrier_low,
                high=barrier_high,
                shape=(1,),  # Single voltage output
                dtype=np.float32,
            )

        self.observation_spaces = spaces.Dict(**self.observation_spaces)
        self.action_spaces = spaces.Dict(**self.action_spaces)

        # Set required MultiAgentEnv properties
        self._agent_ids = set(self.all_agent_ids)
        self.observation_space = self.observation_spaces
        self.action_space = self.action_spaces

        self.agents = self._agent_ids.copy()
        self.possible_agents = self._agent_ids.copy()

    def _extract_agent_observation(
        self, global_obs: Dict[str, np.ndarray], agent_id: str
    ) -> Dict[str, np.ndarray]:
        """
        Extract individual agent observation from global observation.

        Args:
            global_obs: Global environment observation
            agent_id: ID of the agent

        Returns:
            Individual agent observation
        """
        channels = self.agent_channel_map[agent_id]

        # Extract appropriate channels for this agent
        global_image = global_obs["image"]  # Shape: (H, W, N-1)

        if len(channels) == 2:
            # Gate agent: 2 channels with conditional y-axis flipping
            agent_idx = int(agent_id.split("_")[1])
            img1 = global_image[:, :, channels[0]]
            img2 = global_image[:, :, channels[1]]
            
            if agent_idx == 0:
                # First agent: no flipping
                agent_image = np.stack([img1, img2], axis=2)
            elif agent_idx == self.num_gates - 1:
                # Final agent: flip both images
                agent_image = np.stack([np.flipud(img1), np.flipud(img2)], axis=2)
            else:
                # Middle agents: flip only second image
                agent_image = np.stack([img1, np.flipud(img2)], axis=2)
        else:
            # Barrier agent: 1 channel
            agent_image = global_image[:, :, channels[0] : channels[0] + 1]

        # Get agent's current voltage value (currently unused in image-only mode)
        if "plunger" in agent_id:
            # agent_idx = int(agent_id.split("_")[1])
            # voltage = global_obs["obs_gate_voltages"][agent_idx : agent_idx + 1]
            pass
        else:  # barrier agent
            # agent_idx = int(agent_id.split("_")[1])
            # voltage = global_obs["obs_barrier_voltages"][agent_idx : agent_idx + 1]
            pass

        # return {
        #     'image': agent_image.astype(np.float32),
        #     'voltage': voltage.astype(np.float32)
        # }
        # IMAGE ONLY SPACE
        return agent_image.astype(np.float32)

    def _combine_agent_actions(self, agent_actions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Combine individual agent actions into global environment action.

        Args:
            agent_actions: Dictionary mapping agent IDs to their actions

        Returns:
            Global environment action
        """
        # Initialize action arrays
        gate_actions = np.zeros(self.num_gates, dtype=np.float32)
        barrier_actions = np.zeros(self.num_barriers, dtype=np.float32)

        # Collect gate actions
        for i, agent_id in enumerate(self.gate_agent_ids):
            if agent_id in agent_actions:
                action_value = agent_actions[agent_id]
                # Handle both scalar and array inputs
                if hasattr(action_value, "__len__"):
                    gate_actions[i] = float(action_value[0])
                else:
                    gate_actions[i] = float(action_value)

        # Collect barrier actions
        for i, agent_id in enumerate(self.barrier_agent_ids):
            if agent_id in agent_actions:
                action_value = agent_actions[agent_id]
                # Handle both scalar and array inputs
                if hasattr(action_value, "__len__"):
                    barrier_actions[i] = float(action_value[0])
                else:
                    barrier_actions[i] = float(action_value)

        return {
            "action_gate_voltages": gate_actions,
            "action_barrier_voltages": barrier_actions,
        }

    def _distribute_rewards(self, global_rewards: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Distribute global rewards to individual agents.

        Args:
            global_rewards: Global reward dictionary with 'gates' and 'barriers' arrays

        Returns:
            Dictionary mapping agent IDs to individual rewards
        """
        agent_rewards = {}

        # Distribute gate rewards
        if "gates" in global_rewards:
            gate_rewards = global_rewards["gates"]
            for i, agent_id in enumerate(self.gate_agent_ids):
                agent_rewards[agent_id] = float(gate_rewards[i])
        else:
            raise ValueError("Missing gate rewards in global_rewards")

        # Distribute barrier rewards
        if "barriers" in global_rewards:
            barrier_rewards = global_rewards["barriers"]
            for i, agent_id in enumerate(self.barrier_agent_ids):
                agent_rewards[agent_id] = float(barrier_rewards[i])
        else:
            raise ValueError("Missing barrier rewards in global_rewards")

        return agent_rewards

    def reset(self, *, seed=None, options=None):
        """
        Reset environment and return individual agent observations.

        Returns:
            Tuple of (observations, infos) where infos is a per-agent dict
        """
        global_obs, global_info = self.base_env.reset(seed=seed, options=options)

        # Convert to multi-agent observations
        agent_observations = {}
        for agent_id in self.all_agent_ids:
            agent_observations[agent_id] = self._extract_agent_observation(global_obs, agent_id)

        # Create per-agent info dict (MultiAgentEnv requirement)
        agent_infos = dict.fromkeys(self.all_agent_ids, global_info)

        return agent_observations, agent_infos

    def step(self, agent_actions: Dict[str, np.ndarray]):
        """
        Step environment with individual agent actions.

        Args:
            agent_actions: Dictionary mapping agent IDs to their actions

        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos)
        """
        assert len(agent_actions) == len(
            self.all_agent_ids
        ), "Agent actions must match the number of agents"
        assert all(
            agent_id in self.all_agent_ids for agent_id in agent_actions.keys()
        ), "Unknown agent IDs in actions"

        print("[ENV DEBUG] env.step called")
        # Combine agent actions into global action
        global_action = self._combine_agent_actions(agent_actions)

        # Step the base environment
        global_obs, global_rewards, terminated, truncated, info = self.base_env.step(global_action)

        # Convert to multi-agent format
        agent_observations = {}
        for agent_id in self.all_agent_ids:
            agent_observations[agent_id] = self._extract_agent_observation(global_obs, agent_id)

        agent_rewards = self._distribute_rewards(global_rewards)

        # Multi-agent termination/truncation (all agents have same status)
        agent_terminated = dict.fromkeys(self.all_agent_ids, terminated)
        agent_terminated["__all__"] = terminated  # Required by MultiAgentEnv

        agent_truncated = dict.fromkeys(self.all_agent_ids, truncated)
        agent_truncated["__all__"] = truncated  # Required by MultiAgentEnv

        # Create per-agent info dict (MultiAgentEnv requirement)
        agent_infos = dict.fromkeys(self.all_agent_ids, info)

        return (
            agent_observations,
            agent_rewards,
            agent_terminated,
            agent_truncated,
            agent_infos,
        )

    def close(self):
        """Close the base environment."""
        return self.base_env.close()

    def get_agent_ids(self):
        """Get list of all agent IDs."""
        return self.all_agent_ids.copy()


if __name__ == "__main__":
    """Test the multi-agent wrapper."""
    print("=== Testing Multi-Agent Quantum Wrapper ===")

    try:
        # Create wrapper (no need for separate base_env)
        wrapper = MultiAgentEnvWrapper(num_dots=4, training=True)  # Small test
        print("✓ Created multi-agent wrapper")

        print(f"Agent IDs: {wrapper.get_agent_ids()}")

        # Test reset
        obs, info = wrapper.reset()
        print(f"✓ Reset successful - got observations for {len(obs)} agents")

        # Check observation shapes
        for agent_id in wrapper.get_agent_ids()[:2]:  # Check first 2 agents
            agent_obs = obs[agent_id]
            print(f"  {agent_id}:")
            print(f"    Observation shape: {agent_obs.shape}")

        # Test step with random actions
        actions = {}
        for agent_id in wrapper.get_agent_ids():
            actions[agent_id] = wrapper.action_spaces[agent_id].sample()

        obs, rewards, terminated, truncated, info = wrapper.step(actions)
        print(f"✓ Step successful - got {len(rewards)} agent rewards")
        print(f"  Sample rewards: {list(rewards.values())[:4]}")

        wrapper.close()
        print("✓ All tests passed!")

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
