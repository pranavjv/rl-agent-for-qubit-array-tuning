"""
Handles the previous action processing prior to LSTM embedding
to work around using deltas
"""
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import EpisodeType
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

from typing import Optional, Dict, Any, List
import gymnasium as gym
import numpy as np


class CustomPrevActionHandling(ConnectorV2):
    # def __init__(
    #     self,
    #     input_observation_space: Optional[gym.Space] = None,
    #     input_action_space: Optional[gym.Space] = None,
    #     **kwargs,
    # ):
    #     super().__init__(input_observation_space, input_action_space, **kwargs)
        

    @override(ConnectorV2)
    def __call__(
        self,
        *,
        rl_module: RLModule,
        batch: Dict[str, Any],
        episodes: List[EpisodeType],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        metrics: Optional[MetricsLogger] = None,
        **kwargs,
    ) -> Any:

        try:
            for sa_episode in self.single_agent_episode_iterator(
                episodes, agents_that_stepped_only=True
            ):
                # Episode is not numpy'ized yet and thus still operates on lists of items.
                assert not sa_episode.is_numpy

                # Get current observation
                current_obs = sa_episode.get_observations(-1)
                
                # Extract previous gate voltages from environment observations
                # Look back one step to get previous voltages
                if len(sa_episode) >= 2:
                    prev_obs = sa_episode.get_observations(-2)
                    
                    # Extract "obs_gate_voltages" from previous observation
                    if isinstance(prev_obs, dict) and "obs_gate_voltages" in prev_obs:
                        prev_gate_voltages = prev_obs["obs_gate_voltages"]
                    else:
                        # If obs_gate_voltages not found, fill with zeros
                        prev_gate_voltages = np.zeros((1,))  # Adjust shape as needed
                else:
                    # For first step, use zeros
                    prev_gate_voltages = np.zeros((1,))  # Adjust shape as needed

                # Get the most recent action (delta)
                if len(sa_episode) >= 1:
                    last_action = sa_episode.get_actions(-1)
                    
                    # Convert delta action to absolute action by adding previous voltages
                    if isinstance(last_action, (list, tuple, np.ndarray)):
                        absolute_action = np.array(last_action) + prev_gate_voltages
                        
                        # Update the action in the episode
                        sa_episode.set_actions(at_indices=-1, new_data=absolute_action)

            return batch
            
        except Exception as e:
            raise Exception(f"Error in CustomPrevActionHandling: {e}")
