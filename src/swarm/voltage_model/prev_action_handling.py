"""
Handles the previous action processing prior to LSTM embedding
to work around using deltas
"""
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import EpisodeType
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.policy.sample_batch import SampleBatch

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

                # Get current observation
                current_obs = sa_episode.get_observations(-1)
                obs_gate_voltages = current_obs["obs_gate_voltages"]
                # raises an error, current_obs is now an array -> either use whole dict, or keep history as an attribute somehow
                
                last_action = sa_episode.get_actions(-1)
                    
                absolute_action = np.array(last_action) + obs_gate_voltages
                        
                self.add_batch_item(
                    batch,
                    SampleBatch.PREV_ACTIONS,
                    item_to_add=absolute_action,
                    single_agent_episode=sa_episode,
                )

            return batch
            
        except Exception as e:
            raise Exception(f"Error in CustomPrevActionHandling: {e}")
