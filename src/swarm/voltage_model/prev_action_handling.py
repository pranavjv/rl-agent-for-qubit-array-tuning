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

    #     assert delta_max > delta_min
    #     self.delta_min = delta_min
    #     self.delta_max = delta_max

    #     # these are the same min and max that are used in env, prior to adding an offset
    #     assert obs_voltage_max > obs_voltage_min
    #     self.obs_voltage_min = obs_voltage_min
    #     self.obs_voltage_max = obs_voltage_max

    # def _rescale_action(self, action, _min=-1.0, _max=1.0):
    #     assert _max > _min
    #     action = (action - _min) / (_max - _min) # rescale to [0, 1]
    #     action = action * (self.delta_max - self.delta_min) + self.delta_min
    #     return action


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

                agent_id = sa_episode.agent_id
                is_plunger = agent_id is not None and "plunger" in agent_id
                
                if is_plunger:
                    current_obs = sa_episode.get_observations(-1)

                    if isinstance(current_obs, dict) and "obs" in current_obs:
                        current_obs = current_obs["obs"]

                    obs_gate_voltages = current_obs["voltage"]
                    # already normalised in [-1, 1]
                    
                    # NOTE: we do not need to add the actions at all, the last observation is our last 'action'
                    # current_offset = current_obs["offset"]
                    # last_action = sa_episode.get_actions(-1)
                    # last_action = self._rescale_action(last_action)
                    # absolute_action = np.array(last_action) + obs_gate_voltages
                    # absolute_action = np.clip(absolute_action, self.obs_voltage_min + current_offset, self.obs_voltage_max + current_offset) # ensure we are within bounds
                            
                    self.add_batch_item(
                        batch,
                        SampleBatch.PREV_ACTIONS,
                        item_to_add=obs_gate_voltages,
                        single_agent_episode=sa_episode,
                    )
                else:
                    last_action = sa_episode.get_actions(-1, fill=0.0)
                    # could update the fill value to a more realistic one (eg. use the initialisation)
                    
                    self.add_batch_item(
                        batch,
                        SampleBatch.PREV_ACTIONS,
                        item_to_add=last_action,
                        single_agent_episode=sa_episode,
                    )

            return batch
            
        except Exception as e:
            raise Exception(f"Error in CustomPrevActionHandling: {e}")
