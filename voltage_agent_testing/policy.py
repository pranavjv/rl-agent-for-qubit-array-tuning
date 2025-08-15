from typing import Union, Type, Dict, Any, Optional, Tuple
import numpy as np
import torch as th
import torch
from torch import nn
from gymnasium import spaces
from torch.nn import functional as F
import gymnasium as gym
from functools import partial
from tqdm import tqdm

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, Schedule, PyTorchObs
from stable_baselines3.common.utils import FloatSchedule, explained_variance, obs_as_tensor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from stable_baselines3.common.distributions import make_proba_distribution, Distribution
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Space, agent, device, feature_dim: int = None):
        if feature_dim is None:
            feature_dim = getattr(agent, 'feature_dim', 128)

        super().__init__(observation_space, feature_dim)
        self.agent = agent
        self.feature_dim = feature_dim
        self.device = device

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if isinstance(obs, dict):
            img = obs.get('image')
            voltages = obs.get('voltages')

            if not isinstance(img, torch.Tensor):
                img = torch.tensor(img)
            if not isinstance(voltages, torch.Tensor):
                voltages = torch.tensor(voltages)

            if img.ndim == 3:
                img = img.unsqueeze(0)
            if img.shape[-1] in [1, 3]:
                img = img.permute(0, 3, 1, 2)
                
            if voltages.ndim == 1:
                voltages = voltages.unsqueeze(0)
            
            img = img.to(torch.float32).to(self.device)
            voltages = voltages.to(torch.float32).to(self.device)
            return self.agent.encode_obs(img, voltages)
        else:
            raise ValueError(f"Expected obs to be a dict, got {type(obs)}")


class CustomAgentPolicy(BasePolicy):
    """
    Custom policy based on ActorCriticPolicy that handles dict observations
    """
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        agent_class: Type = None,
        agent_kwargs: Dict[str, Any] = None,
        net_arch: Optional[Union[list[int], dict[str, list[int]]]] = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        device: torch.device = None,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
        **kwargs
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        feature_extractor_class = CustomFeatureExtractor
        self.agent_kwargs = agent_kwargs or {}
        feature_extractor_kwargs = {"agent": agent_class(device=device, **self.agent_kwargs), "device": device}

        super().__init__(
            observation_space,
            action_space,
            feature_extractor_class,
            feature_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            **kwargs
        )
        self.agent_class = agent_class
        self.agent = feature_extractor_kwargs["agent"]

        self.optimizer = self.optimizer_class(self.agent.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        self.feature_extractor = feature_extractor_class(
            observation_space=observation_space,
            agent=self.agent,
            device=device,
        )
        self.feature_dim = self.feature_extractor.feature_dim

        # Action distribution
        self.use_sde = use_sde
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(
            dict(
                agent_class=self.agent_class,
                agent_kwargs=self.agent_kwargs,
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                ortho_init=self.ortho_init,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                lr_schedule=self._dummy_schedule,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
            )
        )
        return data


    def forward(self, obs: PyTorchObs, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.feature_extractor(obs)
        actions, dist, logprob, quality_logits, value = self.agent.forward_step(features)
        
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, value, logprob

    def _get_action_dist_from_latent(self, features: th.Tensor) -> Distribution:
        _, dist, *_ = self.agent.forward_step(features)
        return dist

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy
        """
        features = self.feature_extractor(obs)
        _, dist, logprob, _, values = self.agent.forward_step(features)
        entropy = dist.entropy()

        return values, logprob, entropy

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get estimated values for given observations
        """
        features = self.feature_extractor(obs)
        return self.agent.critic(features)

    def get_distribution(self, obs: PyTorchObs) -> Distribution:
        """
        Get the current policy distribution
        """
        features = self.feature_extractor(obs)
        _, dist, _, _ = self.agent.forward_step(features)
        return dist

    def _predict(self, obs: PyTorchObs, deterministic: bool = False) -> th.Tensor:
        """
        Get action according to policy for given observation
        """
        feats = self.feature_extractor(obs)
        return self.agent.predict_action(feats)

    def set_training_mode(self, mode: bool) -> None:
        """
        Set training mode
        """
        self.train(mode)
        if hasattr(self.agent, 'train'):
            self.agent.train(mode)


class PPO(OnPolicyAlgorithm):
    
    policy_aliases = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
        "CustomAgentPolicy": CustomAgentPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        policy_kwargs=None,
        device: Union[th.device, str] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        # Custom agent parameters
        agent_class=None,
        agent_kwargs: Dict[str, Any] = None,
    ):
        # Handle custom agent policy
        if policy == "CustomAgentPolicy" or agent_class is not None:
            if policy_kwargs is None:
                policy_kwargs = {}
            policy_kwargs.update({
                "agent_class": agent_class,
                "agent_kwargs": agent_kwargs or {}
            })
            policy = "CustomAgentPolicy"

        assert batch_size > 1, "`batch_size` must be greater than 1"

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        self._setup_model()

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a RolloutBuffer.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        # Add progress bar for rollout collection
        with tqdm(total=n_rollout_steps, desc="Collecting Rollouts") as pbar:
            while n_steps < n_rollout_steps:
                if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.policy.reset_noise(env.num_envs)

                with th.no_grad():
                    # Convert to pytorch tensor
                    obs_tensor = obs_as_tensor(self._last_obs, self.device)
                    actions, values, log_probs = self.policy(obs_tensor)
                
                actions = actions.cpu().numpy()

                # Rescale and perform action
                clipped_actions = actions
                if isinstance(self.action_space, spaces.Box):
                    # Clip the actions to avoid out of bound error
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

                new_obs, rewards, dones, infos = env.step(clipped_actions)

                self.num_timesteps += env.num_envs

                # Give access to local variables
                callback.update_locals(locals())
                if not callback.on_step():
                    return False

                self._update_info_buffer(infos, dones)
                n_steps += 1
                pbar.update(1)  # Update progress bar

                if isinstance(self.action_space, spaces.Discrete):
                    # Reshape in case of discrete action
                    actions = actions.reshape(-1, 1)

                # Handle timeout by bootstrapping with value function
                for idx, done in enumerate(dones):
                    if (
                        done
                        and infos[idx].get("terminal_observation") is not None
                        and infos[idx].get("TimeLimit.truncated", False)
                    ):
                        terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                        with th.no_grad():
                            terminal_value = self.policy.predict_values(terminal_obs)[0]
                        rewards[idx] += self.gamma * terminal_value

                rollout_buffer.add(
                    self._last_obs,
                    actions,
                    rewards,
                    self._last_episode_starts,
                    values,
                    log_probs,
                )
                self._last_obs = new_obs
                self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True

    def _setup_model(self) -> None:
        super()._setup_model()

        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            self.clip_range_vf = FloatSchedule(self.clip_range_vf)


    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        for epoch in range(self.n_epochs):
            print('Epoch: ', epoch)
            for rollout_data in tqdm(self.rollout_buffer.get(self.batch_size)):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)

