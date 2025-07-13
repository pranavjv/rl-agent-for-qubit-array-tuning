import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class PPO:
    def __init__(self, network, env):
        """
        Implements PPO-clip algorithm for an arbitrary environment and agent network

        Parameters:
            network: the actor / critic network
            env: the environment we train in
            hyperparams: override training hyperparameters
        """

        self.env = env
        self.obs_dim = env.observation_space.shape[0] if hasattr(env.observation_space, 'shape') else None
        self.act_dim = env.action_space.shape[0]
        self.act_low = env.action_space.low
        self.act_high = env.action_space.high

        self.actor = network(self.obs_dim, self.act_dim)
        self.critic = network(self.obs_dim, 1)

        #made this a parameter so that it can be optimized
        self.log_std = nn.Parameter(torch.zeros(self.act_dim))

        self.actor_optim = torch.optim.Adam(list(self.actor.parameters()) + [self.log_std], self.lr)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), self.lr)

        self.loss_fn = nn.MSELoss()

        self.timesteps_per_batch = 2000
        self.max_timesteps_per_episode = 500
        self.updates_per_iteration = 5
        self.lr = 5e-3
        self.gamma = 0.95
        self.epsilon = 0.2 # the clip parameter


    def learn(self, T):
        """
        Trains the actor critic networks

        Parameters:
            T: total timesteps
        """

        t = 0 # timestep count
        i = 0 # iteration count

        while t < T:
            obs, acts, logprobs, rtgs, lens = self.rollout() # batched trajectories

            t += np.sum(lens)
            i += 1

            V, _ = self.evaluate(obs, acts) # calls value estimator
            A = rtgs - V.detach() # calculate advantage
            # can normalize A for better stability

            for _ in range(self.updates_per_iteration):
                # update policy by maximising PPO-clip objective
                # update value function on regression on the total reward

                V, current_logprobs = self.evaluate(obs, acts)

                ratios = torch.exp(current_logprobs - logprobs) # updated policy to current policy

                loss1 = ratios * A
                loss2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * A

                actor_loss = - (torch.min(loss1, loss2)).mean()
                critic_loss = self.loss_fn(V, rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
    

    def rollout(self):
        """
        Collect data from running the simulation with updated network
        since PPO is on-policy

        Returns:
            obs
            acts
            logprobs: log probabilities of each action taken; Shape = (num timesteps,)
            rtgs: rewards to go for each timestep of the batch; Shape = (num timesteps,)
            lens: length of the episode for eatch batch; Shape = (num episodes,)
        """
        batch_obs, batch_acts, batch_logprobs, batch_rewards, batch_rtgs, batch_lens = [], [], [], [], [], []

        t = 0

        while t < self.timesteps_per_batch:
            episode_rewards = []
            obs, _ = self.env.reset()
            done = False

            for i in range(self.max_timesteps_per_episode):
                t += 1
                batch_obs.append(obs)
                action, logprob = self.get_action(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated | truncated # we do this from the way gymnasium divides dones

                episode_rewards.append(reward)
                batch_acts.append(action)
                batch_logprobs.append(logprob)

                if done:
                    break

            batch_lens.append(i+1)
            batch_rewards.append(episode_rewards)

        batch_obs = torch.tensor(batch_obs, dtype=torch.float32)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float32)
        batch_logprobs = torch.tensor(batch_logprobs, dtype=torch.float32)
        batch_rtgs = self.compute_rtgs(batch_rewards)

        return batch_obs, batch_acts, batch_logprobs, batch_rtgs, batch_lens
                

    def compute_rtgs(self, rewards):
        """
        Computes RTGs (rewards to go) i.e. total discounted reward from a timestep t to the end of the episode

        Parameters:
            rewards: all rewards in the given batch; Shape = (num episodes, num timesteps per episode)

        Returns:
            rtgs: rewards to go; Shape = (num timesteps in batch,)
        """
        batch_rtgs = []

        for ep_rews in reversed(rewards):
            discounted_rew = 0

            for rew in reversed(ep_rews):
                discounted_rew = rew + discounted_rew * self.gamma
                batch_rtgs.append(discounted_rew)

        # note this assumes the batch loops over the episodes

        batch_rtgs = torch.tensor(reversed(batch_rtgs), dtype=torch.float)
        return batch_rtgs


    def get_action(self, state):
        """
        Queries the actor network, should be called from rollout function

        Returns:
            action: action to take
            logprob: log prob of the action selected from distribution
        """
        # convert state to tensor just in case it's not already
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        
        # Get action mean
        mean = self.actor(state)
        
        # Create normal distribution with fixed std
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        
        # Sample action
        action = dist.sample()
        
        # Clip action to valid range
        action = torch.clamp(action, self.act_low, self.act_high)
        
        # Get log probability
        logprob = dist.log_prob(action).sum()
        
        return action.detach().numpy(), logprob.detach()


    def evaluate(self, states, actions):
        """
        Estimates the value of the state and the log probabilities of each action

        Parameters:
            states: observations (states) from the most recently collected batch (Tensor); Shape: (timesteps, obs dim)
            actions: actions taken from the most recent batch (Tensor); Shape: (timesteps, action dim)

        Returns:
            V: predicted value of the observations
            logprobs: log probs of the actions given the states
        """
        V = self.critic(states).squeeze()

        # Get action mean from actor network
        mean = self.actor(states)
        
        # Create normal distribution with fixed std
        std = torch.exp(self.log_std)
        dist = Normal(mean, std)
        
        # Get log probabilities
        logprobs = dist.log_prob(actions).sum(dim=-1)

        return V, logprobs

