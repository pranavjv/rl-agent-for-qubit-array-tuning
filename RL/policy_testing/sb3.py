import matplotlib
matplotlib.use('Agg')  # Use Agg backend for rendering without a display
import matplotlib.pyplot as plt
import imageio
import numpy as np
import os
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor
import torch
import torch.nn as nn
import gymnasium as gym

from nav_env import NavEnv

# Suppress MoviePy verbose output
logging.getLogger("moviepy").setLevel(logging.ERROR)

class PolicyNetwork(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        self.hidden_dim = 256  # Increased hidden dimension for more parameters
        self.observation_space = observation_space
        self.action_space = action_space

        if isinstance(action_space, gym.spaces.Discrete):
            self.action_dim = action_space.n
        elif isinstance(action_space, gym.spaces.Box):
            self.action_dim = action_space.shape[0]
        else:
            raise ValueError("Unsupported action space type")

        super(PolicyNetwork, self).__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        # Replace Sequential with MlpExtractor
        self.mlp_extractor = MlpExtractor(
            feature_dim=observation_space.shape[0],
            net_arch=[self.hidden_dim, self.hidden_dim, self.hidden_dim],
            activation_fn=nn.ReLU
        )

        self.log_std = nn.Parameter(torch.zeros(self.action_dim))  # Fixed log_std for simplicity
        self.optimizer = torch.optim.Adam(list(self.parameters()) + [self.log_std], lr=lr_schedule(1))

    def forward(self, obs, deterministic=False):
        # Ensure the input tensor is correctly reshaped
        if len(obs.shape) == 1:
            obs = obs.unsqueeze(0)  # Add batch dimension if missing

        features = self.mlp_extractor(obs)[0]  # extractor returns a tuple
        actions = self.action_net(features)
        values = self.value_net(features)

        if isinstance(self.action_space, gym.spaces.Discrete):
            log_probs = torch.log_softmax(actions, dim=-1) if not deterministic else None
        elif isinstance(self.action_space, gym.spaces.Box):
            mean = actions
            log_std = torch.zeros_like(mean)  # Fixed log_std for simplicity
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(mean).sum(dim=-1) if not deterministic else None
        else:
            raise ValueError("Unsupported action space type")

        return actions, values, log_probs

    def _build(self, lr_schedule):
        # Build the actor and critic networks with more parameters
        self.action_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim)
        )

        self.value_net = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )


def train_ppo(mode, save_path=None, load_path=None, train_timesteps=100_000):
    assert mode in ['train', 'train_resume', 'infer'], "Mode must be one of 'train', 'train_resume', 'infer'."

    config = {
        'save_freq': train_timesteps // 5,  # 5 videos over total_timesteps
        'total_timesteps': train_timesteps,
        'video_length': 200,
        'video_folder': './nav_videos/'
    }

    # Create vectorized training env
    train_env = DummyVecEnv([lambda: NavEnv() for _ in range(4)])

    if mode == 'train':
        model = PPO(
            #policy=PolicyNetwork, # "MlpPolicy"
            policy='MlpPolicy',
            env=train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1
        )
    elif mode == 'train_resume' or mode == 'infer':
        model = PPO.load(load_path, env=train_env)

    if mode == 'train' or mode == 'train_resume':
        assert save_path is not None

        model.learn(
            total_timesteps=config['total_timesteps'],
            progress_bar=True
        )

        model.save(save_path)

    train_env.close()
    return model


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    os.makedirs('./nav_videos/', exist_ok=True)

    print('Training ...')
    #model = train_ppo(mode='train_resume', load_path='ppo_navigation.zip', save_path='ppo_navigation_v2')
    model = train_ppo(mode='train', save_path='ppo_navigation_train', train_timesteps=1_000_000)

    # eval
    save_runs = 2
    print(f'Saving {save_runs} eval runs ...')
    for i in range(save_runs):
        eval_env = NavEnv()
        obs, _ = eval_env.reset()
        frames = []

        for t in range(1000):
            action = model.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            state = eval_env.render()

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.scatter(state[0], state[1], c='blue', label='agent')
            ax.scatter(state[2], state[3], c='red', label='goal')
            goal_circle = plt.Circle((state[2], state[3]), eval_env.target_radius, color='red', fill=False, linestyle='--', label='goal radius')
            ax.add_artist(goal_circle)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend()
            ax.set_title(f"Step {t} dist={obs[4]:.2f}")
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, 1:]
            plt.close(fig)
            frames.append(img)

            if terminated or truncated:
                print(f"Episode finished after {t+1} steps.")
                break

        eval_env.close()
        out_file = f'./nav_videos/final_rollout_{i+1}.mp4'
        imageio.mimwrite(out_file, frames, fps=30)
        print(f"Saved final video: {out_file} ({len(frames)} frames)")

    print('Done.')
