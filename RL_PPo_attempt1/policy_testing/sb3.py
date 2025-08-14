import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio
import numpy as np
import os
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor, BaseFeaturesExtractor
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from utils import CustomLoggingCallback
from nav_env import NavEnv

logging.getLogger("moviepy").setLevel(logging.ERROR)


class CNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for image inputs.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(CNNFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # image should be (channels, height, width)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # manually get cnn latent dim
        with torch.no_grad():
            sample_input = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_input).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class PolicyNetwork(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):
        net_arch = dict(
            pi=[512, 512, 256], # policy network
            vf=[512, 512, 256] # value network
        )

        features_extractor_class = CNNFeatureExtractor
        
        super(PolicyNetwork, self).__init__(
            observation_space, 
            action_space, 
            lr_schedule,
            net_arch=net_arch,
            activation_fn=nn.ReLU,
            log_std_init=-0.5,
            features_extractor_class=features_extractor_class,
            # features_extractor_kwargs=features_extractor_kwargs,
            *args, 
            **kwargs
        )


def train_ppo(mode, save_path=None, load_path=None, train_timesteps=100_000):
    assert mode in ['train', 'train_resume', 'infer'], "Mode must be one of 'train', 'train_resume', 'infer'."

    config = {
        'save_freq': train_timesteps // 5,  # 5 videos over total_timesteps
        'total_timesteps': train_timesteps,
        'video_length': 200,
        'video_folder': './nav_videos/'
    }

    train_env = DummyVecEnv([lambda: NavEnv(render_mode='human') for _ in range(4)])

    if mode == 'train':
        model = PPO(
            policy=PolicyNetwork,
            #policy='MlpPolicy',
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

        custom_callback = CustomLoggingCallback(verbose=1)

        model.learn(
            total_timesteps=config['total_timesteps'],
            progress_bar=True,
            callback=custom_callback
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
    #model = train_ppo(mode='infer', load_path='ppo_navigation_train')

    # eval
    save_runs = 4
    print(f'Saving {save_runs} eval runs ...')
    for i in range(save_runs):
        eval_env = NavEnv(render_mode='human')
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
