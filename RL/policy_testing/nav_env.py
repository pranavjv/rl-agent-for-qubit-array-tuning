import matplotlib
matplotlib.use('Agg')
import gymnasium as gym
from gymnasium import spaces

import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image

class NavEnv(gym.Env):

    metadata = {
        'render_modes': ['human', 'array'],
    }

    def __init__(self, **kwargs):
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        self.render_mode = kwargs.get('render_mode', 'array')
        assert self.render_mode in self.metadata['render_modes'], f"Invalid render mode: {self.render_mode}"

        self.ndim = 2
        self.max_size = 250
        self.max_step_size = 10
        self.max_distance = math.sqrt(self.max_size**2 + self.max_size**2)
        self.max_steps = 200 # encourage short episodes, 15 steps should be enough
        self.current_step = 0

        self.image_size = 200

        self.hit_rate = 0
        self.total_episodes = 0

        self.target_tolerance = 10
        self.target_radius = self.target_tolerance / self.max_size

        self.action_space = spaces.Box(
            low = -self.max_step_size,
            high = self.max_step_size,
            shape=(self.ndim,),
            dtype=np.float32
        )

        if self.render_mode == 'array':
            self.observation_space = spaces.Box(
                low=0,
                high=1,
                shape=(2*self.ndim + 1,),
                dtype=np.float32
            )
        else:
            # 400x400 normalised image (figsize=4x4 at 100 DPI)
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(3, self.image_size, self.image_size),
                dtype=np.float32
            )

        self.reset()

    def get_accuracy(self):
        return 100 * self.hit_rate / self.total_episodes if self.total_episodes > 0 else 0

    def _distance(self):
        return math.sqrt(pow((self.target_x - self.agent_x), 2) + pow(self.target_y - self.agent_y, 2))

    def _get_observation(self):
        if self.render_mode == 'array':
            obs = [self.agent_x, self.agent_y, self.target_x, self.target_y, self._distance()]
            return obs
        
        img_size = self.image_size
        img = np.zeros((3, img_size, img_size), dtype=np.uint8)
        
        # normalise coords to image size
        agent_x_norm = int((self.agent_x / self.max_size) * (img_size - 1))
        agent_y_norm = int((self.agent_y / self.max_size) * (img_size - 1))
        target_x_norm = int((self.target_x / self.max_size) * (img_size - 1))
        target_y_norm = int((self.target_y / self.max_size) * (img_size - 1))

        # draw agent
        y_coords, x_coords = np.ogrid[:img_size, :img_size]
        agent_mask = (x_coords - agent_x_norm)**2 + (y_coords - agent_y_norm)**2 <= 9
        img[2, agent_mask] = 255  # Blue channel
        
        # draw target
        target_mask = (x_coords - target_x_norm)**2 + (y_coords - target_y_norm)**2 <= 9
        img[0, target_mask] = 255  # Red channel
        return img

        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.scatter(self.agent_x, self.agent_y, c='blue', s=50)
        ax.scatter(self.target_x, self.target_y, c='red', s=50)
        ax.axis('off')
        ax.set_xlim(0, self.max_size)
        ax.set_ylim(0, self.max_size)
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8')
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, 1:]
        plt.close(fig)
        img = np.transpose(img, (2, 0, 1))  # (channels, height, width)
        return img

    def _normalise_observation(self, obs):
        if self.render_mode == 'array':
            obs = [o/self.max_size for o in obs[:-1]] + [obs[-1]/self.max_distance]
            return np.array(obs, dtype=np.float32)

        return obs / 255.0

    def random_action(self):
        return self.action_space.sample()

    def render(self):
        if self.render_mode == 'array':
            obs = self._get_observation()
            obs = self._normalise_observation(obs)
            return obs[:-1]
        
        img = self._get_observation()
        img = self._normalise_observation(img)
        return img

    def reset(self, seed=None):
        if seed is not None:
            self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.current_step = 0

        self.agent_x = np.random.uniform(0, self.max_size)
        self.agent_y = np.random.uniform(0, self.max_size)
        self.target_x = np.random.uniform(0, self.max_size)
        self.target_y = np.random.uniform(0, self.max_size)

        if self._distance() < self.target_tolerance * 1.1:
            self.target_x = np.random.uniform(0, self.max_size)
            self.target_y = np.random.uniform(0, self.max_size)

        obs = self._get_observation()
        return self._normalise_observation(obs), None

    def step(self, action):
        self.current_step += 1

        terminated = False
        truncated = False

        step_x, step_y = action
        self.agent_x += step_x
        self.agent_y += step_y
        self.agent_x = np.clip(self.agent_x, 0, self.max_size)
        self.agent_y = np.clip(self.agent_y, 0, self.max_size)

        if self._distance() < self.target_tolerance:
            reward = 100.0
            terminated = True
            self.total_episodes += 1
            self.hit_rate += 1
            # print(f"Reached target in {self.current_step} steps")
        elif self.current_step >= self.max_steps:
            reward = -1.0
            truncated = True
            self.total_episodes += 1
            # print(f"Episode truncated after {self.current_step} steps")
        else:
            reward = (1.0 - (self._distance() / self.max_distance)) * 0.1
            # reward = -0.01

        obs = self._get_observation()
        obs = self._normalise_observation(obs)

        return obs, reward, terminated, truncated, {"info": None}


if __name__ == "__main__":

    env = NavEnv(render_mode='human')
    obs, _ = env.reset()
    done = False

    i = 0

    while not done:
        i += 1
        action = env.random_action()
        obs, reward, terminated, truncated, _ = env.step(action)
        img = env.render()
        print(img.shape)

        img = (img * 255).astype(np.uint8)
        # Convert from (C, H, W) to (H, W, C) for PIL
        img = np.transpose(img, (1, 2, 0))
        image = Image.fromarray(img)
        image.save(f"test_ims/agent_target_image_{i}.png")

        if terminated or truncated:
            done = True

    env.close()