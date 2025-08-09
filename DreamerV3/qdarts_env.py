import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml
import os
import time

# qdarts imports
from qdarts_config_loader_v5 import load_qdarts_config
from qdarts.experiment_with_barriers import Experiment

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io


class QdartsEnv(gym.Env):
    """
    Gymnasium environment for qdarts with a 5D action space:
    3 barrier voltages + 2 plunger voltages (x, y).

    Observation is a multi-modal dict with:
      - image: normalized sensor response (H, W, 1) uint8
      - voltages: current [b0, b1, b2, px, py] float32
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        env_config_path: str = 'qdarts_env_config.yaml',
        qdarts_config_path: str = 'qdarts_config_v5.yaml',
        render_mode: str | None = None,
        **overrides,
    ):
        super().__init__()

        # --- Load Configs ---
        self._script_dir = os.path.dirname(os.path.abspath(__file__))
        self.env_config = self._load_yaml(os.path.join(self._script_dir, env_config_path))
        self.qdarts_config = load_qdarts_config(qdarts_config_path)

        # Apply simple overrides (flat keys only)
        for k, v in overrides.items():
            self.env_config = self._set_in_dict(self.env_config, k, v)

        training = self.env_config['training']
        self.debug = bool(training.get('debug', False))
        self.seed_value = training.get('seed', None)
        self.max_steps = int(training['max_episode_steps'])
        self.render_fps = int(training.get('render_fps', 30))
        self.render_mode = render_mode or training.get('render_mode', 'rgb_array')

        env = self.env_config['env']
        self.tolerance = float(env['tolerance'])
        self.clip_actions = bool(env['action_space'].get('clip_actions', True))

        # Barrier names and ranges
        self.barrier_names = list(env['action_space']['barrier_names'])
        self.barrier_range = env['action_space']['barrier_range']  # [min, max] or per-barrier
        if isinstance(self.barrier_range[0], (list, tuple)):
            barrier_lows = [float(x[0]) for x in self.barrier_range]
            barrier_highs = [float(x[1]) for x in self.barrier_range]
        else:
            barrier_lows = [float(self.barrier_range[0])] * 3
            barrier_highs = [float(self.barrier_range[1])] * 3

        # Plunger range
        self.plunger_range = env['action_space']['plunger_range']  # [min, max] or per-axis
        if isinstance(self.plunger_range[0], (list, tuple)):
            plunger_lows = [float(self.plunger_range[0][0]), float(self.plunger_range[1][0])]
            plunger_highs = [float(self.plunger_range[0][1]), float(self.plunger_range[1][1])]
        else:
            plunger_lows = [float(self.plunger_range[0])] * 2
            plunger_highs = [float(self.plunger_range[1])] * 2

        # Assemble action space bounds
        self._action_low = np.array(barrier_lows + plunger_lows, dtype=np.float32)
        self._action_high = np.array(barrier_highs + plunger_highs, dtype=np.float32)

        self.action_space = spaces.Box(low=self._action_low, high=self._action_high, shape=(5,), dtype=np.float32)

        # Observation space
        obs_cfg = self.env_config['env']['observation_space']
        self.obs_image_size = tuple(obs_cfg['image_size'])  # (H, W)
        self.obs_channels = int(obs_cfg.get('channels', 1))
        self.obs_dtype = obs_cfg.get('dtype', 'uint8')
        self.include_voltages = bool(obs_cfg.get('include_voltages', True))

        obs_spaces = {
            'image': spaces.Box(low=0, high=255, shape=(self.obs_image_size[0], self.obs_image_size[1], self.obs_channels), dtype=np.uint8)
        }
        if self.include_voltages:
            obs_spaces['voltages'] = spaces.Box(low=self._action_low, high=self._action_high, shape=(5,), dtype=np.float32)
        self.observation_space = spaces.Dict(obs_spaces)

        # Measurement configuration for local window
        meas_cfg = self.env_config['simulator']['measurement']
        self.window_size = tuple(meas_cfg['window_size'])  # (dx, dy)
        self.resolution = int(meas_cfg['resolution'])
        self.use_sensor_signal = bool(meas_cfg.get('use_sensor_signal', True))
        self.compensate_sensors = bool(meas_cfg.get('compensate_sensors', True))
        self.compute_polytopes = bool(meas_cfg.get('compute_polytopes', False))

        # Reward config
        reward_cfg = self.env_config.get('reward', {})
        self.reward_type = reward_cfg.get('type', 'distance_to_target')
        weights = reward_cfg.get('weights', {})
        self.distance_weight = float(weights.get('distance_weight', 0.1))
        self.step_penalty = float(weights.get('step_penalty', 0.1))
        self.success_bonus = float(weights.get('success_bonus', 200.0))
        self.success_threshold = float(reward_cfg.get('success_threshold', self.tolerance))
        self.target_center = np.array(self.env_config['env'].get('target_center', [0.0, 0.0]), dtype=np.float64)

        # Internal state holders
        self.current_step = 0
        self._experiment: Experiment | None = None
        self._plane_axes: list[int] | None = None
        self._last_sensor = None  # np.ndarray (H, W, C)
        self._last_image = None   # normalized uint8 image
        self._last_voltages = None  # np.ndarray shape (5,)
        self._last_window_extents = None  # (x_min, x_max, y_min, y_max)

        # Normalization parameters
        self._init_normalization_params()

        # Validate barrier names against qdarts config
        default_barriers = self.env_config['simulator']['barrier'].get('default_barrier_voltages', None)
        if default_barriers is None:
            # try from qdarts_config
            default_barriers = self.qdarts_config['simulator']['barrier']['default_barrier_voltages']
        missing = [b for b in self.barrier_names if b not in default_barriers]
        if missing:
            raise ValueError(f"Barrier names {missing} not present in qdarts barrier config")

    def _init_normalization_params(self) -> None:
        self.data_min = 0.13
        self.data_max = 0.16
        self.bounds_initialized = False
        self.episode_min = float('inf')
        self.episode_max = float('-inf')
        self.global_min = float('inf')
        self.global_max = float('-inf')
        self.update_count = 0

    def _update_normalization_bounds(self, raw_data: np.ndarray) -> None:
        self.episode_min = min(self.episode_min, float(np.min(raw_data)))
        self.episode_max = max(self.episode_max, float(np.max(raw_data)))
        self.global_min = min(self.global_min, float(np.min(raw_data)))
        self.global_max = max(self.global_max, float(np.max(raw_data)))

        needs_update = False
        new_min = self.data_min
        new_max = self.data_max

        if self.global_min < self.data_min:
            safety_margin = (self.data_max - self.data_min) * 0.05
            new_min = float(self.global_min - safety_margin)
            needs_update = True
        if self.global_max > self.data_max:
            safety_margin = (self.data_max - self.data_min) * 0.05
            new_max = float(self.global_max + safety_margin)
            needs_update = True

        if needs_update:
            self.data_min = new_min
            self.data_max = new_max
            self.update_count += 1
            if self.debug:
                print(f"Updated normalization bounds to [{self.data_min:.4f}, {self.data_max:.4f}] (update #{self.update_count})")

    def _normalize_image(self, raw_2d: np.ndarray) -> np.ndarray:
        if raw_2d.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {raw_2d.shape}")
        self._update_normalization_bounds(raw_2d)
        normalized = (raw_2d - self.data_min) / max(self.data_max - self.data_min, 1e-12)
        normalized = np.clip(normalized, 0.0, 1.0)
        normalized = (normalized * 255).astype(np.uint8)
        normalized = normalized.reshape(normalized.shape[0], normalized.shape[1], 1)
        return normalized

    def _build_experiment(self) -> None:
        cap_cfg = self.qdarts_config['simulator']['capacitance']
        tun_cfg = self.qdarts_config['simulator']['tunneling']
        sen_cfg = self.qdarts_config['simulator']['sensor']
        bar_cfg = self.qdarts_config['simulator']['barrier']
        self._experiment = Experiment(
            capacitance_config=cap_cfg,
            tunneling_config=tun_cfg,
            sensor_config=sen_cfg,
            barrier_config=bar_cfg,
            print_logs=self.debug,
        )
        sweep_matrix = self.qdarts_config['measurement']['sweep_matrix']
        self._plane_axes = self._plane_axes_from_sweep_matrix(sweep_matrix)

    @staticmethod
    def _plane_axes_from_sweep_matrix(sweep_matrix) -> list[int]:
        plane_axes: list[int] = []
        for i in range(2):
            for j, row in enumerate(sweep_matrix):
                if abs(row[i]) > 1e-10:
                    plane_axes.append(j)
                    break
        if len(plane_axes) != 2:
            raise ValueError(f"Could not infer 2 plane axes from sweep_matrix: {sweep_matrix}")
        return plane_axes

    def _set_barriers(self, barrier_values: dict) -> None:
        assert self._experiment is not None
        self._experiment.update_tunnel_couplings(barrier_values)

    def _generate_window(self, center_xy: np.ndarray) -> tuple[np.ndarray, dict]:
        assert self._experiment is not None and self._plane_axes is not None
        dx, dy = self.window_size
        x_voltages = np.linspace(center_xy[0] - dx, center_xy[0] + dx, self.resolution)
        y_voltages = np.linspace(center_xy[1] - dy, center_xy[1] + dy, self.resolution)

        t0 = time.time()
        xout, yout, _, polytopes, sensor_values, v_offset = self._experiment.generate_CSD(
            x_voltages=x_voltages,
            y_voltages=y_voltages,
            plane_axes=self._plane_axes,
            target_state=self.qdarts_config['device']['target_state'],
            use_sensor_signal=self.use_sensor_signal,
            compensate_sensors=self.compensate_sensors,
            compute_polytopes=self.compute_polytopes,
        )
        dt = time.time() - t0

        meta = {
            'xout': xout,
            'yout': yout,
            'polytopes': polytopes,
            'v_offset': v_offset,
            'elapsed_s': dt,
        }
        return sensor_values, meta

    def _get_obs(self) -> dict:
        z = self._last_sensor
        channel = z[:, :, 0]
        img = self._normalize_image(channel)
        self._last_image = img
        obs = {'image': img}
        if self.include_voltages:
            obs['voltages'] = self._last_voltages.astype(np.float32)
        return obs

    def _get_info(self, meta: dict | None = None) -> dict:
        info = {
            'current_step': self.current_step,
            'normalization_range': [self.data_min, self.data_max],
            'normalization_updates': self.update_count,
            'global_data_range': [self.global_min, self.global_max],
            'episode_data_range': [self.episode_min, self.episode_max],
            'plane_axes': self._plane_axes,
            'window_extents': self._last_window_extents,
            'barriers': {name: float(val) for name, val in zip(self.barrier_names, self._last_voltages[:3])},
            'plunger_center': self._last_voltages[3:].tolist(),
        }
        if meta is not None:
            info.update({'qdarts_meta': meta})
        return info

    def _compute_reward(self, plunger_xy: np.ndarray) -> tuple[float, bool]:
        # Distance-based reward to target_center
        delta = plunger_xy - self.target_center
        distance = float(np.linalg.norm(delta))
        pl_low = np.array([self._action_low[3], self._action_low[4]], dtype=np.float64)
        pl_high = np.array([self._action_high[3], self._action_high[4]], dtype=np.float64)
        max_possible_distance = float(np.linalg.norm(pl_high - pl_low))
        shaped = max(self.distance_weight * (max_possible_distance - distance), 0.0)
        shaped -= self.current_step * self.step_penalty
        success = distance <= self.success_threshold
        if success:
            shaped += self.success_bonus
        return shaped, success

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            super().reset(seed=seed)
        elif self.seed_value is not None:
            super().reset(seed=self.seed_value)
        else:
            super().reset()

        self.current_step = 0
        self._init_normalization_params()

        # Build experiment
        self._build_experiment()

        # Initialize barriers
        default_barriers = self.qdarts_config['simulator']['barrier']['default_barrier_voltages']
        barrier_values = {name: float(default_barriers[name]) for name in self.barrier_names}
        self._set_barriers(barrier_values)

        # Sample initial plunger center
        pl_low = np.array([self._action_low[3], self._action_low[4]], dtype=np.float64)
        pl_high = np.array([self._action_high[3], self._action_high[4]], dtype=np.float64)
        init_center = np.random.uniform(low=pl_low, high=pl_high)

        # Generate initial window
        sensor_values, meta = self._generate_window(init_center)
        self._last_sensor = sensor_values
        self._last_window_extents = (
            float(init_center[0] - self.window_size[0]),
            float(init_center[0] + self.window_size[0]),
            float(init_center[1] - self.window_size[1]),
            float(init_center[1] + self.window_size[1]),
        )

        # Current applied voltages vector
        self._last_voltages = np.array([
            barrier_values[self.barrier_names[0]],
            barrier_values[self.barrier_names[1]],
            barrier_values[self.barrier_names[2]],
            init_center[0],
            init_center[1],
        ], dtype=np.float64)

        observation = self._get_obs()
        info = self._get_info(meta)
        return observation, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (5,):
            raise ValueError(f"Action must have shape (5,), got {action.shape}")
        if self.clip_actions:
            action = np.clip(action, self._action_low, self._action_high)

        self.current_step += 1

        # Split action
        barrier_vals = {
            self.barrier_names[0]: float(action[0]),
            self.barrier_names[1]: float(action[1]),
            self.barrier_names[2]: float(action[2]),
        }
        plunger_xy = np.array([float(action[3]), float(action[4])], dtype=np.float64)

        # Apply barriers
        self._set_barriers(barrier_vals)

        # Generate new window
        sensor_values, meta = self._generate_window(plunger_xy)
        self._last_sensor = sensor_values
        self._last_window_extents = (
            float(plunger_xy[0] - self.window_size[0]),
            float(plunger_xy[0] + self.window_size[0]),
            float(plunger_xy[1] - self.window_size[1]),
            float(plunger_xy[1] + self.window_size[1]),
        )
        self._last_voltages = np.array([
            barrier_vals[self.barrier_names[0]],
            barrier_vals[self.barrier_names[1]],
            barrier_vals[self.barrier_names[2]],
            plunger_xy[0],
            plunger_xy[1],
        ], dtype=np.float64)

        # Reward and termination
        reward, success = self._compute_reward(plunger_xy)
        terminated = bool(success)
        truncated = bool(self.current_step >= self.max_steps)

        # Observation and info
        observation = self._get_obs()
        info = self._get_info(meta)

        return observation, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
            return None
        else:
            return None

    def _render_frame(self):
        z = self._last_sensor
        if z is None:
            return None

        # Normalize for visualization
        channel_data = z[:, :, 0]
        normalized_obs = self._normalize_image(channel_data)
        normalized_data = normalized_obs[:, :, 0]

        vmin_x, vmax_x = self._last_window_extents[0], self._last_window_extents[1]
        vmin_y, vmax_y = self._last_window_extents[2], self._last_window_extents[3]

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(normalized_data, cmap='viridis', aspect='auto', vmin=0.0, vmax=1.0)

        # Axis ticks to voltage values
        num_ticks = 5
        xticks = np.linspace(0, z.shape[1] - 1, num_ticks)
        yticks = np.linspace(0, z.shape[0] - 1, num_ticks)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_xticklabels([f'{v:.2f}' for v in np.linspace(vmin_x, vmax_x, num_ticks)])
        ax.set_yticklabels([f'{v:.2f}' for v in np.linspace(vmin_y, vmax_y, num_ticks)])
        ax.set_xlabel("Plunger X (V)")
        ax.set_ylabel("Plunger Y (V)")
        ax.set_title("Normalized Sensor Response (Agent Observation)")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

        # Annotate barriers
        barrier_text = ", ".join([f"{name}={self._last_voltages[i]:.2f}V" for i, name in enumerate(self.barrier_names)])
        ax.text(0.02, 0.98, barrier_text, transform=ax.transAxes, fontsize=9, va='top', ha='left', color='w', bbox=dict(facecolor='black', alpha=0.3, pad=3))

        if self.render_mode == "human":
            plot_path = os.path.join(self._script_dir, 'qdarts_env_plot.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            if self.debug:
                print(f"Plot saved as '{plot_path}'")
            return None
        else:
            fig.canvas.draw()
            try:
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                buf.seek(0)
                from PIL import Image
                img = Image.open(buf)
                data = np.array(img)
                if data.shape[-1] == 4:
                    data = data[:, :, :3]
            except Exception:
                h, w = normalized_data.shape
                data = np.zeros((h, w, 3), dtype=np.uint8)
                data[:, :, :] = (normalized_data[..., None]).repeat(3, axis=-1)
            plt.close()
            return data

    def close(self):
        pass

    @staticmethod
    def _load_yaml(path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    @staticmethod
    def _set_in_dict(d: dict, key: str, value):
        # Flat override; nested dotted keys supported (e.g., 'env.tolerance')
        if '.' not in key:
            d[key] = value
            return d
        parts = key.split('.')
        cur = d
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value
        return d


if __name__ == "__main__":
    env = QdartsEnv()
    obs, info = env.reset()
    frame = env.render()
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'qdarts_env_frame.png')
    if frame is not None:
        plt.imsave(out_path, frame)
    else:
        print("Rendered in human mode, image saved by renderer.") 