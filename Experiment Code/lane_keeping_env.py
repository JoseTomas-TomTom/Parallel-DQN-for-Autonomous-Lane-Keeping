# lane_keeping_env.py
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class LaneKeepingEnv(gym.Env):
    """
    Discrete steering lane-keeping environment with:
    - Squared penalties: lateral error, heading error, lateral velocity
    - Soft boundary penalty near lane edges
    - Safety layer: steering clamp + rate limit
    - Lane curvature preview (k_now, k_preview)
    - Noise and randomized lane parameters
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # Vehicle dynamics
        self.dt = 0.05
        self.v = 15.0
        self.wheelbase = 2.7
        self.max_delta = 0.25
        self.max_steps = 600
        self.lane_half_width = 1.8

        # Discrete actions
        self.n_actions = 15
        self.action_space = spaces.Discrete(self.n_actions)
        self.action_map = np.linspace(-self.max_delta, self.max_delta, self.n_actions)

        # Lane curvature parameters
        self.curv_amp_range = (0.0, 0.005)
        self.curv_freq_range = (0.0005, 0.002)
        self.preview_dist = 15.0

        # Noise
        self.lat_noise_std = 0.005
        self.psi_noise_std = 0.001

        # Observation space: y, psi, x, k_now, k_preview
        high = np.array([10.0, np.pi, 1e6, 0.02, 0.02], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.global_steps = 0
        self.last_delta = 0.0
        self.reset()

    # ----- Lane curvature model -----
    def lane_curvature(self, x):
        return self.k_amp * np.sin(self.k_freq * x + self.k_phase)

    # ----- Observation construction -----
    def _get_obs(self):
        y, psi, x = self.state
        k_now = self.lane_curvature(x)
        k_prev = self.lane_curvature(x + self.preview_dist)

        y_n = y + self.np_random.normal(0, self.lat_noise_std)
        psi_n = psi + self.np_random.normal(0, self.psi_noise_std)

        return np.array([y_n, psi_n, x, k_now, k_prev], dtype=np.float32)

    # ----- Reset -----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Random lane curvature
        a_lo, a_hi = self.curv_amp_range
        self.k_amp = self.np_random.uniform(a_lo, a_hi)
        self.k_freq = self.np_random.uniform(*self.curv_freq_range)
        self.k_phase = self.np_random.uniform(0, 2*np.pi)

        # Initial state
        y0 = self.np_random.uniform(-0.5, 0.5)
        psi0 = self.np_random.uniform(-0.1, 0.1)
        x0 = 0.0
        self.state = np.array([y0, psi0, x0], dtype=np.float32)

        self.steps = 0
        self.last_delta = 0.0

        return self._get_obs(), {}

    # ----- Step -----
    def step(self, action):
        cmd = float(self.action_map[int(action)])

        # Safety layer: steering clamp + rate limit
        cmd = np.clip(cmd, -self.max_delta, self.max_delta)
        max_rate = 0.5
        cmd = np.clip(cmd,
                      self.last_delta - max_rate * self.dt,
                      self.last_delta + max_rate * self.dt)

        y, psi, x = self.state
        psi += (self.v / self.wheelbase) * math.tan(cmd) * self.dt
        x += self.v * math.cos(psi) * self.dt
        y += self.v * math.sin(psi) * self.dt

        self.state = np.array([y, psi, x], dtype=np.float32)
        self.last_delta = cmd
        self.steps += 1
        self.global_steps += 1

        y_dot = self.v * math.sin(psi)
        # reward weights
        λy, λψ, λydot = 0.8, 0.25, 0.08
        λδ, λdchange = 0.002, 0.05
        d_change = cmd - self.last_delta

        # soft boundary penalty
        margin = 0.3
        dist_to_edge = self.lane_half_width - abs(y)
        edge_pen = 0 if dist_to_edge > margin else 0.5 * (margin - dist_to_edge) / margin

        progress_bonus = 0.001 * (self.v * math.cos(psi) * self.dt)

        reward = (
            1
            - λy * y**2
            - λψ * psi**2
            - λydot * y_dot**2
            - λδ * cmd**2
            - λdchange * d_change**2
            - edge_pen
            + progress_bonus
        )

        terminated = abs(y) > self.lane_half_width + 0.05 or abs(psi) > np.pi / 2
        truncated = self.steps >= self.max_steps

        info = {"y": y, "psi": psi, "x": x, "delta": cmd}
        return self._get_obs(), float(reward), terminated, truncated, info
