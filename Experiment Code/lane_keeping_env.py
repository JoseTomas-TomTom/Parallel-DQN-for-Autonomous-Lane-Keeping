import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor


class LaneKeepingEnv(gym.Env):
    """
    Discrete steering (DQN) lane keeping with:
    - Squared penalties (y, psi, y_dot) + soft boundary penalty + progress bonus
    - Lane curvature (now + preview) in observation
    - Sensor noise, randomized lane shapes, and light curriculum on curvature amplitude
    - Safety layer: steering clamp + rate limit
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, random_lane=True, random_noise=True):
        super().__init__()
        # Flags to control randomness (for training vs fixed test)
        self.random_lane = random_lane
        self.random_noise = random_noise

        # Dynamics
        self.dt = 0.05          # s
        self.v = 15.0           # m/s
        self.wheelbase = 2.7    # m
        self.max_delta = 0.25   # rad (steer)
        self.lane_half_width = 1.8
        self.max_steps = 600

        # Discrete actions
        self.n_actions = 15
        self.action_space = spaces.Discrete(self.n_actions)
        self.action_map = np.linspace(-self.max_delta, self.max_delta, self.n_actions)

        # Lane (curvature) + noise
        self.curv_amp_range = (0.0, 0.005)        # rad/m
        self.curv_freq_range = (0.0005, 0.002)    # cycles per meter
        self.lat_noise_std = 0.005                # m
        self.psi_noise_std = 0.001                # rad
        self.preview_dist = 15.0                  # m

        # Observation: [y, psi, x, k_now, k_preview]
        high = np.array([10.0, np.pi, 1e6, 0.02, 0.02], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # For curriculum & safety
        self.global_steps = 0
        self.last_delta = 0.0
        self.reset()

    # ----- Lane model -----
    def lane_curvature(self, x):
        return self.k_amp * np.sin(self.k_freq * x + self.k_phase)

    def _get_obs(self):
        y, psi, x = self.state
        k_now = self.lane_curvature(x)
        k_prev = self.lane_curvature(x + self.preview_dist)

        # Sensor noise (can be turned off for deterministic tests)
        if self.random_noise:
            y_n = y + self.np_random.normal(0, self.lat_noise_std)
            psi_n = psi + self.np_random.normal(0, self.psi_noise_std)
        else:
            y_n, psi_n = y, psi

        return np.array([y_n, psi_n, x, k_now, k_prev], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.random_lane:
            # Light curriculum on curvature amplitude over training time
            max_amp = float(np.interp(self.global_steps, [0, 1_000_000], [0.0, 0.008]))
            a_lo, a_hi = self.curv_amp_range
            self.k_amp = self.np_random.uniform(a_lo, max(a_hi, max_amp))
            self.k_freq = self.np_random.uniform(*self.curv_freq_range)
            self.k_phase = self.np_random.uniform(0, 2*np.pi)

            # Random initial state
            y0   = self.np_random.uniform(-0.5, 0.5)
            psi0 = self.np_random.uniform(-0.1, 0.1)
        else:
            # Fixed test lane (deterministic for fair comparison)
            self.k_amp = 0.004
            self.k_freq = 0.001
            self.k_phase = 0.0

            # Fixed starting pose
            y0, psi0 = 0.0, 0.0

        x0 = 0.0
        self.state = np.array([y0, psi0, x0], dtype=np.float32)

        self.steps = 0
        self.last_delta = 0.0
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        # Map action -> steering command
        cmd = float(self.action_map[int(action)])

        # Safety layer: clamp and rate-limit
        cmd = np.clip(cmd, -self.max_delta, self.max_delta)
        max_rate = 0.5  # rad/s
        cmd = np.clip(cmd,
                      self.last_delta - max_rate*self.dt,
                      self.last_delta + max_rate*self.dt)

        # Kinematics
        y, psi, x = self.state
        psi += (self.v / self.wheelbase) * math.tan(cmd) * self.dt
        x   += self.v * math.cos(psi) * self.dt
        y   += self.v * math.sin(psi) * self.dt
        self.state = np.array([y, psi, x], dtype=np.float32)

        self.global_steps += 1
        self.steps += 1

        # Reward terms
        y_dot = self.v * math.sin(psi)
        lambda_y, lambda_psi, lambda_ydot = 0.8, 0.25, 0.08
        lambda_delta, lambda_dchange = 0.002, 0.05
        d_change = cmd - self.last_delta

        # Soft boundary penalty
        margin = 0.3
        dist_to_edge = self.lane_half_width - abs(y)
        edge_pen = 0.0 if dist_to_edge > margin else \
            0.5 * (margin - dist_to_edge) / margin

        # Progress bonus
        progress_bonus = 0.001 * (self.v * math.cos(psi) * self.dt)

        reward = (
            1.0
            - lambda_y*(y**2)
            - lambda_psi*(psi**2)
            - lambda_ydot*(y_dot**2)
            - lambda_delta*(cmd**2)
            - lambda_dchange*(d_change**2)
            - edge_pen
            + progress_bonus
        )

        self.last_delta = cmd

        # Termination
        terminated = (abs(y) > self.lane_half_width + 0.05) or (abs(psi) > math.pi/2)
        truncated  = self.steps >= self.max_steps

        info = {"y": y, "psi": psi, "x": x, "delta": cmd}
        return self._get_obs(), float(reward), terminated, truncated, info


# ===== Vec env builders (used by train & simulate) =====
def make_env(seed_offset=0):
    """Training env: random lanes + noise."""
    def _init():
        env = LaneKeepingEnv(random_lane=True, random_noise=True)
        env = Monitor(env)
        env.reset(seed=seed_offset)
        return env
    return _init


def make_test_env(seed_offset=0):
    """Test env: fixed lane + no noise (for consistent graphs)."""
    def _init():
        env = LaneKeepingEnv(random_lane=False, random_noise=False)
        env = Monitor(env)
        env.reset(seed=seed_offset)
        return env
    return _init
