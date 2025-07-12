import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CloudCostEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.max_usage = 1.0
        self.max_cost = 1.0
        self.max_cpu = 1.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.usage = 0.0
        self.cost = 0.0
        self.cpu = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.usage = float(np.random.uniform(0.1, 0.9))
        self.cost = float(self.usage * 0.8)
        self.cpu = float(np.random.uniform(0.1, 0.9))
        obs = [self.usage, self.cost, self.cpu]
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        action_value = float(action[0]) if isinstance(action, (list, np.ndarray)) else float(action)
        self.usage = float(np.clip(self.usage + action_value * 0.1, 0, 1))
        self.cost = float(self.usage * (0.7 + float(np.random.rand()) * 0.3))
        self.cpu = float(np.clip(self.cpu + action_value * 0.05, 0, 1))
        obs = [self.usage, self.cost, self.cpu]
        reward = float(-1 * self.cost - 0.5 * (1 - self.cpu))
        terminated = bool(self.cost > 0.9 or self.cpu > 0.95)
        truncated = False
        info = {}
        return np.array(obs, dtype=np.float32), reward, terminated, truncated, info
