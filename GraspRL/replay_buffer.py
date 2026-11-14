"""Relabel sampled shaped rewards from saved components and current weights."""
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer


class ComponentReplayBuffer(ReplayBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.components = np.zeros((self.buffer_size, self.n_envs, 3), dtype=np.float32)
        self.reward_weights = np.array([0.6, 0.3, 0.1], dtype=np.float32)

    def add(self, obs, next_obs, action, reward, done, infos):
        self.components[self.pos] = np.asarray([i["reward_components"] for i in infos])
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds, env=None):
        # SB3's single-env sample is used here; multi-env indexing would need
        # to share sampled env indices with the parent implementation.
        if self.n_envs != 1:
            raise ValueError("ComponentReplayBuffer supports one environment only")
        samples = super()._get_samples(batch_inds, env)
        rewards = (self.components[batch_inds, 0] @ self.reward_weights).reshape(-1, 1)
        return samples._replace(rewards=self.to_torch(self._normalize_reward(rewards, env)))
