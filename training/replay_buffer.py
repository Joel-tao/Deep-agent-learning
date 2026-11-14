"""SB3 replay extension: relabel old components with current curriculum weights."""
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from rewards.scaling import TERM_NAMES, scaled_reward


class ComponentReplayBuffer(ReplayBuffer):
    """Single-environment SAC buffer retaining SB3 timeout handling and sampling."""

    def __init__(self, *args, scheduler, **kwargs):
        super().__init__(*args, **kwargs)
        if self.n_envs != 1:
            raise ValueError("ComponentReplayBuffer currently supports one training environment")
        self.scheduler = scheduler
        # Align with SB3's ring indices and store components before applying any weights.
        self.components = np.zeros((self.buffer_size, 3), dtype=np.float32)
        self.successes = np.zeros(self.buffer_size, dtype=bool)

    def add(self, obs, next_obs, action, reward, done, infos):
        """Store components/success, then let SB3 store states, actions, and timeouts."""
        terms = infos[0]["reward_terms"]
        # Write before super().add advances or wraps self.pos.
        self.components[self.pos] = [terms[name] for name in TERM_NAMES]
        self.successes[self.pos] = infos[0]["is_success"]
        super().add(obs, next_obs, action, reward, done, infos)

    def _get_samples(self, batch_inds, env=None):
        """Reuse SB3 sampling and termination masks, replacing only scalar rewards."""
        # n_envs=1 ensures SB3's chosen environment index is always zero.
        samples = super()._get_samples(batch_inds, env=env)
        # Relabel with current curriculum weights rather than weights at insertion time.
        weights = self.scheduler.get_weights()
        vector = np.array([weights[name] for name in TERM_NAMES], dtype=np.float32)
        rewards = scaled_reward(self.components[batch_inds], vector,
                                self.successes[batch_inds]).astype(np.float32).reshape(-1, 1)
        return samples._replace(rewards=self.to_torch(self._normalize_reward(rewards, env)))
