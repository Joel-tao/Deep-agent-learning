"""Optional observation normalization; task rewards are never normalized here."""
import gymnasium as gym
import numpy as np
from stable_baselines3.common.running_mean_std import RunningMeanStd


class ObservationNormalization(gym.ObservationWrapper):
    """Share training statistics with evaluation without updating them there.

    Joint velocity/acceleration and Cartesian position have very different
    numerical scales. This opt-in training wrapper standardizes policy inputs
    while preserving the common reward scale and all physical task criteria.
    """

    def __init__(self, env, *, stats=None, update=True, clip=10.0):
        super().__init__(env)
        self.stats = stats if stats is not None else RunningMeanStd(shape=env.observation_space.shape)
        self.update = update
        self.clip = clip
        self.observation_space = gym.spaces.Box(-clip, clip, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        """Standardize inputs; use update=False to freeze statistics for evaluation."""
        observation = np.asarray(observation, dtype=np.float64)
        if self.update:
            self.stats.update(observation[None, :])
        # Avoid division by zero and clip outliers; scalar rewards stay unchanged.
        normalized = (observation - self.stats.mean) / np.sqrt(self.stats.var + 1e-8)
        return np.clip(normalized, -self.clip, self.clip).astype(np.float32)
