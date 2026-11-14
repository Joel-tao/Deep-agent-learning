import numpy as np
import gymnasium as gym
import torch
from replay_buffer import ComponentReplayBuffer


def test_relabel_uses_current_weights_and_preserves_timeout_mask():
    b = ComponentReplayBuffer(10, gym.spaces.Box(-1, 1, (2,), dtype=np.float32),
                              gym.spaces.Box(-1, 1, (1,), dtype=np.float32), device="cpu")
    z = np.zeros((1, 2), dtype=np.float32)
    b.add(z, z, np.zeros((1, 1)), np.array([0.99]), np.array([True]),
          [{"reward_components": [1, 0, 0], "TimeLimit.truncated": True}])
    b.reward_weights = np.array([0.2, 0.6, 0.2], dtype=np.float32)
    samples = b.sample(4)
    torch.testing.assert_close(samples.rewards, torch.full((4, 1), 0.2))
    assert samples.dones.sum().item() == 0  # timeout must bootstrap
    b.reward_weights = np.array([0.1, 0.2, 0.7], dtype=np.float32)
    torch.testing.assert_close(b.sample(4).rewards, torch.full((4, 1), 0.1))
