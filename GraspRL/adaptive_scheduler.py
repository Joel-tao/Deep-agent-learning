"""A full episode window, periodic updates, and consecutive confirmations."""
from collections import deque
import numpy as np


class AdaptiveRewardScheduler:
    weight_vectors = ((0.6, 0.3, 0.1), (0.2, 0.6, 0.2), (0.1, 0.2, 0.7))

    def __init__(self, window_size=20, update_interval=5, confirmations=2,
                 reach_threshold=0.8, grasp_threshold=0.6):
        if min(window_size, update_interval, confirmations) < 1:
            raise ValueError("Window, interval and confirmations must be positive")
        if not all(0 <= x <= 1 for x in (reach_threshold, grasp_threshold)):
            raise ValueError("Thresholds must be in [0, 1]")
        self.window_size, self.update_interval = window_size, update_interval
        self.confirmations = confirmations
        self.reach_threshold, self.grasp_threshold = reach_threshold, grasp_threshold
        self.history = deque(maxlen=window_size)
        self.reset()

    @property
    def weights(self):
        return np.array(self.weight_vectors[self.stage], dtype=np.float32)

    def reset(self):
        self.history.clear()
        self.episodes = self.stage = self._pending_stage = self._confirm_count = 0

    def record(self, reach, grasp, lift):
        self.history.append((bool(reach), bool(grasp), bool(lift)))
        self.episodes += 1
        if len(self.history) == self.window_size and self.episodes % self.update_interval == 0:
            rates = np.mean(self.history, axis=0)
            candidate = (2 if rates[0] >= self.reach_threshold and rates[1] >= self.grasp_threshold
                         else 1 if rates[0] >= self.reach_threshold else 0)
            if candidate == self.stage:
                self._pending_stage, self._confirm_count = self.stage, 0
            else:
                self._confirm_count = self._confirm_count + 1 if candidate == self._pending_stage else 1
                self._pending_stage = candidate
                if self._confirm_count >= self.confirmations:
                    self.stage, self._confirm_count = candidate, 0
        return self.snapshot()

    def snapshot(self):
        rates = np.mean(self.history, axis=0).tolist() if self.history else [0.0] * 3
        return {"stage": self.stage, "weights": self.weights.tolist(),
                "episodes": self.episodes, "window_count": len(self.history), "rates": rates}
