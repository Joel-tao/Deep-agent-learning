"""Episode-based reward curriculum with warm-up and consecutive confirmations."""
from collections import deque
from rewards.scaling import normalize_weights


class AdaptiveRewardScheduler:
    """Advance Reach -> Grasp -> Lift after two confirmed updates by default.

    Weights stay constant within each episode. Monotonic progression avoids
    oscillation; regression-aware scheduling remains a possible extension.
    """

    # Relative ratios for the Reach/Grasp/Lift stages are normalized before use.
    WEIGHT_RATIOS = (
        {"reach": 1.0, "grasp": 0.2, "lift": 0.1},
        {"reach": 0.3, "grasp": 1.0, "lift": 0.3},
        {"reach": 0.1, "grasp": 0.3, "lift": 1.0},
    )
    WEIGHTS = tuple(normalize_weights(ratios) for ratios in WEIGHT_RATIOS)

    def __init__(self, window_size=100, reach_threshold=0.7, grasp_threshold=0.7,
                 update_interval=10, confirmations=2, transition_episodes=0,
                 grasp_stage_lift_ratio=0.3):
        if min(window_size, update_interval, confirmations) < 1:
            raise ValueError("Window, update interval, and confirmations must be positive")
        if not all(0 <= t <= 1 for t in (reach_threshold, grasp_threshold)):
            raise ValueError("Success thresholds must be in [0, 1]")
        if transition_episodes < 0 or grasp_stage_lift_ratio < 0:
            raise ValueError('Transition length and Lift ratio must be nonnegative')
        ratios = [row.copy() for row in self.WEIGHT_RATIOS]
        ratios[1]['lift'] = grasp_stage_lift_ratio
        self.stage_weights = tuple(normalize_weights(row) for row in ratios)
        self.transition_episodes = transition_episodes
        self.transition_remaining = 0
        self.current_weights = self.stage_weights[0].copy()
        self.window_size = window_size
        self.update_interval = update_interval
        self.confirmations = confirmations
        self.reach_threshold = reach_threshold
        self.grasp_threshold = grasp_threshold
        # Bounded deques retain recent completed episodes; each entry is one episode.
        self.reach_history = deque(maxlen=window_size)
        self.grasp_history = deque(maxlen=window_size)
        self.lift_history = deque(maxlen=window_size)
        self.episodes = 0
        self.stage = 0
        self.pending_confirmations = 0

    @staticmethod
    def _rate(history):
        return float(sum(history) / len(history)) if history else 0.0

    def __setstate__(self, state):
        """Migrate historical checkpoints without changing their active weights."""
        self.__dict__.update(state)
        if 'current_weights' not in state:
            self.stage_weights = self.WEIGHTS
            self.current_weights = self.WEIGHTS[self.stage].copy()
            self.transition_episodes = self.transition_remaining = 0

    def update(self, reached: bool, grasped: bool, lifted: bool) -> None:
        """Append one completed episode and advance one stage after confirmation.

        Call after an episode ends. Stage changes affect subsequent online
        rewards; replay rewards use the current weights at sampling time.
        """
        self.reach_history.append(bool(reached))
        self.grasp_history.append(bool(grasped))
        self.lift_history.append(bool(lifted))
        self.episodes += 1
        # Advance at completed-episode boundaries only. Do not chain a second
        # stage switch while the current interpolation is still in progress.
        if self.transition_remaining:
            self._advance_transition()
            return
        # Check only at the chosen interval after warm-up; default first check: episode 100.
        if len(self.reach_history) < self.window_size or self.episodes % self.update_interval:
            return
        if self.stage == 2:
            # Keep logging history in the final stage without advancing or regressing.
            return
        rates = self.get_rates()
        ready = rates["reach_rate"] >= self.reach_threshold
        if self.stage == 1:
            # Require retained reaching ability as well as grasping before advancing.
            ready &= rates["grasp_rate"] >= self.grasp_threshold
        # A failed check resets the count; confirmations must be consecutive.
        self.pending_confirmations = self.pending_confirmations + 1 if ready else 0
        if self.pending_confirmations >= self.confirmations:
            self.transition_start = self.get_weights()
            self.stage += 1
            self.pending_confirmations = 0
            if self.transition_episodes:
                self.transition_remaining = self.transition_episodes
                self._advance_transition()
            else:
                self.current_weights = self.stage_weights[self.stage].copy()

    def _advance_transition(self):
        fraction = (self.transition_episodes - self.transition_remaining + 1) / self.transition_episodes
        target = self.stage_weights[self.stage]
        self.current_weights = {k:(1-fraction)*self.transition_start[k]+fraction*target[k] for k in target}
        self.transition_remaining -= 1

    def get_rates(self) -> dict[str, float]:
        """Return window skill rates; partial-window rates are for diagnostics only."""
        return {"reach_rate": self._rate(self.reach_history),
                "grasp_rate": self._rate(self.grasp_history),
                "lift_rate": self._rate(self.lift_history)}

    def get_weights(self) -> dict[str, float]:
        """Return a copy of normalized weights to protect the stage configuration."""
        # Old checkpoints contain only stage; preserve their loading semantics.
        return getattr(self, 'current_weights', self.WEIGHTS[self.stage]).copy()
