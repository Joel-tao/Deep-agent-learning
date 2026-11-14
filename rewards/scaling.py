"""A shared reward scale for sparse, fixed, adaptive, and replay rewards."""
import numpy as np

TERM_NAMES = ("reach", "grasp", "lift")
# Online rewards and replay share this component order; success overrides shaping.
SUCCESS_REWARD = 1.0
SHAPING_BUDGET = 0.5
REWARD_SCALE_VERSION = "unit_success_v1"


def normalize_weights(ratios: dict[str, float]) -> dict[str, float]:
    """Preserve relative priorities while keeping the total shaping weight one."""
    values = np.array([ratios[name] for name in TERM_NAMES], dtype=np.float64)
    total = float(values.sum())
    if not np.all(np.isfinite(values)) or np.any(values < 0) or not np.isfinite(total) or total <= 0:
        raise ValueError("Reward weights must be finite, nonnegative, and have positive total")
    return dict(zip(TERM_NAMES, (values / total).tolist()))


def scaled_reward(components, weights, success):
    """Compose one reward or a batch using the same rule online and in replay.

    Components are in [0,1]; weights are nonnegative and sum to one. Thus
    unsuccessful shaped rewards stay in [0,0.5]. Every successful transition
    receives exactly 1, matching robosuite's sparse reward_scale=1.0.
    Reward distributions still differ because sparse feedback has no shaping.
    """
    # @ handles both (3,) and (B,3), keeping online and replay reward formulas identical.
    shaping = np.asarray(components) @ np.asarray(weights)
    return np.where(np.asarray(success, dtype=bool), SUCCESS_REWARD, SHAPING_BUDGET * shaping)
