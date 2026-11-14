"""Reward components, separated from physics for testing and inspection."""
import numpy as np


def compute_reach_reward(eef_pos, cube_pos, k_d=5.0):
    distance = float(np.linalg.norm(np.asarray(eef_pos) - np.asarray(cube_pos)))
    return float(1.0 - np.tanh(k_d * distance)), distance


def compute_grasp_reward(valid_contact):
    """A closed gripper alone is insufficient; caller checks both finger groups."""
    return float(bool(valid_contact))


def compute_lift_reward(cube_z, initial_z, success, target_height=0.04):
    if target_height <= 0:
        raise ValueError("target_height must be positive")
    return 1.0 if success else float(np.clip((cube_z - initial_z) / target_height, 0, 1))


def combine_reward(mode, components, weights, success):
    if mode not in ("sparse", "fixed", "adaptive"):
        raise ValueError(f"Unknown reward mode: {mode}")
    return float(success) if mode == "sparse" else float(np.dot(weights, components))
