"""Interpretable reward terms for Panda Lift."""
import numpy as np


def sustained_grasp_reward(contact_run, target_steps):
    """Ramp up contact reward over uninterrupted steps, resetting on contact loss."""
    if target_steps < 1:
        raise ValueError('target_steps must be positive')
    return float(np.clip(contact_run / target_steps, 0.0, 1.0))


def reach_reward(eef_pos: np.ndarray, cube_pos: np.ndarray, scale: float = 10.0) -> float:
    """Compute reach shaping from positions in meters; smaller distance approaches 1."""
    return float(1.0 - np.tanh(scale * np.linalg.norm(eef_pos - cube_pos)))


def grasp_reward(grasped: bool) -> float:
    """Convert the simulator's valid bilateral-contact flag to a 0/1 component."""
    return float(grasped)


def lift_reward(cube_height: float, resting_height: float, target_height: float = 0.04) -> float:
    """Compute progress above the resting cube center; all quantities are in meters.

    target_height is the required displacement: success center minus resting
    center. This function measures geometry; the caller gates it by valid grasp.
    """
    if target_height <= 0:
        raise ValueError("target_height must be positive")
    return float(np.clip((cube_height - resting_height) / target_height, 0.0, 1.0))
