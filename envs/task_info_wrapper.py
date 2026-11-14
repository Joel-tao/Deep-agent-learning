"""Expose version-specific robosuite Lift task signals through Gymnasium info."""
import gymnasium as gym
import numpy as np
from envs.skill_diagnostics import SkillDiagnostics


class LiftTaskInfoWrapper(gym.Wrapper):
    """Add reach, grasp, and success signals while preserving the environment reward."""

    def __init__(self, env: gym.Env, reach_distance: float = 0.06):
        super().__init__(env)
        if reach_distance <= 0:
            raise ValueError("reach_distance must be positive")
        self.reach_distance = reach_distance
        self.episode_reached = False
        self.episode_grasped = False
        self.episode_lifted = False
        self.diagnostics = SkillDiagnostics()

    @property
    def base_env(self):
        """Unwrap composition layers to access robosuite and its MuJoCo model."""
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        return env

    def reset(self, *, seed=None, options=None):
        self.episode_reached = self.episode_grasped = self.episode_lifted = False
        self.diagnostics.reset()
        return self.env.reset(seed=seed, options=options)

    def task_state(self) -> tuple[np.ndarray, np.ndarray, float, float, bool, bool]:
        """Return EEF/cube positions, cube/table heights, grasp, and Lift flags.

        Positions are in meters. Use the pinned robosuite contact/success checks.
        Closing the gripper alone does not imply a grasp; instantaneous Lift
        does not imply that a grasp has been maintained.
        """
        base = self.base_env
        eef_site_id = next(iter(base.robots[0].eef_site_id.values()))
        eef_pos = np.asarray(base.sim.data.site_xpos[eef_site_id]).copy()
        cube_pos = np.asarray(base.sim.data.body_xpos[base.cube_body_id]).copy()
        cube_height = float(cube_pos[2])
        table_height = float(base.model.mujoco_arena.table_offset[2])
        grasped = bool(base._check_grasp(base.robots[0].gripper, base.cube))
        lifted = bool(base._check_success())
        return eef_pos, cube_pos, cube_height, table_height, grasped, lifted

    def task_geometry(self) -> tuple[float, float]:
        """Return physical resting center and Lift's verified success height."""
        base = self.base_env
        table = float(base.model.mujoco_arena.table_offset[2])
        # Subtract the resting center offset so a stationary cube earns no lift progress.
        resting_center = table - float(base.cube.bottom_offset[2])
        return resting_center, table + 0.04

    def add_task_info(self, info: dict, terminated: bool, truncated: bool, action=None) -> tuple[dict, tuple]:
        """Add step/episode skill flags and return info plus the physical state."""
        state = self.task_state()
        eef_pos, cube_pos, _cube_height, _table_height, grasped, lifted = state
        reached = bool(np.linalg.norm(eef_pos - cube_pos) < self.reach_distance)
        # |= tracks any occurrence; evaluation separately counts consecutive held steps.
        self.episode_reached |= reached
        self.episode_grasped |= grasped
        self.episode_lifted |= lifted
        if action is not None:
            resting, _ = self.task_geometry()
            self.diagnostics.update(grasped, lifted, _cube_height - resting, action)
        info.update({"reached": reached, "grasped": grasped, "is_success": lifted})
        if terminated or truncated:
            # Publish completed-episode outcomes for the curriculum and training logs.
            info.update({
                "episode_reached": self.episode_reached,
                "episode_grasped": self.episode_grasped,
                "episode_success": self.episode_lifted,
            })
            info['diagnostics'] = self.diagnostics.summary()
        return info, state

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info, _ = self.add_task_info(info, terminated, truncated, action)
        return observation, reward, terminated, truncated, info
