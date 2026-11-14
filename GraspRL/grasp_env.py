"""Gymnasium adapter with project-specific rewards for robosuite Panda Lift."""
import gymnasium as gym
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from adaptive_scheduler import AdaptiveRewardScheduler
from reward_functions import (compute_reach_reward, compute_grasp_reward,
                              compute_lift_reward, combine_reward)


class AdaptiveGraspEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(self, reward_mode="fixed", render_mode=None, horizon=200,
                 fixed_weights=(0.3, 0.3, 0.4), reach_threshold=0.05,
                 scheduler=None, width=640, height=480, camera_name="frontview"):
        if reward_mode not in ("sparse", "fixed", "adaptive"):
            raise ValueError("reward_mode must be sparse, fixed, or adaptive")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("Unsupported render mode")
        self.reward_mode, self.render_mode = reward_mode, render_mode
        self.horizon, self.reach_threshold = int(horizon), float(reach_threshold)
        if self.horizon < 1 or self.reach_threshold <= 0:
            raise ValueError("horizon and reach_threshold must be positive")
        self.fixed_weights = np.array(fixed_weights, dtype=np.float32)
        if self.fixed_weights.shape != (3,) or not np.isfinite(self.fixed_weights).all() or np.any(self.fixed_weights < 0) or not np.isclose(self.fixed_weights.sum(), 1):
            raise ValueError("fixed_weights must be three nonnegative numbers summing to one")
        self.scheduler = scheduler or AdaptiveRewardScheduler()
        self.width, self.height = width, height
        self.camera_name = camera_name
        config = load_composite_controller_config(controller="BASIC")
        config["body_parts"] = {"right": {"type": "OSC_POSE", "input_max": 1,
            "input_min": -1, "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            "kp": 150, "damping_ratio": 1, "impedance_mode": "fixed",
            "kp_limits": [0, 300], "damping_ratio_limits": [0, 10],
            "position_limits": None, "orientation_limits": None,
            "uncouple_pos_ori": True, "input_type": "delta", "input_ref_frame": "world",
            "interpolation": None, "ramp_ratio": 0.2, "gripper": {"type": "GRIP"}}}
        self.sim_env = suite.make("Lift", robots="Panda", controller_configs=config,
            has_renderer=render_mode == "human", renderer="mjviewer", render_camera=camera_name,
            has_offscreen_renderer=render_mode == "rgb_array", use_camera_obs=False,
            use_object_obs=True, reward_shaping=False, horizon=self.horizon,
            control_freq=20, hard_reset=False, initialization_noise=None, seed=0,
            camera_names=camera_name, camera_heights=height, camera_widths=width)
        self.observation_keys = ("robot0_proprio-state", "object-state")
        raw = self.sim_env.reset()
        obs = self._flatten(raw)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype=np.float32)
        low, high = self.sim_env.action_spec
        self.action_space = gym.spaces.Box(low.astype(np.float32), high.astype(np.float32), dtype=np.float32)
        self._done = True
        self._steps = 0
        self._episode_weights = self.fixed_weights.copy()

    def _flatten(self, obs):
        return np.concatenate([np.asarray(obs[k]).ravel() for k in self.observation_keys]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if options.get("reset_scheduler", False):
            self.scheduler.reset()
        if seed is not None:
            self.sim_env.rng = np.random.default_rng(seed)
            self.sim_env.placement_initializer.rng = self.sim_env.rng
        raw = self.sim_env.reset()
        self._initial_z = float(raw["cube_pos"][2])
        self._steps, self._done = 0, False
        self._stage_ever = np.zeros(3, dtype=bool)
        self._episode_weights = (self.scheduler.weights if self.reward_mode == "adaptive" else self.fixed_weights.copy())
        return self._flatten(raw), {"reward_mode": self.reward_mode,
                                    "weights": self._episode_weights.copy()}

    def _task_values(self, raw):
        reach, distance = compute_reach_reward(raw["robot0_eef_pos"], raw["cube_pos"])
        robot = self.sim_env.robots[0]
        contact = bool(self.sim_env._check_grasp(gripper=robot.gripper["right"],
                                               object_geoms=self.sim_env.cube.contact_geoms))
        success = bool(self.sim_env._check_success())
        lift = compute_lift_reward(raw["cube_pos"][2], self._initial_z, success)
        return np.array([reach, compute_grasp_reward(contact), lift], dtype=np.float32), distance, contact, success

    def step(self, action):
        if self._done:
            raise RuntimeError("Call reset() before stepping or after episode end")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("Action must have the correct shape and contain finite numbers")
        raw, _, _, _ = self.sim_env.step(np.clip(action, self.action_space.low, self.action_space.high))
        self._steps += 1
        components, distance, contact, success = self._task_values(raw)
        stages = np.array([distance <= self.reach_threshold, contact, success])
        self._stage_ever |= stages
        reward = combine_reward(self.reward_mode, components, self._episode_weights, success)
        terminated, truncated = success, self._steps >= self.horizon and not success
        self._done = terminated or truncated
        info = {"is_success": success, "reach_distance": distance, "valid_grasp": contact,
                "cube_height": float(raw["cube_pos"][2]), "reward_components": components.copy(),
                "weights": self._episode_weights.copy(), "stage_ever": self._stage_ever.copy(),
                "reward_mode": self.reward_mode}
        if self._done and self.reward_mode == "adaptive":
            info["scheduler"] = self.scheduler.record(*self._stage_ever)
        if self.render_mode == "human":
            self.render()
        return self._flatten(raw), reward, bool(terminated), bool(truncated), info

    def render(self):
        if self.render_mode == "human":
            return self.sim_env.render()
        if self.render_mode == "rgb_array":
            return self.sim_env.sim.render(width=self.width, height=self.height,
                                           camera_name=self.camera_name)[::-1].copy()
        return None

    def close(self):
        self.sim_env.close()


if "AdaptiveGrasp-v0" not in gym.registry:
    gym.register("AdaptiveGrasp-v0", entry_point="grasp_env:AdaptiveGraspEnv")
