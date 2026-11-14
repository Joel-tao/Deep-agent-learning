"""Factory for the robosuite Panda Lift task."""
import os

# robosuite 1.5.2 imports numba-cached transforms. In this Conda image numba
# cannot locate its cache reliably; disabling JIT is safe for these helpers.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import robosuite as suite
import gymnasium as gym
import numpy as np
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import GymWrapper


class LiftGymWrapper(gym.Env):
    """Complete the Gymnasium metadata/RNG protocol missing in robosuite 1.5.2."""

    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, env, keys):
        super().__init__()
        self.env = env
        # Compose rather than inherit: robosuite's dummy compute_reward method
        # otherwise makes SB3 misclassify the flat observation as a GoalEnv.
        self._gym_wrapper = GymWrapper(env, keys=keys)
        self.observation_space = self._gym_wrapper.observation_space
        self.action_space = self._gym_wrapper.action_space

    def reset(self, *, seed=None, options=None):
        """Seed Gymnasium's RNG; the outer wrapper seeds the simulator's local RNG."""
        gym.Env.reset(self, seed=seed)
        return self._gym_wrapper.reset(seed=None, options=options)

    def step(self, action):
        return self._gym_wrapper.step(action)

    def render(self):
        if self.render_mode == "human":
            return self.env.render()
        return None

    def close(self):
        self.env.close()


class Float32ObservationWrapper(gym.Wrapper):
    """Adapt dtype, local RNG seeding, clipping, and time-limit semantics."""

    def reset(self, *, seed=None, options=None):
        # GymWrapper seeds legacy np.random, but robosuite uses a local RNG.
        if seed is not None:
            base = self.env.env
            base.seed = seed
            base.rng = np.random.default_rng(seed)
            base.placement_initializer.rng = base.rng
            self.action_space.seed(seed)
        # Do not perturb NumPy's process-wide RNG during held-out evaluation.
        observation, info = self.env.reset(seed=seed, options=options)
        return observation.astype("float32", copy=False), info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        # Reject invalid shapes and NaN/Inf before clipping finite commands.
        if action.shape != self.action_space.shape or not np.all(np.isfinite(action)):
            raise ValueError("Action must be a finite vector matching action_space.shape")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        observation, reward, _terminated, _truncated, info = self.env.step(action)
        # Lift's done is a horizon timeout, which must bootstrap in SAC.
        truncated = self.env.env.timestep >= self.env.env.horizon
        return observation.astype("float32", copy=False), float(reward), False, truncated, info


def make_lift_env(reward_shaping: bool = False, horizon: int = 500,
                  seed: int | None = None, render_mode: str | None = None) -> gym.Env:
    """Create Panda Lift with a 60D state observation and 7D OSC_POSE actions.

    horizon limits control steps; seed controls randomized initialization.
    The default is headless. Human rendering requires a display and OpenGL;
    the default projection GIF demo works without this rendering mode.
    """
    if horizon < 1 or render_mode not in (None, "human"):
        raise ValueError("Use a positive horizon and render_mode=None or 'human'")
    controller = load_composite_controller_config(robot="Panda")
    # Fix full-pose control and detect controller changes after library upgrades.
    assert controller["body_parts"]["right"]["type"] == "OSC_POSE"
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=controller,
        has_renderer=render_mode == "human",
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=reward_shaping,
        reward_scale=1.0,
        horizon=horizon,
        control_freq=20,
        hard_reset=False,
        seed=seed,
    )
    # Concatenate proprioception and object state for every reward condition.
    wrapped = LiftGymWrapper(env, keys=("robot0_proprio-state", "object-state"))
    wrapped.render_mode = render_mode
    return Float32ObservationWrapper(wrapped)
