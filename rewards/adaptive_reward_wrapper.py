"""Common reward components shared by the fixed and adaptive conditions."""
import gymnasium as gym
from envs.task_info_wrapper import LiftTaskInfoWrapper
from rewards.adaptive_reward import AdaptiveRewardScheduler
from rewards.components import grasp_reward, lift_reward, reach_reward
from rewards.components import sustained_grasp_reward
from rewards.preparation import preparation_terms
from rewards.scaling import SUCCESS_REWARD, TERM_NAMES, normalize_weights, scaled_reward


FIXED_WEIGHT_RATIOS = {"reach": 0.3, "grasp": 1.0, "lift": 1.0}
FIXED_WEIGHTS = normalize_weights(FIXED_WEIGHT_RATIOS)


class ShapedRewardWrapper(LiftTaskInfoWrapper):
    """Combine distance, bilateral contact, and grasp-gated upward progress.

    Success overrides shaping with reward 1; otherwise normalized shaping has
    a fixed budget of 0.5, regardless of fixed/adaptive condition or stage.
    """

    def __init__(self, env: gym.Env, *, adaptive=False, reach_distance=0.06,
                 scheduler=None, grasp_hold_steps=1, reward_profile='legacy'):
        super().__init__(env, reach_distance=reach_distance)
        self.scheduler = (scheduler or AdaptiveRewardScheduler()) if adaptive else None
        if grasp_hold_steps < 1:
            raise ValueError('grasp_hold_steps must be positive')
        self.grasp_hold_steps = grasp_hold_steps
        if reward_profile not in ('legacy','prepare_v1'):
            raise ValueError('Unknown reward profile')
        self.reward_profile=reward_profile

    def step(self, action):
        """Advance physics, replace rewards with shared components, and log curriculum state."""
        obs, _original_reward, terminated, truncated, info = self.env.step(action)
        info, state = self.add_task_info(info, terminated, truncated, action)
        eef_pos, cube_pos, cube_height, _table_height, grasped, lifted = state
        resting, success_height = self.task_geometry()
        # Capture weights before updating the curriculum, including the terminal step.
        weights = self.scheduler.get_weights() if self.scheduler else FIXED_WEIGHTS.copy()
        if self.scheduler:
            # This is the stage that actually produced this step's weights.
            info["scheduler_stage"] = self.scheduler.stage
        terms = {
            "reach": reach_reward(eef_pos, cube_pos),
            "grasp": sustained_grasp_reward(self.diagnostics.contact_run, self.grasp_hold_steps),
            # Gate progress by grasp contact so collision-induced bounces earn no lift shaping.
            "lift": lift_reward(cube_height, resting, success_height - resting) if grasped else 0.0,
        }
        if self.reward_profile=='prepare_v1':
            robot=self.base_env.robots[0]
            fingers=self.base_env.sim.data.qpos[next(iter(robot._ref_gripper_joint_pos_indexes.values()))]
            prepared,diagnostic=preparation_terms(eef_pos,cube_pos,abs(float(fingers[0]-fingers[1])),
                float(action[-1]),grasped,terms['grasp'],terms['reach'])
            terms.update(prepared)
            info['preparation']=diagnostic
        reward = float(scaled_reward([terms[name] for name in TERM_NAMES],
                                     [weights[name] for name in TERM_NAMES], lifted))
        info.update({"reward_terms": terms, "reward_weights": weights,
                     "success_reward": SUCCESS_REWARD * float(lifted)})
        if (terminated or truncated) and self.scheduler:
            # Use skill occurrences across the full episode, rather than its final frame.
            # The terminal transition's historical reward uses the old weights.
            self.scheduler.update(self.episode_reached, self.episode_grasped, self.episode_lifted)
            info["skill_rates"] = self.scheduler.get_rates()
            info["next_scheduler_stage"] = self.scheduler.stage
        return obs, reward, terminated, truncated, info


class AdaptiveRewardWrapper(ShapedRewardWrapper):
    """Compatibility entry point for the common shaped wrapper."""

    def __init__(self, env, **kwargs):
        super().__init__(env, adaptive=True, **kwargs)
