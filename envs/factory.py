"""One task configuration for demonstrations, training, and evaluation."""
from envs.lift_env import make_lift_env
from envs.task_info_wrapper import LiftTaskInfoWrapper
from rewards.adaptive_reward_wrapper import ShapedRewardWrapper
from rewards.adaptive_reward import AdaptiveRewardScheduler
from envs.contact_observation import ContactObservation


def make_env(condition="fixed", *, horizon=500, seed=None, render_mode=None, scheduler_config=None, grasp_hold_steps=1, reward_profile='legacy'):
    """Select rewards while sharing the robot, spaces, and success criterion.

    condition is sparse/fixed/adaptive; horizon counts control steps.
    Return a Gymnasium environment; render_mode='human' opens a native viewer.
    """
    if condition not in ("sparse", "fixed", "adaptive"):
        raise ValueError(f"Unknown reward condition: {condition}")
    # Disable native shaping so both shaped conditions use our shared components.
    env = make_lift_env(reward_shaping=False, horizon=horizon, seed=seed, render_mode=render_mode)
    if condition == "sparse":
        # Keep sparse rewards and add the same skill metrics used for evaluation.
        task = LiftTaskInfoWrapper(env)
        return ContactObservation(task, grasp_hold_steps) if grasp_hold_steps > 1 else task
    scheduler = AdaptiveRewardScheduler(**(scheduler_config or {})) if condition == 'adaptive' else None
    task = ShapedRewardWrapper(env, adaptive=condition == "adaptive", scheduler=scheduler, grasp_hold_steps=grasp_hold_steps, reward_profile=reward_profile)
    return ContactObservation(task, grasp_hold_steps) if grasp_hold_steps > 1 else task
