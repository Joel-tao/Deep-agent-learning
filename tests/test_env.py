"""Integration tests against actual Panda MuJoCo dynamics."""
import numpy as np
import pytest
from envs.factory import make_env
from rewards.scaling import TERM_NAMES, scaled_reward
from envs.lift_env import make_lift_env
from rewards.adaptive_reward import AdaptiveRewardScheduler
from rewards.adaptive_reward_wrapper import ShapedRewardWrapper


def test_stage_log_matches_reward_weights_at_curriculum_transition():
    scheduler = AdaptiveRewardScheduler(window_size=1, update_interval=1, confirmations=1)
    with ShapedRewardWrapper(make_lift_env(horizon=1, seed=7), adaptive=True,
                             scheduler=scheduler, reach_distance=1.0) as env:
        env.reset(seed=7)
        _, _, _, _, info = env.step(np.zeros(7))
        assert info['scheduler_stage'] == 0
        assert info['reward_weights'] == scheduler.WEIGHTS[0]
        assert info['next_scheduler_stage'] == 1
        env.reset(seed=7)
        _, _, _, _, info = env.step(np.zeros(7))
        assert info['scheduler_stage'] == 1
        assert info['reward_weights'] == scheduler.WEIGHTS[1]


def test_observation_seed_and_time_limit_semantics():
    with make_env('sparse', horizon=2, seed=7) as env:
        obs, _ = env.reset(seed=7)
        cube = env.task_state()[1].copy()
        assert obs.shape == (60,)
        assert env.action_space.shape == (7,)
        assert env.observation_space.contains(obs)
        env.step(env.action_space.sample())
        repeated, _ = env.reset(seed=7)
        np.testing.assert_allclose(obs, repeated, atol=1e-7)
        np.testing.assert_allclose(cube, env.task_state()[1], atol=1e-7)
        env.reset(seed=8)
        assert not np.allclose(cube, env.task_state()[1])
        for _ in range(2):
            obs, reward, terminated, truncated, info = env.step(np.zeros(7))
        assert not terminated and truncated
        assert {'episode_reached', 'episode_grasped', 'episode_success'} <= info.keys()
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)


def test_fixed_and_adaptive_share_identical_components():
    collected = []
    for condition in ('fixed', 'adaptive'):
        with make_env(condition, horizon=1, seed=7) as env:
            env.reset(seed=7)
            _, reward, _, _, info = env.step(np.zeros(7))
            terms, weights = info['reward_terms'], info['reward_weights']
            expected = scaled_reward([terms[name] for name in TERM_NAMES],
                                     [weights[name] for name in TERM_NAMES], info['is_success'])
            assert reward == pytest.approx(expected)
            assert terms['lift'] == 0  # cube remains on the table
            collected.append(terms)
    assert collected[0] == pytest.approx(collected[1])


@pytest.mark.parametrize('condition', ['sparse', 'fixed', 'adaptive'])
def test_every_condition_has_same_success_reward_in_simulator(condition):
    with make_env(condition, horizon=2, seed=7) as env:
        env.reset(seed=7)
        _, reward, _, _, info = env.step(np.zeros(7))
        assert not info['is_success']
        assert 0 <= reward <= 0.5
        if condition == 'sparse':
            assert reward == 0
        # Set the physical cube free-joint above Lift's threshold. This tests the
        # actual robosuite success/reward code, even without gripper contact.
        base = env.base_env
        qpos = base.sim.data.get_joint_qpos(base.cube.joints[0]).copy()
        qpos[2] = env.task_geometry()[1] + 0.12
        base.sim.data.set_joint_qpos(base.cube.joints[0], qpos)
        base.sim.forward()
        _, reward, _, _, info = env.step(np.zeros(7))
        assert info['is_success']
        assert reward == 1.0


def test_actions_are_clipped_and_nonfinite_actions_rejected():
    with make_env('sparse', horizon=2, seed=7) as env:
        env.reset(seed=7)
        clipped, *_ = env.step(np.ones(7) * 20)
        env.reset(seed=7)
        bounded, *_ = env.step(np.ones(7))
        np.testing.assert_allclose(clipped, bounded, atol=1e-7)
        with pytest.raises(ValueError):
            env.step(np.full(7, np.nan))
