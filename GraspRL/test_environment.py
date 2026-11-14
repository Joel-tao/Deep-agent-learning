"""Integration tests exercise physics and episode semantics, not just imports."""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from stable_baselines3.common.env_checker import check_env as sb3_check_env
from adaptive_scheduler import AdaptiveRewardScheduler
from grasp_env import AdaptiveGraspEnv
from reward_functions import compute_reach_reward, compute_lift_reward


def test_reward_and_scheduler():
    close, _ = compute_reach_reward([0, 0, 0], [0, 0, 0.01])
    far, _ = compute_reach_reward([0, 0, 0], [0, 0, 1])
    assert 0 <= far < close <= 1
    assert compute_lift_reward(0.82, 0.82, False) == 0
    assert compute_lift_reward(0.90, 0.82, True) == 1
    s = AdaptiveRewardScheduler(window_size=4, update_interval=2, confirmations=2)
    for _ in range(4):
        s.record(True, False, False)
    assert s.stage == 0  # first confirmation only
    for _ in range(2):
        s.record(True, False, False)
    assert s.stage == 1
    for _ in range(6):
        s.record(True, True, True)
    assert s.stage == 2
    for _ in range(6):
        s.record(False, False, False)
    assert s.stage == 0  # demotion is possible after sustained failure


def test_gymnasium_and_sb3():
    env = AdaptiveGraspEnv(horizon=20)
    try:
        check_env(env, skip_render_check=True)
        sb3_check_env(env, warn=True)
    finally:
        env.close()


@pytest.mark.parametrize("mode", ["sparse", "fixed", "adaptive"])
def test_episode_and_rewards(mode):
    env = AdaptiveGraspEnv(reward_mode=mode, horizon=4)
    try:
        first, _ = env.reset(seed=7)
        second, _ = env.reset(seed=7)
        np.testing.assert_allclose(first, second, atol=1e-7)
        for i in range(4):
            obs, reward, term, trunc, info = env.step(np.zeros(env.action_space.shape))
            assert env.observation_space.contains(obs)
            expected = float(info["is_success"]) if mode == "sparse" else float(np.dot(info["reward_components"], info["weights"]))
            assert reward == pytest.approx(expected)
            assert not term
            assert trunc == (i == 3)
        assert env.scheduler.episodes == (1 if mode == "adaptive" else 0)
        with pytest.raises(RuntimeError):
            env.step(np.zeros(env.action_space.shape))
    finally:
        env.close()


def test_grasp_and_success_physics():
    env = AdaptiveGraspEnv()
    try:
        env.reset(seed=42)
        zero = np.zeros(env.action_space.shape)
        zero[-1] = 1  # empty closing motion must not count as grasp
        for _ in range(20):
            _, _, _, _, info = env.step(zero)
            assert not info["valid_grasp"]
        # Place the cube high above the table: success must match robosuite,
        # and must terminate rather than wait for the horizon.
        base = env.sim_env
        qpos = base.sim.data.get_joint_qpos(base.cube.joints[0]).copy()
        qpos[2] = base.table_offset[2] + 0.20
        base.sim.data.set_joint_qpos(base.cube.joints[0], qpos)
        base.sim.forward()
        _, reward, term, trunc, info = env.step(np.zeros(env.action_space.shape))
        assert info["is_success"] and term and not trunc
        assert info["reward_components"][2] == 1
    finally:
        env.close()


def test_same_geometry_across_instances():
    first = AdaptiveGraspEnv()
    second = AdaptiveGraspEnv()
    try:
        np.testing.assert_array_equal(first.sim_env.cube.size, second.sim_env.cube.size)
        a, _ = first.reset(seed=42)
        b, _ = second.reset(seed=42)
        np.testing.assert_allclose(a, b, atol=1e-7)
    finally:
        first.close()
        second.close()


def test_scripted_contact_and_lift():
    from demo import ScriptedPolicy
    env = AdaptiveGraspEnv(horizon=300)
    try:
        env.reset(seed=42)
        policy = ScriptedPolicy()
        for _ in range(300):
            _, _, term, trunc, info = env.step(policy.action(env))
            if term or trunc:
                break
        assert info["stage_ever"].all(), info
        assert term and info["is_success"]
    finally:
        env.close()
