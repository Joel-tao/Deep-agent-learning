"""Regression tests for curriculum confirmation, physical height and replay."""
import numpy as np
import pytest
from gymnasium.spaces import Box
from itertools import product
from rewards.adaptive_reward import AdaptiveRewardScheduler
from rewards.components import lift_reward
from training.replay_buffer import ComponentReplayBuffer
from rewards.adaptive_reward_wrapper import FIXED_WEIGHTS
from rewards.scaling import SHAPING_BUDGET, TERM_NAMES, normalize_weights, scaled_reward


@pytest.mark.parametrize('weights', [FIXED_WEIGHTS, *AdaptiveRewardScheduler.WEIGHTS])
def test_reward_scale_is_constant_across_conditions_and_stages(weights):
    vector = [weights[name] for name in TERM_NAMES]
    assert sum(vector) == pytest.approx(1.0)
    corners = np.array(list(product([0.0, 1.0], repeat=3)))
    non_success = scaled_reward(corners, vector, False)
    assert np.all((0 <= non_success) & (non_success <= SHAPING_BUDGET + 1e-12))
    assert scaled_reward(np.ones(3), vector, False) == pytest.approx(0.5)
    np.testing.assert_array_equal(scaled_reward(corners, vector, True), np.ones(8))


def test_weight_normalization_is_invariant_to_global_multiplier():
    ratios = dict(reach=1.0, grasp=0.2, lift=0.1)
    assert normalize_weights(ratios) == pytest.approx(
        normalize_weights({name: 10 * value for name, value in ratios.items()}))


def test_scheduler_requires_window_and_consecutive_updates_and_never_regresses():
    scheduler = AdaptiveRewardScheduler(window_size=2, update_interval=1, confirmations=2)
    scheduler.update(True, False, False)
    assert scheduler.stage == 0  # no full window yet
    scheduler.update(True, False, False)
    assert scheduler.stage == 0  # first confirmation
    scheduler.update(False, False, False)
    assert scheduler.stage == 0  # failure cancels confirmation
    for _ in range(3):
        scheduler.update(True, False, False)
    assert scheduler.stage == 1
    for _ in range(3):
        scheduler.update(True, True, False)
    assert scheduler.stage == 2
    for _ in range(5):
        scheduler.update(False, False, False)
    assert scheduler.stage == 2


def test_scheduler_updates_only_at_episode_interval():
    scheduler = AdaptiveRewardScheduler(window_size=1, update_interval=3, confirmations=1)
    for _ in range(2):
        scheduler.update(True, False, False)
    assert scheduler.stage == 0
    scheduler.update(True, False, False)
    assert scheduler.stage == 1


def test_lift_progress_uses_resting_center_and_clips():
    assert lift_reward(0.82, 0.82, 0.02) == 0
    assert lift_reward(0.83, 0.82, 0.02) == pytest.approx(0.5)
    assert lift_reward(0.90, 0.82, 0.02) == 1
    assert lift_reward(0.80, 0.82, 0.02) == 0


def test_replay_relabels_old_samples_preserves_scale_and_timeout_mask():
    scheduler = AdaptiveRewardScheduler(window_size=1, update_interval=1, confirmations=1)
    buffer = ComponentReplayBuffer(3, Box(-1, 1, (2,)), Box(-1, 1, (1,)),
                                   device='cpu', scheduler=scheduler)
    obs = np.zeros((1, 2), dtype=np.float32)
    action = np.zeros((1, 1), dtype=np.float32)
    # Fill beyond capacity to verify components stay aligned after ring wrap.
    for value in range(4):
        buffer.add(obs + value, obs + value + 1, action, np.array([999.0]), np.array([True]),
                   [dict(reward_terms=dict(reach=0.2 * value, grasp=1.0, lift=0.5),
                         is_success=False, **{'TimeLimit.truncated': True})])
    sample = buffer._get_samples(np.array([0]))
    assert sample.observations[0, 0].item() == 3
    assert sample.rewards.item() == pytest.approx(0.5 * (0.6 + 0.2 + 0.05) / 1.3)
    assert sample.dones.item() == 0  # timeouts bootstrap
    scheduler.update(True, False, False)
    sample = buffer._get_samples(np.array([0]))
    assert sample.rewards.item() == pytest.approx(0.5 * (0.3 * 0.6 + 1 + 0.3 * 0.5) / 1.6)
    # Success must override shaping in replay just as it does online.
    index = buffer.pos
    buffer.add(obs, obs, action, np.array([999.0]), np.array([False]),
               [dict(reward_terms=dict(reach=0.3, grasp=0.0, lift=0.0), is_success=True)])
    assert buffer._get_samples(np.array([index])).rewards.item() == 1.0
    scheduler.update(True, True, True)
    assert buffer._get_samples(np.array([index])).rewards.item() == 1.0
