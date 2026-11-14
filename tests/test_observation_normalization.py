"""Check evaluation isolation and checkpoint portability of policy input stats."""
import numpy as np
from stable_baselines3 import SAC

from envs.factory import make_env
from envs.observation_normalization import ObservationNormalization


def test_evaluation_freezes_statistics_and_preserves_task_reward():
    with ObservationNormalization(make_env('fixed', horizon=2, seed=7)) as train:
        train.reset(seed=7)
        train.step(np.zeros(7))
        count = train.stats.count
        mean, variance = train.stats.mean.copy(), train.stats.var.copy()
        with ObservationNormalization(make_env('fixed', horizon=2, seed=7),
                                      stats=train.stats, update=False) as evaluation:
            evaluation.reset(seed=7)
            normalized, eval_reward, *_ = evaluation.step(np.zeros(7))
        with make_env('fixed', horizon=2, seed=7) as reference:
            reference.reset(seed=7)
            _, raw_reward, *_ = reference.step(np.zeros(7))
        assert eval_reward == raw_reward
        assert train.stats.count == count
        np.testing.assert_array_equal(train.stats.mean, mean)
        np.testing.assert_array_equal(train.stats.var, variance)
        assert normalized.dtype == np.float32
        assert np.max(np.abs(normalized)) <= 10


def test_checkpoint_restores_matching_statistics_and_predictions(tmp_path):
    with ObservationNormalization(make_env('fixed', horizon=2, seed=7)) as env:
        env.reset(seed=7)
        env.step(np.zeros(7))
        model = SAC('MlpPolicy', env, buffer_size=8, policy_kwargs={'net_arch': [16, 16]},
                    seed=7, device='cpu')
        model.observation_stats = env.stats
        checkpoint = tmp_path / 'normalized_policy.zip'
        model.save(checkpoint)
        restored = SAC.load(checkpoint, device='cpu')
        with make_env('fixed', horizon=2, seed=7) as task:
            physical_observation, _ = task.reset(seed=7)
            before = ObservationNormalization(task, stats=env.stats, update=False)
            after = ObservationNormalization(task, stats=restored.observation_stats, update=False)
            before_obs = before.observation(physical_observation)
            after_obs = after.observation(physical_observation)
            np.testing.assert_array_equal(before_obs, after_obs)
            np.testing.assert_array_equal(model.predict(before_obs, deterministic=True)[0],
                                          restored.predict(after_obs, deterministic=True)[0])
