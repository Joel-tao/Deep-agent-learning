import pytest
from stable_baselines3.common.env_checker import check_env
from gymnasium.utils.env_checker import check_env as check_gymnasium_env
from envs.factory import make_env


@pytest.mark.parametrize('condition', ['sparse', 'fixed', 'adaptive'])
def test_environment_passes_sb3_checker(condition):
    with make_env(condition, horizon=5, seed=7) as env:
        check_env(env, warn=True)


@pytest.mark.parametrize('condition', ['sparse', 'fixed', 'adaptive'])
def test_environment_passes_gymnasium_checker(condition):
    with make_env(condition, horizon=5, seed=7) as env:
        check_gymnasium_env(env, skip_render_check=True)
