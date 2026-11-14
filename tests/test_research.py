"""Behavior tests for measured skill duration and episode-boundary smoothing."""
import pickle
import numpy as np
import pytest
from envs.skill_diagnostics import SkillDiagnostics
from rewards.adaptive_reward import AdaptiveRewardScheduler
from envs.factory import make_env
from training.replay_buffer import ComponentReplayBuffer
from gymnasium.spaces import Box


def test_contact_duration_resets_on_loss_and_bounce_is_not_held_lift():
    d=SkillDiagnostics(contact_steps=3,lift_steps=2)
    for contact,lift in [(True,False),(True,False),(False,True),(True,False)]:
        d.update(contact,lift,.01,np.zeros(7))
    assert d.summary()['max_contact_steps']==2
    assert not d.summary()['stable_grasp']
    assert not d.summary()['grasped_lift_after_contact']
    for _ in range(2):d.update(True,True,.03,np.array([0,0,.5,0,0,0,1]))
    assert d.summary()['stable_grasp'] and d.summary()['stable_lift']
    d.reset()
    assert d.summary()['first_contact_step'] is None


def test_smoothing_and_checkpoint_continue_same_path():
    s=AdaptiveRewardScheduler(window_size=1,update_interval=1,confirmations=1,transition_episodes=4)
    initial=s.get_weights()
    s.update(True,False,False)
    assert s.stage==1
    target=s.stage_weights[1]
    assert s.get_weights()==pytest.approx({k:.75*initial[k]+.25*target[k] for k in target})
    restored=pickle.loads(pickle.dumps(s))
    for _ in range(3):
        s.update(True,True,True);restored.update(True,True,True)
        assert restored.get_weights()==s.get_weights()
        assert sum(s.get_weights().values())==pytest.approx(1)
        assert s.stage==1  # no second switch during interpolation
    assert s.get_weights()==pytest.approx(target)


def test_replay_uses_interpolated_weights_and_timeout_bootstraps():
    s=AdaptiveRewardScheduler(window_size=1,update_interval=1,confirmations=1,transition_episodes=4)
    b=ComponentReplayBuffer(2,Box(-1,1,(2,)),Box(-1,1,(1,)),device='cpu',scheduler=s)
    b.add(np.zeros((1,2)),np.ones((1,2)),np.zeros((1,1)),np.array([0]),np.array([True]),
          [dict(reward_terms=dict(reach=0,grasp=1,lift=0),is_success=False,**{'TimeLimit.truncated':True})])
    s.update(True,False,False)
    sample=b._get_samples(np.array([0]))
    assert sample.rewards.item()==pytest.approx(.5*s.get_weights()['grasp'])
    assert sample.dones.item()==0


def test_weights_stay_fixed_inside_episode_and_diagnostics_do_not_terminate():
    with make_env('adaptive',horizon=3,seed=7,scheduler_config=dict(window_size=1,update_interval=1,confirmations=1,transition_episodes=3)) as env:
        env.reset(seed=7)
        weights=env.scheduler.get_weights()
        for step in range(3):
            _,_,terminated,truncated,info=env.step(np.zeros(7))
            assert info['reward_weights']==weights
            assert not terminated
            assert truncated==(step==2)
        assert 'max_contact_steps' in info['diagnostics']


def test_old_scheduler_state_migrates_at_its_stage():
    old=AdaptiveRewardScheduler().__dict__.copy()
    old['stage']=1
    for k in ['stage_weights','current_weights','transition_episodes','transition_remaining']:old.pop(k)
    restored=AdaptiveRewardScheduler.__new__(AdaptiveRewardScheduler)
    restored.__setstate__(old)
    assert restored.get_weights()==restored.WEIGHTS[1]
    restored.update(True,True,False)


def test_hold_reward_ramps_resets_and_observation_exposes_memory():
    from rewards.components import sustained_grasp_reward
    assert sustained_grasp_reward(1,5)==pytest.approx(.2)
    assert sustained_grasp_reward(5,5)==1
    assert sustained_grasp_reward(20,5)==1
    assert sustained_grasp_reward(0,5)==0
    for condition in ['fixed','adaptive','sparse']:
        with make_env(condition,horizon=2,seed=7,grasp_hold_steps=5) as env:
            obs,_=env.reset(seed=7)
            assert obs.shape==(61,) and obs[-1]==0
            env.env.diagnostics.contact_run=3
            assert env.observation(obs[:-1])[-1]==pytest.approx(.6)
            obs,_=env.reset(seed=7)
            assert obs[-1]==0
