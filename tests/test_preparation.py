"""Reward ordering and runtime consistency for the candidate shaping profile."""
import numpy as np
import pytest
from rewards.preparation import preparation_terms
from envs.factory import make_env
from rewards.scaling import scaled_reward,TERM_NAMES


def test_approach_and_closure_priorities_and_bounds():
    cube=np.array([0.,0.,.82]);ready=cube+np.array([0,0,.002]);far=ready+np.array([.1,0,0])
    def terms(pos,gap,cmd,contact=False,hold=0):
        return preparation_terms(pos,cube,gap,cmd,contact,hold,.8)[0]
    assert terms(far,.08,-1)['reach']>terms(far,0,1)['reach']
    assert terms(ready,.08,1)['grasp']>terms(far,.08,1)['grasp']
    assert terms(ready,.08,-1)['grasp']==0
    assert terms(ready,.04,1,True,.2)['grasp']>terms(ready,0,1)['grasp']
    for pos in [ready,far]:
        for contact in [True,False]:
            for progress in [0,.2,1]:
                assert all(0<=v<=1 for v in terms(pos,.04,1,contact,progress).values())


@pytest.mark.parametrize('condition',['fixed','adaptive'])
def test_profile_preserves_spaces_and_online_replay_component_contract(condition):
    with make_env(condition,horizon=3,seed=7,grasp_hold_steps=5,reward_profile='prepare_v1') as env:
        obs,_=env.reset(seed=7)
        assert obs.shape==(61,)
        for _ in range(3):
            obs,r,terminated,truncated,info=env.step(np.zeros(7))
            assert not terminated
            assert 'preparation' in info
            terms=info['reward_terms'];weights=info['reward_weights']
            expected=scaled_reward([terms[k] for k in TERM_NAMES],[weights[k] for k in TERM_NAMES],info['is_success'])
            assert r==pytest.approx(float(expected))
            assert 0<=r<=1
