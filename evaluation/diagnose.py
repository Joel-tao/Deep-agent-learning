"""Record held-contact diagnostics and traces without altering the evaluated policy.

Shaping totals are counterfactual scores under frozen checkpoint weights, not
the original training returns. Historical checkpoints may use another runtime.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import SAC
from envs.factory import make_env
from envs.observation_normalization import ObservationNormalization
from rewards.components import reach_reward, lift_reward
from rewards.adaptive_reward_wrapper import FIXED_WEIGHTS
from rewards.scaling import TERM_NAMES
from demo import scripted_action
from rewards.preparation import preparation_terms


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path)
    p.add_argument('--policy', choices=['model', 'scripted', 'random'], default='model')
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--horizon', type=int, default=500)
    p.add_argument('--seed', type=int, default=30000)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--zero-rotation', action='store_true', help='diagnostic action intervention, not original policy performance')
    a = p.parse_args()
    if min(a.episodes,a.horizon)<1 or (a.policy=='model' and not a.model):
        p.error('Positive budgets and a model path for model policy are required')
    if a.output.exists(): p.error('Choose a new output directory')
    a.output.mkdir(parents=True)
    torch.set_num_threads(1)
    model = SAC.load(a.model, device='cpu') if a.policy=='model' else None
    scheduler = getattr(model, 'reward_scheduler', None)
    weights = scheduler.get_weights() if scheduler else FIXED_WEIGHTS
    hold_steps=getattr(model,'grasp_hold_steps',1)
    raw = make_env('sparse', horizon=a.horizon, seed=a.seed, grasp_hold_steps=hold_steps)
    task = raw.env if hold_steps>1 else raw
    env = ObservationNormalization(raw, stats=model.observation_stats, update=False) if hasattr(model,'observation_stats') else raw
    episodes=[]
    fields=['episode','step','eef_x','eef_y','eef_z','cube_x','cube_y','cube_z','distance',
            'action_z','gripper_action','contact','native_lift','contact_run','held_lift_run',
            'reach_component','grasp_component','lift_component','sparse_reward']
    fields += ['action_x','action_y','rotation_x','rotation_y','rotation_z',
               'finger_left_qpos','finger_right_qpos','left_contact','right_contact',
               'eef_axis_x','eef_axis_y','eef_axis_z']
    fields += ['readiness','horizontal_error','height_error','opening_progress','premature_close']
    try:
        with (a.output/'trace.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
            for i in range(a.episodes):
                obs,_=env.reset(seed=a.seed+i); totals=np.zeros(3); success_steps=0
                for step in range(a.horizon):
                    if model is not None: action,_=model.predict(obs,deterministic=True)
                    elif a.policy=='scripted': action=scripted_action(task,step)
                    else: action=env.action_space.sample()
                    if a.zero_rotation:
                        action=action.copy(); action[3:6]=0
                    obs,reward,terminated,truncated,info=env.step(action)
                    eef,cube,z,_,contact,success=task.task_state()
                    resting,target=task.task_geometry()
                    terms=np.array([reach_reward(eef,cube),float(contact),lift_reward(z,resting,target-resting) if contact else 0.])
                    # Success overrides shaping in production, so do not double-count it.
                    success_steps+=int(success)
                    base=task.base_env; robot=base.robots[0]
                    gripper=next(iter(robot.gripper.values()))
                    joint=base.sim.data.qpos[next(iter(robot._ref_gripper_joint_pos_indexes.values()))]
                    axis=np.asarray(base.sim.data.site_xmat[next(iter(robot.eef_site_id.values()))]).reshape(3,3)[:,2]
                    extra=dict(zip(['action_x','action_y','rotation_x','rotation_y','rotation_z'],
                                   [float(action[k]) for k in [0,1,3,4,5]]))
                    extra.update(finger_left_qpos=float(joint[0]),finger_right_qpos=float(joint[1]),
                         left_contact=int(base.check_contact(gripper.important_geoms['left_finger'],base.cube.contact_geoms)),
                         right_contact=int(base.check_contact(gripper.important_geoms['right_finger'],base.cube.contact_geoms)),
                         **dict(zip(['eef_axis_x','eef_axis_y','eef_axis_z'],axis)))
                    prepared,ready=preparation_terms(eef,cube,abs(float(joint[0]-joint[1])),
                        float(action[-1]),contact,min(task.diagnostics.contact_run/hold_steps,1),terms[0])
                    extra.update(ready)
                    if getattr(model,'reward_profile','legacy')=='prepare_v1':
                        terms[0],terms[1]=prepared['reach'],prepared['grasp']
                    elif hold_steps>1:
                        terms[1]=min(task.diagnostics.contact_run/hold_steps,1)
                    if not success: totals += .5*terms*np.array([weights[k] for k in TERM_NAMES])
                    writer.writerow(dict(episode=i,step=step,**dict(zip(['eef_x','eef_y','eef_z'],eef)),
                        **dict(zip(['cube_x','cube_y','cube_z'],cube)),distance=float(np.linalg.norm(eef-cube)),
                        action_z=float(action[2]),gripper_action=float(action[-1]),contact=int(contact),
                        native_lift=int(success),contact_run=task.diagnostics.contact_run,
                        held_lift_run=task.diagnostics.held_run,reach_component=terms[0],
                        grasp_component=terms[1],lift_component=terms[2],sparse_reward=reward,**extra))
                    if terminated or truncated: break
                d=task.diagnostics.summary()
                episodes.append(dict(seed=a.seed+i,reached=task.episode_reached,grasped=task.episode_grasped,
                    native_lift=task.episode_lifted,**d,
                    frozen_weight_nonsuccess_contributions=dict(zip(TERM_NAMES,totals.tolist())),
                    success_reward_sum=success_steps))
    finally: env.close()
    contact=[e for e in episodes if e['grasped']]
    result=dict(policy=a.policy,reward_profile=getattr(model,'reward_profile','legacy'),zero_rotation_intervention=a.zero_rotation,model=str(a.model) if a.model else None,episodes=episodes,
        checkpoint_steps=int(model.num_timesteps) if model else None,frozen_weights=weights,
        rates={key:float(np.mean([e[key] for e in episodes])) for key in ['reached','grasped','stable_grasp','native_lift','stable_lift']},
        grasped_lift_given_contact=(sum(e['grasped_lift_after_contact'] for e in contact)/len(contact) if contact else None),
        caveats=['Small-sample diagnostic, not a performance comparison.',
                 'Historical Linux checkpoints are evaluated in the current Windows dependency environment.',
                 'Reward contributions are recomputed with frozen checkpoint weights; they are not historical training returns.'])
    (a.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['rates']))


if __name__=='__main__': main()
