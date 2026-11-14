"""Compare real simulator trajectories; labels describe observations, not causes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def main():
    root=Path('diagnostics/failure_analysis'); out=root/'comparison';out.mkdir(exist_ok=True)
    names=['scripted','fixed_200k','adaptive_200k','fixed_370k','adaptive_380k']
    data={name:pd.read_csv(root/name/'trace.csv') for name in names}
    rows=[]
    for name,df in data.items():
        for ep,g in df.groupby('episode'):
            near=g.distance<.06
            horizontal=np.hypot(g.eef_x-g.cube_x,g.eef_y-g.cube_y)
            contact=g.contact.astype(bool)
            # Panda's opposing slider coordinates have opposite signs.
            opening=(g.finger_left_qpos-g.finger_right_qpos).abs()
            near_g=g[near]
            label=('held_lift' if g.held_lift_run.max()>=10 else
                   'contact_without_held_lift' if contact.any() else
                   'near_without_bilateral_contact' if near.any() else 'never_reached')
            rows.append(dict(policy=name,seed=30003+int(ep),label=label,
                minimum_distance_m=g.distance.min(),minimum_horizontal_distance_m=horizontal.min(),
                near_steps=int(near.sum()),contact_steps=int(contact.sum()),max_contact_steps=int(g.contact_run.max()),
                left_contact_steps=int(g.left_contact.sum()),right_contact_steps=int(g.right_contact.sum()),
                near_close_command_fraction=float((near_g.gripper_action>0).mean()) if len(near_g) else None,
                near_mean_finger_opening_m=float(opening[near].mean()) if near.any() else None,
                closed_command_far_fraction=float(((g.gripper_action>0)&~near).mean()),
                rotation_command_saturation_fraction=float((g[['rotation_x','rotation_y','rotation_z']].abs()>.95).any(axis=1).mean()),
                native_lift=bool(g.native_lift.any()),held_lift=bool(g.held_lift_run.max()>=10)))
    stats=pd.DataFrame(rows);stats.to_csv(out/'episode_findings.csv',index=False)
    fig,axes=plt.subplots(5,5,figsize=(19,13),sharex='col',constrained_layout=True)
    for j,name in enumerate(names):
        g=data[name];g=g[g.episode==0];t=g.step/20
        axes[0,j].plot(t,g.distance);axes[0,j].axhline(.06,color='r',ls='--');axes[0,j].set_title(name)
        axes[1,j].plot(t,g.eef_z,label='EEF');axes[1,j].plot(t,g.cube_z,label='cube');axes[1,j].axhline(.84,color='r',ls='--')
        axes[2,j].plot(t,g.gripper_action,label='close command');axes[2,j].plot(t,(g.finger_left_qpos-g.finger_right_qpos).abs()/.08,label='opening / .08m')
        axes[3,j].plot(t,g.left_contact,label='left');axes[3,j].plot(t,g.right_contact,ls='--',label='right');axes[3,j].plot(t,g.contact,lw=2,label='bilateral')
        axes[4,j].plot(t,g.rotation_x,label='rx');axes[4,j].plot(t,g.rotation_y,label='ry');axes[4,j].plot(t,g.rotation_z,label='rz');axes[4,j].set_xlabel('Seconds')
        for i in range(5):axes[i,j].grid(alpha=.2)
    for ax,label in zip(axes[:,0],['Distance (m)','Height (m)','Gripper','Contact','Rotation command']):ax.set_ylabel(label)
    for i in range(1,5):axes[i,0].legend(fontsize=7,loc='best')
    fig.suptitle('Same scene seed 30003 | actual simulated states | scripted is a feasibility control',fontsize=14)
    fig.savefig(out/'signals.png',dpi=160);plt.close(fig)
    import sys
    if '--no-animation' in sys.argv:
        print(stats.to_string(index=False))
        return
    fig,axes=plt.subplots(1,5,figsize=(18,4),constrained_layout=True)
    artists=[]; sampled={n:data[n][data[n].episode==0].iloc[::5].reset_index(drop=True) for n in names}
    for ax,name in zip(axes,names):
        g=sampled[name]; ax.set_title(name);ax.set_xlim(-.3,.3);ax.set_ylim(.7,1.25)
        ax.set_xlabel('X (m)');ax.set_ylabel('Z (m)');ax.axhline(.8,color='gray');ax.axhline(.84,color='red',ls='--')
        trail,=ax.plot([],[],alpha=.5);eef,=ax.plot([],[],'o',label='EEF');cube,=ax.plot([],[],'s',label='cube');ax.legend(fontsize=7)
        artists.append((trail,eef,cube))
    title=fig.suptitle('XZ projection only; arm geometry and Y are omitted')
    def update(k):
        for n,(trail,eef,cube) in zip(names,artists):
            g=sampled[n]; r=g.iloc[k];trail.set_data(g.eef_x[:k+1],g.eef_z[:k+1]);eef.set_data([r.eef_x],[r.eef_z]);cube.set_data([r.cube_x],[r.cube_z])
        title.set_text(f'Actual state XZ projection | seed 30003 | t={k*.25:.2f}s | geometry and Y omitted')
    FuncAnimation(fig,update,frames=min(map(len,sampled.values())),interval=100).save(out/'comparison.gif',writer=PillowWriter(fps=10))
    plt.close(fig)
    print(stats.to_string(index=False))


if __name__=='__main__':main()
