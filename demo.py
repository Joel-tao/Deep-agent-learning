"""Run a real MuJoCo episode and save a simplified physical-state animation.

No trained model or OpenGL context is needed for the default visualization.
The scripted policy is a hand-written controller for demonstration only.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/robot_grasp_matplotlib')

import numpy as np
from envs.factory import make_env
from envs.observation_normalization import ObservationNormalization
from rewards.scaling import REWARD_SCALE_VERSION


def scripted_action(env, step):
    """Return 7D actions from a four-stage hand-written demonstration controller.

    The first six values are OSC_POSE translation/rotation commands; the last
    controls the gripper (-1 opens, +1 closes). Rotation commands remain zero.
    """
    eef, cube, *_ = env.task_state()
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    target = cube.copy()
    if step < 60:
        # Align above the cube with an open gripper to reduce lateral collisions.
        target[2] += 0.10
        action[-1] = -1.0
    elif step < 120:
        # Descend to grasp height while keeping the gripper open.
        target[2] += 0.002
        action[-1] = -1.0
    elif step < 170:
        # Hold position and close the gripper, allowing time for bilateral contact.
        target[2] += 0.002
        action[-1] = 1.0
    else:
        # Target a cube center 0.12 m above the success height, as an absolute position.
        target[2] = env.task_geometry()[1] + 0.12
        action[-1] = 1.0
    # OSC input is scaled by 0.05 m; keep rotational increments at zero.
    action[:3] = np.clip((target - eef) / 0.05, -1, 1)
    return action


def save_visualization(frames, output, *, stride, hz, title):
    """Render XY/XZ projections and EEF trails from recorded simulator states.

    Robot geometry is omitted. stride is the recording interval and hz is the
    control frequency; both determine the GIF playback rate.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    figure, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    eefs = np.array([f['eef'] for f in frames])
    cubes = np.array([f['cube'] for f in frames])
    artists = []
    for ax, dimensions, name in zip(axes, ((0, 1), (0, 2)), ('Top view (XY)', 'Side view (XZ)')):
        x, y = dimensions
        points = np.concatenate([eefs[:, [x, y]], cubes[:, [x, y]]])
        lo, hi = points.min(axis=0) - 0.08, points.max(axis=0) + 0.08
        ax.set(xlim=(lo[0], hi[0]), ylim=(lo[1], hi[1]), xlabel='X (m)',
               ylabel='Y (m)' if y == 1 else 'Z (m)', title=name)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=0.25)
        if y == 2:
            ax.axhline(frames[0]['table'], color='#888888', label='Table')
            ax.axhline(frames[0]['success_height'], color='#228833', linestyle='--', label='Lift threshold')
        trail, = ax.plot([], [], color='#4477aa', alpha=0.35)
        eef, = ax.plot([], [], 'o', color='#4477aa', markersize=9, label='End effector')
        cube, = ax.plot([], [], 's', color='#ee7733', markersize=12, label='Cube center')
        ax.legend(loc='best', fontsize=7)
        artists.append((dimensions, trail, eef, cube))
    status = figure.suptitle('')
    figure.text(0.5, 0.02, 'Simplified visualization of real MuJoCo states; robot geometry omitted.',
                ha='center', fontsize=8)
    figure.subplots_adjust(top=0.78, bottom=0.18, wspace=0.35)

    def update(index):
        frame = frames[index]
        for (x, y), trail, eef, cube in artists:
            trail.set_data(eefs[:index + 1, x], eefs[:index + 1, y])
            eef.set_data([eefs[index, x]], [eefs[index, y]])
            cube.set_data([cubes[index, x]], [cubes[index, y]])
        status.set_text(f'{title} | step {frame["step"]} | reward {frame["reward"]:.3f}\n'
                        f'Reach {int(frame["reached"])}  Grasp {int(frame["grasped"])}  '
                        f'Lift {int(frame["success"])} | weights '
                        f'{tuple(round(w, 3) for w in frame["weights"].values())}')
        return [status]

    # Replay recorded states without advancing physics or synthesizing object motion.
    animation = FuncAnimation(figure, update, frames=len(frames), interval=1000 * stride / hz, blit=False)
    animation.save(output / 'episode.gif', writer=PillowWriter(fps=hz / stride))
    update(len(frames) - 1)
    figure.savefig(output / 'episode.png', dpi=150)
    plt.close(figure)


def main():
    """Parse options, run one episode, and save step data plus a simplified visualization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', choices=['random', 'scripted', 'model'], default='scripted')
    parser.add_argument('--condition', choices=['sparse', 'fixed', 'adaptive'], default='adaptive')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--frame-stride', type=int, default=10)
    parser.add_argument('--output', type=Path, default=Path('demo_output'))
    parser.add_argument('--viewer', action='store_true', help='also open native viewer; needs display/OpenGL')
    args = parser.parse_args()
    if args.horizon < 1 or args.frame_stride < 1:
        parser.error('horizon and frame-stride must be positive')
    if args.policy == 'model' and args.model is None:
        parser.error('--policy model requires --model path/to/policy.zip')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f'{args.output} is not empty; use a fresh --output directory')
    # The default scripted controller needs no checkpoint; load weights only in model mode.
    model = None
    if args.policy == 'model':
        from stable_baselines3 import SAC
        model = SAC.load(args.model, device='cpu')
    args.output.mkdir(parents=True, exist_ok=True)
    env = make_env(args.condition, horizon=args.horizon, seed=args.seed,
                   render_mode='human' if args.viewer else None,
                   grasp_hold_steps=getattr(model, 'grasp_hold_steps', 1))
    # Reuse and freeze the checkpoint's input statistics when normalization was enabled.
    normalizer = (ObservationNormalization(env, stats=model.observation_stats, update=False)
                  if model is not None and hasattr(model, 'observation_stats') else None)
    frames, records = [], []
    total_reward = 0.0
    try:
        obs, _ = env.reset(seed=args.seed)
        for step in range(args.horizon):
            if model is not None:
                policy_obs = normalizer.observation(obs) if normalizer is not None else obs
                action, _ = model.predict(policy_obs, deterministic=True)
            elif args.policy == 'random':
                action = env.action_space.sample()
            else:
                action = scripted_action(env, step)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            eef, cube, _height, table, _grasped, _lifted = env.task_state()
            row = dict(step=step + 1, reward=reward, reached=info['reached'],
                       grasped=info['grasped'], success=info['is_success'],
                       eef_x=eef[0], eef_y=eef[1], eef_z=eef[2],
                       cube_x=cube[0], cube_y=cube[1], cube_z=cube[2])
            records.append(row)
            # Log every step to CSV; subsample GIF frames without changing simulation frequency.
            if step % args.frame_stride == 0 or terminated or truncated:
                frames.append(dict(step=step + 1, eef=eef, cube=cube, table=table,
                                   success_height=env.task_geometry()[1], reward=reward,
                                   reached=info['reached'], grasped=info['grasped'],
                                   success=info['is_success'], weights=info.get('reward_weights', {})))
            if args.viewer:
                env.render()
                time.sleep(1 / 20)
            if terminated or truncated:
                break
        # Summarize skill occurrences across the episode, rather than its final frame alone.
        summary = dict(policy=args.policy, condition=args.condition, seed=args.seed,
                       reward_scale_version=REWARD_SCALE_VERSION,
                       steps=len(records), observation_dim=int(obs.size),
                       action_dim=int(env.action_space.shape[0]), episode_return=total_reward,
                       reached=env.episode_reached, grasped=env.episode_grasped,
                       success=env.episode_lifted, visualization='simplified MuJoCo state projections')
    finally:
        # Release simulator resources even if policy execution or data collection fails.
        env.close()
    with (args.output / 'trajectory.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    save_visualization(frames, args.output, stride=args.frame_stride, hz=20,
                       title=f'{args.policy} / {args.condition} / seed {args.seed}')
    print(json.dumps(summary, indent=2))
    print(f'Saved GIF, PNG, trajectory and summary to {args.output}')


if __name__ == '__main__':
    main()
