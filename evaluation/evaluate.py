"""Evaluate learned, scripted, or random policies under a common 500-step task."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from stable_baselines3 import SAC
from envs.factory import make_env
from envs.observation_normalization import ObservationNormalization
from demo import scripted_action


def main():
    """Evaluate on held-out seeds, reporting instantaneous and sustained grasped Lift."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', nargs='?')
    parser.add_argument('--policy', choices=['model', 'scripted', 'random'], default='model')
    parser.add_argument('--label', help='explicit label for the saved policy or controller')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20_000)
    parser.add_argument('--horizon', type=int, default=500)
    parser.add_argument('--hold-steps', type=int, default=10)
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.episodes, args.horizon, args.hold_steps, args.torch_threads) < 1:
        parser.error('Episode count, horizon, hold steps, and thread count must be positive')
    if args.policy == 'model' and not args.model:
        parser.error('A model path is required for --policy model')
    if args.policy != 'model' and args.model:
        parser.error('A model path applies only to --policy model')
    torch.set_num_threads(args.torch_threads)
    model = SAC.load(args.model, device='cpu') if args.policy == 'model' else None
    # Evaluate every policy on the same sparse task without updating its training curriculum.
    hold_steps = getattr(model, 'grasp_hold_steps', 1)
    env = make_env('sparse', horizon=args.horizon, seed=args.seed, grasp_hold_steps=hold_steps)
    task = env.env if hold_steps > 1 else env
    if model is not None and hasattr(model, 'observation_stats'):
        env = ObservationNormalization(env, stats=model.observation_stats, update=False)
    episodes = []
    try:
        for index in range(args.episodes):
            # Use distinct reproducible seeds; reset also seeds the action sampler.
            obs, _ = env.reset(seed=args.seed + index)
            total, success, stable = 0.0, False, False
            held_steps = 0
            for step in range(args.horizon):
                if model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                elif args.policy == 'scripted':
                    action = scripted_action(env, step)
                else:
                    action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                success |= info['is_success']
                # Reset on lost contact or height; held Lift requires consecutive valid steps.
                held_steps = held_steps + 1 if info['grasped'] and info['is_success'] else 0
                stable |= held_steps >= args.hold_steps
                if terminated or truncated:
                    break
            episodes.append(dict(seed=args.seed + index, reached=task.episode_reached,
                                 grasped=task.episode_grasped, success=bool(success),
                                 stable_lift_success=bool(stable), sparse_return=total,
                                 steps=step + 1))
    finally:
        env.close()
    # Average rates across episodes and retain each episode's outcome for inspection.
    result = dict(model=args.model, policy=args.policy, label=args.label or args.policy,
                  training_timesteps=int(model.num_timesteps) if model is not None else 0,
                  horizon=args.horizon, hold_steps=args.hold_steps, episodes=episodes,
                  success_rate=float(np.mean([e['success'] for e in episodes])),
                  reach_rate=float(np.mean([e['reached'] for e in episodes])),
                  grasp_rate=float(np.mean([e['grasped'] for e in episodes])),
                  stable_lift_rate=float(np.mean([e['stable_lift_success'] for e in episodes])),
                  mean_sparse_return=float(np.mean([e['sparse_return'] for e in episodes])))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"{result['label']}: {args.episodes} episodes, horizon {args.horizon}")
    print(f"Reach {result['reach_rate']:.1%}; Grasp {result['grasp_rate']:.1%}; "
          f"Lift {result['success_rate']:.1%}; held Lift {result['stable_lift_rate']:.1%}")
    print('Lift uses the unchanged robosuite criterion; held Lift is an additional diagnostic.')


if __name__ == '__main__':
    main()
