"""Plot real checkpoint skill rates; explicitly reject unlabelled smoke-test curves."""
import argparse
import csv
import glob
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/robot_grasp_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    """Group recorded checkpoints by reward condition and plot the three skill curves."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pattern', default='runs/*/seed_*/evaluations.csv')
    parser.add_argument('--output', type=Path, default=Path('results/success_curves.png'))
    parser.add_argument('--target', type=float, default=0.90)
    parser.add_argument('--smoke-test', action='store_true', help='allow clearly labelled pipeline diagnostics')
    args = parser.parse_args()
    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise SystemExit(f'No evaluation CSV files match: {args.pattern}')
    groups, efficiency, short_runs = {}, [], []
    for path in paths:
        path = Path(path)
        config_path = path.parent / 'config.json'
        config = {}
        if config_path.exists():
            config = json.loads(config_path.read_text())
            if config.get('run_kind') == 'pipeline_check' or config.get('horizon', 500) < 100 or config.get('timesteps', 0) < 1000:
                short_runs.append(str(path))
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        # Read identity from the saved run, so copied submission snapshots do
        # not accidentally merge different conditions into one multi-seed mean.
        condition = config.get('condition', path.parent.parent.name)
        seed = config.get('seed', path.parent.name)
        groups.setdefault(condition, []).append(frame.set_index('timesteps'))
        crossed = frame.loc[frame.success_rate >= args.target, 'timesteps']
        efficiency.append(dict(condition=condition, seed=seed, target=args.target,
                               first_checkpoint=int(crossed.iloc[0]) if len(crossed) else 'not_reached'))
    if short_runs and not args.smoke_test:
        raise SystemExit('These are short pipeline checks, not learning-performance runs. '
                         'Use --smoke-test and an output such as pipeline_check.png, or train with horizon=500.')
    if not groups:
        raise SystemExit('Evaluation files contain no checkpoints')
    frames = [frame for runs in groups.values() for frame in runs]
    metrics = [('reach_rate', 'Reach rate'), ('grasp_rate', 'Grasp rate'), ('success_rate', 'Lift success rate')]
    metrics = [(key, label) for key, label in metrics if all(key in frame for frame in frames)]
    figure, axes = plt.subplots(len(metrics), 1, figsize=(8, 3 * len(metrics)), sharex=True, squeeze=False)
    for axis, (metric, label) in zip(axes[:, 0], metrics):
        for condition, runs in groups.items():
            # Compare shared checkpoints; only multiple training seeds support a variation band.
            frame = pd.concat([run[metric] for run in runs], axis=1, join='inner').sort_index()
            if frame.empty:
                raise SystemExit(f'No shared checkpoints for {condition}')
            mean, std = frame.mean(axis=1), frame.std(axis=1, ddof=0)
            axis.plot(frame.index, mean, marker='o', label=f'{condition} (n={len(runs)})')
            if len(runs) > 1:
                axis.fill_between(frame.index, np.maximum(0, mean - std), np.minimum(1, mean + std), alpha=0.15)
        if metric == 'success_rate':
            axis.axhline(args.target, linestyle='--', color='gray', label=f'Target {args.target:.0%}')
        axis.set(ylabel=label, ylim=(-0.02, 1.02))
        axis.grid(alpha=0.25)
        axis.legend()
    axes[-1, 0].set_xlabel('Training environment steps')
    title = 'Pipeline check only - not learning performance' if args.smoke_test else 'Held-out skills during SAC training'
    figure.suptitle(title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    # Summarize the first target-crossing checkpoint, or explicitly record not_reached.
    with args.output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['condition', 'seed', 'target', 'first_checkpoint'])
        writer.writeheader()
        writer.writerows(efficiency)
    print(f'Saved {args.output}; single-seed curves show no variability band')


if __name__ == '__main__':
    main()
