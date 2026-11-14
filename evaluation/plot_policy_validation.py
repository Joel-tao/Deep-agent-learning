"""Plot measured policy validation, clearly separate from RL learning curves."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/robot_grasp_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    """Compare measured policy validation outcomes separately from training learning curves."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, default=Path('artifacts/diagnostics/policy_validation.png'))
    args = parser.parse_args()
    data = [json.loads(path.read_text()) for path in args.inputs]
    # Compare only matching horizons, hold criteria, and reset seeds.
    protocols = {(item['horizon'], item['hold_steps'], tuple(e['seed'] for e in item['episodes'])) for item in data}
    if len(protocols) != 1:
        parser.error('Use matching horizons, hold criteria, and evaluation seeds')
    figure, axis = plt.subplots(figsize=(9, 5))
    x = np.arange(len(data))
    width = 0.19
    for index, (key, label) in enumerate([('reach_rate', 'Reach'), ('grasp_rate', 'Grasp'),
                                         ('success_rate', 'Lift (task)'), ('stable_lift_rate', 'Held Lift')]):
        values = [item[key] for item in data]
        bars = axis.bar(x + (index - 1.5) * width, values, width, label=label)
        axis.bar_label(bars, labels=[f'{value:.0%}' for value in values], fontsize=8, padding=3)
    axis.set_xticks(x, [f"{item['label']}\n(n={len(item['episodes'])})" for item in data])
    axis.set(ylabel='Episode rate', ylim=(0, 1.16), title='Measured policy validation - not a learning curve')
    axis.legend(loc='upper center', ncol=4)
    axis.grid(axis='y', alpha=0.2)
    horizon, hold_steps, _seeds = protocols.pop()
    figure.text(0.5, 0.015, f'{horizon}-step episodes; held Lift requires grasp + lift for {hold_steps} consecutive steps.',
                ha='center', fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(args.output, dpi=160)
    plt.close(figure)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
