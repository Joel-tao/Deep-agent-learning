"""Compare a single-seed candidate to its matched sustained-contact baseline."""
from pathlib import Path
import json
import csv
import pandas as pd


def main():
    root=Path('diagnostics');rows=[]
    for name in ['preparation_baseline_10k','preparation_candidate_10k']:
        p=root/name
        summary=json.loads((p/'summary.json').read_text(encoding='utf-8'))
        trace=pd.read_csv(p/'trace.csv')
        closing=trace.gripper_action>.1
        # CSV stores Boolean values as True/False strings or inferred booleans.
        premature=trace.premature_close.astype(str).str.lower().eq('true')
        metrics=dict(group=name,episodes=len(summary['episodes']),**summary['rates'],
            longest_contact_steps=max(e['max_contact_steps'] for e in summary['episodes']),
            premature_close_fraction_of_closing=float(premature[closing].mean()) if closing.any() else None,
            ready_episode_rate=float(trace.groupby('episode').readiness.max().ge(.5).mean()),
            median_minimum_distance_m=float(trace.groupby('episode').distance.min().median()))
        rows.append(metrics)
    out=Path('runs/preparation_pilot_10k/comparison');out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'comparison.csv',index=False)
    (out/'comparison.json').write_text(json.dumps(dict(scope='Matched single-seed 10k pilot; no superiority claim',
        results=rows),indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
