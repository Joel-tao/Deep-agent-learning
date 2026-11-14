"""Summarize a single-seed pilot without treating it as a controlled comparison."""
import csv
import json
from pathlib import Path


def main():
    root=Path('runs/hold_pilot_10k/fixed/seed_42')
    evaluations=list(csv.DictReader((root/'evaluations.csv').open()))
    episodes=list(csv.DictReader((root/'episode_metrics.csv').open()))
    result=dict(scope='Single-seed 10000-step pilot; no causal or convergence claim',
                monitoring_evaluations=evaluations,completed_training_episodes=len(episodes))
    for key in ['reached','grasped','success','stable_grasp','stable_lift','max_contact_steps']:
        values=[float(e[key]=='True') if e[key] in ('True','False') else float(e[key])
                for e in episodes if e.get(key) not in ('',None)]
        result['training_mean_'+key]=sum(values)/len(values) if values else None
        if key=='max_contact_steps':result['longest_training_contact_steps']=max(values) if values else None
    result['independent_diagnostics']={}
    for name in ['initial','final','best']:
        p=Path('diagnostics')/('hold_pilot_10k_'+name)/'summary.json'
        if p.exists():
            data=json.loads(p.read_text())
            result['independent_diagnostics'][name]=dict(rates=data['rates'],
                episodes=len(data['episodes']),
                longest_contact_steps=max(e['max_contact_steps'] for e in data['episodes']))
    (root.parent.parent/'pilot_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
