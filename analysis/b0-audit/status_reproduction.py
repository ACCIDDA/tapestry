"""Compact progress for the reproduction and its paired controls."""
import json
from pathlib import Path
from tapestry.experiment.planner import read_jobs,seed_state
for name in ('b0-exact-reproduction','b0-exact-reproduction-l40','b0-current-training-control','b0-current-training-normalized','b0-training-code-reference'):
    root=Path('data/experiments')/name
    for job in read_jobs(root):
        for seed in job['seeds']:
            run,record,done=seed_state(root,job['scenario'],seed)
            label=('target' if 'fit_partition=target' in job['scenario'] else 'pathogen')+('/Wednesday' if 'finalized_available' in job['scenario'] else '/final')
            if run is None:print(name,label,seed,'planned');continue
            evaluation_root=run/'legacy' if (run/'legacy').exists() else run
            folds=len(list(evaluation_root.glob('eval_*/forecasts.npz')))
            lines=(run/'run.log').read_text().splitlines()
            latest=next((line for line in reversed(lines) if line.startswith('{"epoch"') or line.startswith('{"component"') or line.startswith('{"evaluated"')),'')
            print(name,label,seed,record['status'],f'{folds}/3 folds',latest,flush=True)
