"""Matched finalized-training/no-finality controls and their real-vintage replays."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
from tapestry.model.scenario import Scenario
from tapestry.experiment.planner import seed_state
from tapestry.experiment.provenance import save, sha256

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('mode', choices=('fit', 'replay'))
parser.add_argument('-e', '--experiment')
a = parser.parse_args()
source = Path('data/experiments/b2-reporting-controls-20261001')
experiment = a.experiment or ('b2-reporting-controls-20261001' if a.mode == 'fit' else 'b2-reporting-control-replay-20261001')
folder = Path('data/experiments') / experiment
if (folder/'jobs.csv').exists():
    raise SystemExit('Already planned; resume with manager status/run, without re-pinning')
settings = json.loads(Path('data/experiments/b-2-t0/experiment.json').read_text())
leaders = pd.read_csv('docs/experiments/b-2-t0/named_ranking.csv').head(5)
scenarios, checkpoints, labels = [], {}, {}
for row in leaders.itertuples():
    base = replace(Scenario.from_string(row.config_id), supplied_final=False)
    labels[base.run_id] = row.label
    if a.mode == 'fit':
        scenarios.append(base.scenario_string)
        continue
    for seed in (42,43,44):
        attempt, _, done = seed_state(source, base.scenario_string, seed)
        if not done:
            raise ValueError(f'Control not complete: {base.run_id}, seed {seed}')
        for season in base.scored_seasons:
            path = attempt/f'eval_{season}'/'model.pt'
            metadata = json.loads((path.parent/'manifest.json').read_text())
            if metadata['dataset_sha256'] != settings['dataset_sha256']:
                raise ValueError('Control data differs from B2')
            checkpoints[f'{base.run_id}/s{seed}/{season}'] = dict(path=str(path),sha256=sha256(path))
    scenarios.extend(replace(base,replay_from=str(source),replay_inputs=arm,replay_flags='off').scenario_string
                     for arm in ('vintage','nowcast'))
subprocess.run([sys.executable,'-m','tapestry.experiment.planner','plan','-e',experiment,'-s',*scenarios,
    '--seeds','42','43','44','--device','cuda','--eval-members','256',
    '--dataset',settings['dataset'],'--frozen',settings['frozen']],check=True)
if a.mode == 'fit':
    save(folder/'controls-source.json',dict(labels=labels,base_experiment='b-2-t0',
         purpose='Same no-finality architecture without reporting augmentation'))
else:
    save(folder/'replay-source.json',dict(source=str(source),dataset_sha256=settings['dataset_sha256'],
         checkpoints=checkpoints,labels=labels))
print(f'Planned {len(scenarios)} configurations × 3 seeds; mode={a.mode}')
