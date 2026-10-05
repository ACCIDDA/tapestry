"""Plan C1–C3 fixed-checkpoint replay via the shared experiment manager."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
from tapestry.model.scenario import Scenario
from tapestry.experiment.planner import seed_state
from tapestry.experiment.provenance import sha256, save

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('-e','--experiment',default='b2-input-replay-20261001')
p.add_argument('--flag-controls',action='store_true',help='Final inputs with flag off; vintage/nowcast with B2 training encoding')
a=p.parse_args()
source=Path('data/experiments/b-2-t0')
settings=json.loads((source/'experiment.json').read_text())
leaders=pd.read_csv('docs/experiments/b-2-t0/named_ranking.csv').head(3)
folder=Path('data/experiments')/a.experiment
if (folder/'jobs.csv').exists():
    raise SystemExit('Already planned; use manager status/run to resume without re-pinning checkpoints')
scenarios=[];checkpoints={};labels={}
for row in leaders.itertuples():
    base=Scenario.from_string(row.config_id)
    labels[base.run_id]=row.label
    for seed in (42,43,44):
        attempt,_,complete=seed_state(source,base.scenario_string,seed)
        if not complete:raise ValueError(f'Missing completed source {base.run_id}, seed {seed}')
        for fold in base.scored_seasons:
            path=attempt/f'eval_{fold}'/'model.pt'
            metadata=json.loads((path.parent/'manifest.json').read_text())
            if metadata['dataset_sha256']!=settings['dataset_sha256']:
                raise ValueError('B2 checkpoints do not share the pinned panel')
            checkpoints[f'{base.run_id}/s{seed}/{fold}']=dict(path=str(path),sha256=sha256(path))
    arms = [('finalized','off'),('vintage','available'),('nowcast','available')] if a.flag_controls else [
        (arm,'native') for arm in ('finalized','vintage','nowcast')]
    scenarios.extend(replace(base,replay_from=str(source),replay_inputs=arm,replay_flags=flags).scenario_string
                     for arm,flags in arms)
subprocess.run([sys.executable,'-m','tapestry.experiment.planner','plan','-e',a.experiment,
    '-s',*scenarios,'--seeds','42','43','44','--device','cuda','--eval-members','256',
    '--dataset',settings['dataset'],'--frozen',settings['frozen']],check=True)
save(folder/'replay-source.json',dict(source=str(source),dataset_sha256=settings['dataset_sha256'],
    checkpoints=checkpoints,labels=labels))
print('Pinned 18 original CV checkpoints; planned 27 replay runs / 54 fold evaluations.')
