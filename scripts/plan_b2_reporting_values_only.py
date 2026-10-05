"""Plan the focused C1 numerical-error-only augmentation diagnostic."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
from tapestry.model.scenario import Scenario
from tapestry.experiment.provenance import save, sha256

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('-e','--experiment',default='b2-reporting-values-only-20261002')
a=p.parse_args()
folder=Path('data/experiments')/a.experiment
if (folder/'jobs.csv').exists():raise SystemExit('Already planned; use manager status/run to resume')
settings=json.loads(Path('data/experiments/b-2-t0/experiment.json').read_text())
if sha256(settings['dataset'])!=settings['dataset_sha256']:raise ValueError('B2 dataset changed')
row=pd.read_csv('docs/experiments/b-2-t0/named_ranking.csv').iloc[0]
base=Scenario.from_string(row.config_id)
scenarios=[replace(base,supplied_final=False,reporting_augmentation=arm,reporting_missingness=False)
           for arm in ('vintage','nowcast')]
subprocess.run([sys.executable,'-m','tapestry.experiment.planner','plan','-e',a.experiment,
    '-s',*[s.scenario_string for s in scenarios],'--seeds','42','43','44','--device','cuda',
    '--eval-members','256','--dataset',settings['dataset'],'--frozen',settings['frozen']],check=True)
save(folder/'augmentation-source.json',dict(labels={s.run_id:dict(label='C1',arm=s.reporting_augmentation) for s in scenarios},
    hypothesis='Transporting old reporting missingness may cause the forward loss; preserve native availability.',
    diagnostic_only=True,source_experiment='b2-reporting-augmentation-20261001',node='g1803jles01'))
print('Planned C1, raw/residual numerical errors only, three seeds: 6 runs / 12 fold evaluations.')
