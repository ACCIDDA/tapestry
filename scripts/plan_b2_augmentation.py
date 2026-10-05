"""Plan retraining of the five named B2 leaders with raw and corrected revision errors."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
from tapestry.model.scenario import Scenario
from tapestry.experiment.provenance import save

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('-e', '--experiment', default='b2-reporting-augmentation-20261001')
a = p.parse_args()
source = Path('data/experiments/b-2-t0')
settings = json.loads((source / 'experiment.json').read_text())
leaders = pd.read_csv('docs/experiments/b-2-t0/named_ranking.csv').head(5)
folder = Path('data/experiments') / a.experiment
if (folder / 'jobs.csv').exists():
    raise SystemExit('Already planned; use manager status/run to resume')
scenarios, labels = [], {}
for row in leaders.itertuples():
    base = Scenario.from_string(row.config_id)
    for arm in ('vintage', 'nowcast'):
        scenario = replace(base, supplied_final=False, reporting_augmentation=arm)
        scenarios.append(scenario.scenario_string)
        labels[scenario.run_id] = dict(label=row.label, arm=arm, source_scenario=base.scenario_string)
subprocess.run([sys.executable, '-m', 'tapestry.experiment.planner', 'plan', '-e', a.experiment,
    '-s', *scenarios, '--seeds', '42', '43', '44', '--device', 'cuda', '--eval-members', '256',
    '--dataset', settings['dataset'], '--frozen', settings['frozen']], check=True)
planned = json.loads((folder / 'experiment.json').read_text())
if planned['dataset_sha256'] != settings['dataset_sha256']:
    raise ValueError('Dataset differs from the original B2 experiment')
save(folder / 'augmentation-source.json', dict(source=str(source), labels=labels,
    dataset_sha256=settings['dataset_sha256'], node='g1803jles01', seeds=[42, 43, 44]))
print('Planned 5 B2 configurations × 2 augmentation arms × 3 seeds = 30 retraining runs / 60 scored folds.')
