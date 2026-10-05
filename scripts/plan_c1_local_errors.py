"""Plan four C1 nowcast-error treatments, three seeds, two held-out seasons."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
from tapestry.model.scenario import Scenario
from tapestry.experiment.provenance import save, sha256

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('-e', '--experiment', default='c1-local-errors-20261002')
a = p.parse_args()
folder = Path('data/experiments') / a.experiment
if (folder / 'jobs.csv').exists():
    raise SystemExit('Already planned; use manager status/run to resume')
settings = json.loads(Path('data/experiments/b-2-t0/experiment.json').read_text())
if sha256(settings['dataset']) != settings['dataset_sha256']:
    raise ValueError('B2 dataset changed')
base = Scenario.from_string(pd.read_csv('docs/experiments/b-2-t0/named_ranking.csv').iloc[0].config_id)
scenarios = [replace(base, supplied_final=False, reporting_augmentation='nowcast',
                    reporting_missingness=False, reporting_method=method, reporting_strength=strength)
             for method, strength in [('local_log', 1.), ('local_log', .5),
                                      ('calendar_log', 1.), ('local_additive', 1.)]]
subprocess.run([sys.executable, '-m', 'tapestry.experiment.planner', 'plan', '-e', a.experiment,
    '-s', *[s.scenario_string for s in scenarios], '--seeds', '42', '43', '44', '--device', 'cuda',
    '--eval-members', '256', '--dataset', settings['dataset'], '--frozen', settings['frozen']], check=True)
# The nowcaster and its folds are unchanged. Reuse only its cached numerical
# predictions; prepare()/cached_nowcasts() still verify their exact provenance.
old = Path('data/experiments/b2-reporting-values-only-20261002')
for pattern in ('augmentation-nowcasts-*.npz', 'replay-nowcasts.npz', 'replay-nowcast-parity.json'):
    for path in old.glob(pattern):
        shutil.copy2(path, folder / path.name)
save(folder / 'augmentation-source.json', dict(model='C1',
    labels={s.run_id: dict(method=s.reporting_method, strength=s.reporting_strength) for s in scenarios},
    source_experiment='b-2-t0', seeds=[42,43,44],
    evaluation_seasons=list(base.scored_seasons),
    comparison='C1 finalized-training nowcast replay and previous numerical-error-only nowcast training',
    labels_unchanged=True, reporting_missingness=False,
    copied_nowcaster_cache_from=str(old),
    caveat='2024-2025 fold is retrospective; 2025-2026 is forward relative to forecast training'))
print('Planned four treatments x three seeds x two folds = 24 C1 fits/evaluations.')
