#!/usr/bin/env bash
# Matched artificial-masking ablation. Run from the repository root.
set -euo pipefail
scenarios=()
while IFS= read -r scenario; do
  scenarios+=("$scenario")
done < <(PYTHONPATH=data/experiments/forecast-geography-v2/code/src .venv/bin/python - <<'PY'
import csv
import pandas as pd
from pathlib import Path
if any(Path('data/experiments/forecast-no-mask-top32-v3').glob('*/s*/attempt-*')):
    raise RuntimeError('Do not re-plan an experiment with run attempts.')
from dataclasses import replace
from tapestry.model.scenario import Scenario
with open('data/experiments/forecast-geography-v2/jobs.csv') as stream:
    jobs=list(csv.DictReader(stream))
assert len(jobs)==63
ranking=pd.read_csv('data/experiments/forecast-geography-v2/ranking-e53fd92c4f32/configuration_ranking.csv').sort_values('combined_mean')
selected=set(ranking.head(32).config_id)
jobs=[j for j in jobs if j['scenario'] in selected]
assert len(jobs)==32
for job in jobs:
    scenario=Scenario.from_string(job['scenario'])
    assert scenario.mask_rate==.5
    print(replace(scenario,mask_rate=0.).scenario_string)
PY
)
[ "${#scenarios[@]}" -eq 32 ]
.venv/bin/python -m tapestry.experiment.planner plan -e forecast-no-mask-top32-v3 \
  -s "${scenarios[@]}" --seeds 42 43 44 --device cuda
# Preserve the exact trained implementation of the comparator. Only scenario inputs differ.
.venv/bin/python - <<'PY'
import csv,json,shutil
from pathlib import Path
source=Path('data/experiments/forecast-geography-v2')
output=Path('data/experiments/forecast-no-mask-top32-v3')
if any(output.glob('*/s*/attempt-*')):
    raise RuntimeError('Do not re-plan an experiment with run attempts.')
a=json.loads((source/'experiment.json').read_text());b=json.loads((output/'experiment.json').read_text())
for key in ['dataset_sha256','population_sha256','frozen_manifest_sha256','eval_members']:
    assert a[key]==b[key], f'Comparator mismatch: {key}'
shutil.rmtree(output/'code')
shutil.copytree(source/'code',output/'code')
shutil.copy2(source/'notifications/notify.py',output/'notifications/notify.py')
(output/'comparison.json').write_text(json.dumps(dict(comparator=str(source),changed_field='mask_rate',
    before=.5,after=0.,code_snapshot='Exact copy of comparator code snapshot',
    Wednesday_availability='Unchanged for targets and covariates',seeds=[42,43,44],selection='Top 32 of 63 original three-seed mean scores',ranking='ranking-e53fd92c4f32'),indent=2)+'\n')
PY
