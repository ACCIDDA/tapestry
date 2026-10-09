"""Combine a release's operational forecasts and write the Hub quantile file, with its checks.

Recipes are combined with `evaluation.ensembles.combine` (equal weight per recipe, equal
weight per seed within a recipe, the release's rule) for the targets each recipe was
trained to predict. Checks against the local Hub clone: reference date in the current
round, quantile grid, units, ordering, horizon dates, no duplicate tasks. ED proportions
above the Hub's `max_prop_ed_visits` (0.25) are capped and each cap is recorded.
Writes `<reference>-ACCIDDA-<model_abbr>.csv` and a JSON record of the sources, hashes and
release; nothing is published.

History: scripts/export_b6.py (2026-10-07), unchanged in what it computes.
"""
from datetime import date, timedelta
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from tapestry.evaluation.ensembles import combine, trained_channels
from tapestry.evaluation.quantiles import LEVELS
from tapestry.experiment.provenance import sha256

TARGETS = [(0, 'wk inc flu hosp'), (3, 'wk inc flu prop ed visits')]
MAX_ED = .25  # Hub validations.yml max_prop_ed_visits


def export(release, members, hub, out):
    """`members` {recipe: [operational forecast folders]} from `forecast.forecast_release`."""
    if not re.fullmatch(r'[A-Za-z0-9_+]{1,16}', release['model_abbr']):
        raise ValueError('Invalid model abbreviation')
    if release['rule'] not in ('vincent', 'mixture'):
        raise ValueError('Unknown ensemble rule')
    hub, out = Path(hub), Path(out)
    recipes, sources, first = [], [], None
    panel_hashes, issues = set(), set()
    for name, folders in members.items():
        values, scenarios, seeds = [], set(), []
        for folder in map(Path, folders):
            meta = json.loads((folder / 'operational-manifest.json').read_text())
            scenarios.add(meta['scenario'])
            seeds.append(meta['seed'])
            panel_hashes.add(meta['operational_dataset_sha256'])
            issues.add(meta['issuance'])
            with np.load(folder / 'forecasts.npz') as f:
                data = dict(f)
            if first is None:
                first = data
            for key in ('context_end', 'target_dates', 'locations', 'quantile_levels'):
                if not np.array_equal(data[key], first[key]):
                    raise ValueError(f'Misaligned operational {key}')
            if data['mask'].any() or data['quantiles'].shape[1] != 1:
                raise ValueError('Expected one issuance with unknown future labels')
            values.append(data['quantiles'])
            sources.append(dict(recipe=name, directory=str(folder), checkpoint_sha256=meta['checkpoint_sha256'],
                                forecast_sha256=sha256(folder / 'forecasts.npz')))
        if len(scenarios) != 1 or len(set(seeds)) != len(seeds):
            raise ValueError(f'{name}: inconsistent recipe or duplicate seed')
        recipes.append(dict(channels=trained_channels(scenarios.pop()), values=np.stack(values)))
    if len(panel_hashes) != 1 or len(issues) != 1 or not np.array_equal(first['quantile_levels'], LEVELS):
        raise ValueError('Inconsistent operational snapshot, issuance or quantile grid')
    reference = date.fromisoformat(str(first['context_end'][0])) + timedelta(days=7)
    tasks = json.loads((hub / 'hub-config/tasks.json').read_text())['rounds'][0]['model_tasks']
    locations = pd.read_csv(hub / 'auxiliary-data/locations.csv', dtype={'location': str})
    mapping = dict(zip(locations.abbreviation, locations.location))
    allowed = lambda spec: (spec.get('required') or []) + (spec.get('optional') or [])
    rows, adjustments = [], []
    for channel, target in TARGETS:
        if not any(channel in r['channels'] for r in recipes):
            continue
        task, = [t for t in tasks if target in allowed(t['task_ids']['target'])]
        if reference.isoformat() not in allowed(task['task_ids']['reference_date']):
            raise ValueError('Reference date not in current Hub tasks')
        if not np.array_equal(task['output_type']['quantile']['output_type_id']['required'], LEVELS):
            raise ValueError('Current Hub quantile grid changed')
        q = combine(recipes, channel, release['rule'])[:, 0]
        if channel == 0:
            q = np.floor(q + .5)
        if not np.isfinite(q).all() or (q < 0).any() or (np.diff(q, axis=0) < 0).any() or (channel == 3 and (q > 1).any()):
            raise ValueError('Invalid quantile values, units or ordering')
        if channel == 3:
            for qi, h, li in np.argwhere(q > MAX_ED):
                adjustments.append(dict(target=target, horizon=int(h), location=str(first['locations'][li]),
                                        quantile=float(LEVELS[qi]), original=float(q[qi, h, li]), exported=MAX_ED,
                                        reason=f'Hub max_prop_ed_visits={MAX_ED}'))
            q = np.minimum(q, MAX_ED)
        for h, day in enumerate(first['target_dates'][0]):
            if str(day) != (reference + timedelta(weeks=h)).isoformat() or h not in allowed(task['task_ids']['horizon']):
                raise ValueError('Invalid horizon/date alignment')
            for li, loc in enumerate(first['locations']):
                fips = mapping[str(loc)]
                if fips not in allowed(task['task_ids']['location']):
                    continue
                for qi, level in enumerate(LEVELS):
                    rows.append(dict(reference_date=reference.isoformat(), target=target, horizon=h,
                                     target_end_date=str(day), location=fips, output_type='quantile',
                                     output_type_id=level, value=int(q[qi, h, li]) if channel == 0 else float(q[qi, h, li])))
    table = pd.DataFrame(rows)
    if table.empty or table.drop(columns='value').duplicated().any():
        raise ValueError('Empty or duplicate Hub tasks')
    out.mkdir(parents=True, exist_ok=True)
    path = out / f'{reference}-ACCIDDA-{release["model_abbr"]}.csv'
    table.to_csv(path, index=False)
    record = dict(release=release, sources=sources, adjustments=adjustments, rows=len(table), issuance=next(iter(issues)),
                  operational_panel_sha256=next(iter(panel_hashes)), hub_tasks_sha256=sha256(hub / 'hub-config/tasks.json'),
                  output_sha256=sha256(path), definition=dict(note=release['description']))
    path.with_suffix('.json').write_text(json.dumps(record, indent=2) + '\n')
    return path
