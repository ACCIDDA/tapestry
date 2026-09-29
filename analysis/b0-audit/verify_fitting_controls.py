"""Check selected epochs and forecast equality for the completed fitting controls."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tapestry.evaluation.hubs import CHANNEL

out = Path('docs/results/b0-fitting-controls')
runs = json.loads((out / 'manifest.json').read_text())['runs']
folds, arrays = {}, {}
epoch_rows = []
support = pd.read_csv(out / 'season_scores.csv')[['season', 'target']].drop_duplicates()
for run in runs:
    for season in ('2023-2024', '2024-2025', '2025-2026'):
        path = Path(run['path']) / f'eval_{season}'
        manifest = json.loads((path / 'manifest.json').read_text())
        records = [r for r in manifest['records'] if r['phase'] == 'select']
        epochs = [r['selected_epoch'] for r in records]
        key = (run['config_id'], run['seed'], season)
        folds[key] = epochs
        for record in records:
            epoch_rows.append(dict(arm=run['config_id'], seed=run['seed'], season=season,
                                   channels=json.dumps(record['channels']), epoch=record['selected_epoch']))
        with np.load(path / 'forecasts.npz') as data:
            arrays[key] = {name: data[name] for name in data.files}

rows = []
for before, after in [('current', 'draws'), ('split', 'split-draws'), ('current', 'split')]:
    for seed in (42, 43, 44):
        for season in ('2023-2024', '2024-2025', '2025-2026'):
            k0, k1 = (before, seed, season), (after, seed, season)
            a, b = arrays[k0], arrays[k1]
            for name in ('quantile_levels', 'truth', 'mask', 'context_end', 'target_dates', 'locations'):
                assert np.array_equal(a[name], b[name]), (k0, k1, name)
            q0, q1 = a['quantiles'], b['quantiles']
            row = dict(before=before, after=after, seed=seed, season=season,
                       equal_epochs=folds[k0] == folds[k1],
                       epochs_before=json.dumps(folds[k0]), epochs_after=json.dumps(folds[k1]),
                       byte_identical=q0.dtype == q1.dtype and q0.tobytes() == q1.tobytes(),
                       max_abs_difference=float(np.max(np.abs(q0-q1))))
            for target, channel in CHANNEL.items():
                row[target] = q0[:, :, :, channel, :].tobytes() == q1[:, :, :, channel, :].tobytes()
            row['supported_targets_equal'] = all(row[target] for target in support[support.season.eq(season)].target)
            rows.append(row)
            if (before, after) == ('current', 'split') and season == '2023-2024':
                assert row['equal_epochs'] and row['byte_identical'], row
pd.DataFrame(rows).to_csv(out / 'forecast_comparisons.csv', index=False)
pd.DataFrame(epoch_rows).to_csv(out / 'selected_epochs.csv', index=False)
print(pd.DataFrame(rows).groupby(['before', 'after']).agg(
    folds=('season', 'size'), equal_epochs=('equal_epochs', 'sum'), byte_identical=('byte_identical', 'sum')).to_string())
