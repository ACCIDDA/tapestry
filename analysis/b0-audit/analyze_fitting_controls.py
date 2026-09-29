"""Rank the completed 2x2 fitting experiment and write paired effects/interactions."""
import json
from pathlib import Path

import pandas as pd
from tapestry.evaluation.totals import rank
from tapestry.experiment.planner import read_jobs, seed_state

out = Path('docs/results/b0-fitting-controls')
out.mkdir(parents=True, exist_ok=True)
arms = {'current': 'b0-current-training-normalized',
        'split': 'b0-fitting-split', 'draws': 'b0-fitting-draws',
        'split-draws': 'b0-fitting-split-draws'}
runs, progress = [], []
for arm, experiment in arms.items():
    root = Path('data/experiments') / experiment
    job = next(j for j in read_jobs(root) if 'input_mode=finalized_available' not in j['scenario'])
    for seed in job['seeds']:
        attempt, record, complete = seed_state(root, job['scenario'], seed)
        progress.append(dict(arm=arm, seed=seed, status=record['status'], complete=complete,
                             folds=len(list(attempt.glob('eval_*/forecasts.npz'))) if attempt else 0))
        if complete:
            runs.append(dict(config_id=arm, seed=seed, path=attempt))
progress = pd.DataFrame(progress)
progress.to_csv(out / 'progress.csv', index=False)
print(progress.to_string(index=False), flush=True)
if not progress.complete.all():
    raise SystemExit(0)
assert len(runs) == 12
ranking = rank(runs, out)
effects = []
for file, scope in [('run_scores.csv', 'all-seasons'), ('season_composite_scores.csv', 'season')]:
    scores = pd.read_csv(out / file)
    keys = ['seed', 'geography'] + (['season'] if scope == 'season' else [])
    wide = scores.pivot(index=keys, columns='config_id', values='combined')
    for effect, before, after in [('split_with_current_draws', 'current', 'split'),
                                  ('split_with_b0_draws', 'draws', 'split-draws'),
                                  ('draws_with_current_split', 'current', 'draws'),
                                  ('draws_with_b0_split', 'split', 'split-draws'),
                                  ('both_changes', 'current', 'split-draws')]:
        part = wide.reset_index()[keys].copy()
        part['scope'], part['effect'] = scope, effect
        part['before'], part['after'] = wide[before].to_numpy(), wide[after].to_numpy()
        part['delta'] = part.after - part.before
        effects.append(part)
    part = wide.reset_index()[keys].copy()
    part['scope'], part['effect'] = scope, 'interaction'
    part['delta'] = (wide['split-draws'] - wide['split'] - wide['draws'] + wide['current']).to_numpy()
    effects.append(part)
paired = pd.concat(effects, ignore_index=True)
paired.to_csv(out / 'paired_effects.csv', index=False)
summary = paired.groupby(['scope', 'season', 'geography', 'effect'], dropna=False).delta.agg(['mean', 'std', 'count'])
summary.to_csv(out / 'effect_summary.csv')
print(ranking[['config_id', 'combined_mean', 'combined_sd']].to_string(index=False))
print(summary.to_string())
