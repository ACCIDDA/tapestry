"""Score completed runs and rank configurations: the one internal evaluation.

Every run (a fitted configuration and seed, or a saved-forecast ensemble from
`evaluation.ensembles`) holds `eval_<season>/forecasts.npz` for the `raw` input view and
`eval_<season>/<view>/forecasts.npz` for the others (`experiment/fit.py`); an ensemble holds
only the views it was built for. `views` exposes each view present as a run folder,
`cache_scores` scores each once (immutable cache), and `rank` writes the experiment tables.

Each run is ranked in its recipe's own forecast-time view (`Scenario.forecast_view`, or an
ensemble's `forecast_view`), the view production uses; the other views are scored as
diagnostics (`deployed` is False) and never ranked (2026-10-09 review: picking each
configuration's best view after scoring was an extra selection on the evaluation data).

Headline score (user decisions 2026-10-05 and 2026-10-08): raw WIS on every task whose
reference date falls in October-May, horizons 0-3, finalized truth; mean per task with
states/DC pooled and the US separate (no 80/20 mixture); admissions on the natural and
log(x + 1) scales, ED proportions untransformed; held-out seasons averaged equally, then
fitting seeds. Lower is better. Alongside it: WIS relative to the Hub ensemble on frozen
Hub tasks, the official-style pairwise relative WIS among Hub models
(`evaluation.standard`), score details by month, horizon and location, and coverage.

History: this replaces the B0 composite ranking (`totals.rank`, per-location ratios with
states 80% / US 20% and admissions weighted twice ED) and the B3-B7 `rank_pilot` in
`experiment/pilot.py`, whose prescribed-revision branch it keeps.
"""
import json
from pathlib import Path

import pandas as pd

from .standard import hub_relative, raw_wis, pairwise, raw_score_details
from .distribution import distribution_scores

VIEWS = ('half', 'corrected', 'calibrated', 'sampled', 'delayed', 'nokinsa')
CACHE = ('scores-hub-relative.csv', 'scores-raw.csv', 'scores-pairwise.csv', 'scores-details.csv',
         'scores-distribution.csv')
# Bump when a cached table gains columns; older caches are then recomputed.
CACHE_VERSION = '3'  # 2: raw WIS components and coverage; 3 (2026-10-09): Hub pool kept for the own view
METRICS = ['mean_wis', 'mean_dispersion', 'mean_underprediction', 'mean_overprediction',
           'covered_50', 'covered_80', 'covered_90', 'covered_95']


def link_view(run, name, manifest, source):
    """A run-shaped folder of symlinks: `run/evaluation-<name>/eval_<season>/forecasts.npz`."""
    view = run / f'evaluation-{name}'
    view.mkdir(exist_ok=True)
    (view / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for season in manifest['folds']:
        folder = view / f'eval_{season}'
        folder.mkdir(exist_ok=True)
        dest = folder / 'forecasts.npz'
        if not dest.exists():
            dest.symlink_to(source(season).resolve())
    return view


def views(run):
    """{view name: run-shaped folder} for every input view present in every season of `run`.

    `raw` is the run itself, when its forecasts exist. With several artificial evaluation
    histories (`evaluation_draws`), `drawK_<view>` holds draw K of raw, half and corrected."""
    run = Path(run)
    manifest = json.loads((run / 'manifest.json').read_text())
    present = all((run / f'eval_{season}' / 'forecasts.npz').exists() for season in manifest['folds'])
    result = {'raw': run} if present else {}
    for name in VIEWS:
        if all((run / f'eval_{season}' / name).exists() for season in manifest['folds']):
            result[name] = link_view(run, name, manifest, lambda s, n=name: run / f'eval_{s}' / n / 'forecasts.npz')
    draw = 1
    while all((run / f'eval_{season}' / f'draw-{draw}').exists() for season in manifest['folds']):
        for name in ('raw', 'half', 'corrected'):
            folder = lambda s, n=name, d=draw: run / f'eval_{s}' / f'draw-{d}' / ('' if n == 'raw' else n) / 'forecasts.npz'
            if all(folder(season).exists() for season in manifest['folds']):
                result[f'draw{draw}_{name}'] = link_view(run, f'draw{draw}_{name}', manifest, folder)
        draw += 1
    return result


def deployed_view(run):
    """The view a run is ranked and deployed in: its recipe's `forecast_view`."""
    manifest = json.loads((Path(run) / 'manifest.json').read_text())
    if 'forecast_view' in manifest:
        return manifest['forecast_view']
    from tapestry.model.scenario import Scenario
    return Scenario.from_string(manifest['scenario']).forecast_view


def cache_scores(run, frozen):
    """Score each view of one immutable completed run once, in its own folder."""
    run = Path(run)
    version = run / 'scores-version.txt'
    if all((run / name).exists() for name in CACHE) and version.exists() and version.read_text().strip() == CACHE_VERSION:
        return
    tables = {name: [] for name in CACHE}
    own = deployed_view(run)
    for label, path in views(run).items():
        draw = int(label.split('_')[0][4:]) if label.startswith('draw') else 0
        history = label.split('_', 1)[1] if label.startswith('draw') else label
        tag = dict(history=history, revision_draw=draw)
        relative = hub_relative(path, frozen)
        tables['scores-hub-relative.csv'].append(relative[relative.geography != 'all'].assign(**tag))
        tables['scores-raw.csv'].append(raw_wis(path, frozen).assign(**tag))
        table = pairwise(path, frozen)
        # Every Hub model's value in this run's pool, for the own view only (the figure's background).
        tables['scores-pairwise.csv'].append((table if label == own else table[table.ours]).assign(**tag))
        tables['scores-details.csv'].append(raw_score_details(path).assign(**tag))
    tables['scores-distribution.csv'] = [distribution_scores(run)]
    for name, frames in tables.items():
        pd.concat(frames, ignore_index=True).to_csv(run / name, index=False)
    version.write_text(CACHE_VERSION + '\n')


def rank(runs, frozen, destination):
    """Rank runs [{'config_id', 'name', 'seed', 'path'}] into `destination`.

    Requires every run to be scored on the same tasks and truth (support hash) and in
    every season its scenario evaluates; ensembles carry their own season list."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for run in runs:
        cache_scores(run['path'], frozen)
        run['view'] = deployed_view(run['path'])
    tag = lambda frame, run: frame.assign(config_id=run['config_id'], name=run['name'], seed=run['seed'],
                                          deployed=frame.history.eq(run['view']) if 'history' in frame else False)
    read = lambda name: pd.concat([tag(pd.read_csv(Path(r['path']) / name, dtype={'value': str}), r) for r in runs],
                                  ignore_index=True)
    raw = read('scores-raw.csv')
    missing = [f'{r["name"]} seed {r["seed"]}: {r["view"]}' for r in runs
               if not ((raw.config_id == r['config_id']) & (raw.seed == r['seed']) & raw.deployed).any()]
    if missing:
        raise ValueError(f'Runs lack scores in their own forecast_view: {missing}')
    support = raw.groupby(['target', 'season', 'scale', 'geography']).support_sha256.nunique()
    if not support.eq(1).all():
        raise ValueError('Headline comparisons have different truth/task support: '
                         f'{support[support > 1].index.tolist()}')
    keys = ['config_id', 'name', 'seed', 'history', 'deployed', 'target', 'scale', 'geography']
    means = raw.groupby(keys).agg(score=('mean_wis', 'mean'), seasons=('season', 'nunique')).reset_index()
    for run in runs:
        expected = len(json.loads((Path(run['path']) / 'manifest.json').read_text())['folds'])
        mine = means[(means.config_id == run['config_id']) & (means.seed == run['seed'])]
        if not mine.seasons.eq(expected).all():
            raise ValueError(f'{run["name"]} seed {run["seed"]}: every target must be scored in every evaluated season')
    # The full scenario string (config_id) appears only in headline-rankings.csv; the other
    # tables identify configurations by name, which keeps them small.
    means.drop(columns='config_id').to_csv(destination / 'headline-seed-scores.csv', index=False, float_format='%.6g')
    # Every metric (WIS, its components, coverage) per run: seasons averaged equally.
    raw.groupby(keys)[METRICS].mean().reset_index().drop(columns='config_id') \
        .to_csv(destination / 'metric-seed-scores.csv', index=False, float_format='%.6g')
    summary = means.groupby([k for k in keys if k != 'seed']).score.agg(['mean', 'std', 'count']).reset_index()
    # Only each recipe's own view competes; other views are diagnostics without a rank.
    summary['rank'] = summary[summary.deployed].groupby(['target', 'scale', 'geography'])['mean'].rank(method='min')
    summary = summary.sort_values(['target', 'scale', 'geography', 'rank'])
    summary.to_csv(destination / 'headline-rankings.csv', index=False, float_format='%.6g')
    # Compact tables only (user rule, 2026-10-09): seed and draw averages; per-location and
    # per-seed detail stays in each run's cached scores-*.csv.
    config = ['name', 'history', 'deployed']
    raw.groupby([*config, 'season', 'target', 'scale', 'geography'])[METRICS].mean().reset_index() \
        .to_csv(destination / 'headline-season-scores.csv', index=False, float_format='%.6g')
    details = read('scores-details.csv')
    details = details[details.deployed]
    measures = [c for c in details if c == 'wis' or c.startswith('covered_')]
    details.groupby([*config, 'season', 'target', 'scale', 'geography', 'dimension', 'value'])[measures + ['tasks']] \
        .mean().reset_index().to_csv(destination / 'score-details.csv', index=False, float_format='%.6g')
    relative = read('scores-hub-relative.csv')
    relative.groupby([*config, 'hub', 'season', 'target', 'scale', 'geography']).wis_ratio.mean().reset_index() \
        .to_csv(destination / 'hub-relative-scores.csv', index=False, float_format='%.6g')
    pool = read('scores-pairwise.csv')
    pool = pool[pool.deployed | pool.ours]
    ours = pool[pool.ours].groupby([*config, 'hub', 'season', 'target', 'scale']).agg(
        relative_wis=('relative_wis', 'mean'), rank=('rank', 'mean'), models=('models', 'first')).reset_index()
    hub = pool[~pool.ours].groupby(['model', 'hub', 'season', 'target', 'scale']).agg(
        relative_wis=('relative_wis', 'mean'), google=('google', 'first')).reset_index()
    pd.concat([ours.assign(ours=True), hub.assign(ours=False)], ignore_index=True) \
        .to_csv(destination / 'hub-pairwise-scores.csv', index=False, float_format='%.6g')
    distribution = read('scores-distribution.csv')
    distribution = distribution[distribution.deployed]
    distribution['geography'] = distribution.location.astype(str).eq('US').map({True: 'US', False: 'states_dc'})
    numeric = [c for c in distribution if c.startswith(('weekly_', 'all_four_', 'four_week_'))]
    distribution.groupby([*config, 'season', 'target', 'geography'])[numeric].mean().reset_index() \
        .to_csv(destination / 'distribution-scores.csv', index=False, float_format='%.6g')
    large = [p.name for p in destination.glob('*.csv') if p.stat().st_size > 1_000_000]
    if large:
        print(f'Warning: ranking tables over 1 MB are not copied to the report: {large}', flush=True)
    (destination / 'manifest.json').write_text(json.dumps(dict(
        runs=[str(r['path']) for r in runs], definition=__doc__,
        views={f'{r["name"]} s{r["seed"]}': r['view'] for r in runs},
        support_sha256={'/'.join(k): v for k, v in raw.groupby(['target', 'season', 'scale', 'geography'])
                        .support_sha256.first().items()},
        selection_metric='states/DC log-admission WIS in each recipe\'s forecast_view (headline-rankings.csv, '
                         'deployed rows); lower is better'), indent=2) + '\n')
    print('Headline: October-May, seasons equal; states/DC and US separate; lower WIS is better.', flush=True)
    print(summary[summary.deployed & (summary.target == 'wk inc flu hosp') & (summary.scale == 'log') & (summary.geography == 'states_dc')]
          .sort_values('rank')[['name', 'history', 'mean', 'std', 'count', 'rank']].head(20).to_string(index=False), flush=True)
    return summary
