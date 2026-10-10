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
HUB_RELATIVE_COLUMNS = ['hub', 'target', 'season', 'scale', 'geography', 'ensemble', 'wis_ratio', 'tasks',
                        'history', 'revision_draw']
PAIRWISE_COLUMNS = ['hub', 'target', 'season', 'scale', 'model', 'relative_wis', 'rank', 'models', 'tasks', 'hub_tasks',
                    'qualifies', 'google', 'ours', 'history', 'revision_draw']
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


def headline_pathogen(runs):
    """Group of the common problem's headline target."""
    from chromantis.problem import Problem
    manifests = [json.loads((Path(r['path']) / 'manifest.json').read_text()) for r in runs]
    hashes = {m['problem_sha256'] for m in manifests}
    if len(hashes) != 1:
        raise ValueError('One ranking may contain only one problem definition')
    problem = Problem.load(manifests[0]['problem'])
    return problem.dataset.by_name[problem.headline_target].group


def deployed_view(run):
    """The view a run is ranked and deployed in: its recipe's `forecast_view`."""
    manifest = json.loads((Path(run) / 'manifest.json').read_text())
    if 'forecast_view' in manifest:
        return manifest['forecast_view']
    from chromantis.model.scenario import Scenario
    return Scenario.from_string(manifest['scenario']).forecast_view


def cache_scores(run, frozen):
    """Score each view of one immutable completed run once, in its own folder."""
    run = Path(run)
    version = run / 'scores-version.txt'
    if all((run / name).exists() for name in CACHE) and version.exists() and version.read_text().strip() == CACHE_VERSION:
        return
    tables = {name: [] for name in CACHE}
    own = deployed_view(run)
    from chromantis.problem import Problem
    compared = Problem.load(json.loads((run / 'manifest.json').read_text())['problem']).comparison.get('kind') == 'hub'
    for label, path in views(run).items():
        draw = int(label.split('_')[0][4:]) if label.startswith('draw') else 0
        history = label.split('_', 1)[1] if label.startswith('draw') else label
        tag = dict(history=history, revision_draw=draw)
        tables['scores-raw.csv'].append(raw_wis(path, frozen).assign(**tag))
        if compared:
            relative = hub_relative(path, frozen)
            tables['scores-hub-relative.csv'].append(relative[relative.geography != 'all'].assign(**tag))
            table = pairwise(path, frozen)
            # Every Hub model's value in this run's pool, for the own view only (the figure's background).
            tables['scores-pairwise.csv'].append((table if label == own else table[table.ours]).assign(**tag))
        else:  # no comparison declared by the problem: header-only tables keep `rank` uniform
            tables['scores-hub-relative.csv'].append(pd.DataFrame(columns=HUB_RELATIVE_COLUMNS))
            tables['scores-pairwise.csv'].append(pd.DataFrame(columns=PAIRWISE_COLUMNS))
        tables['scores-details.csv'].append(raw_score_details(path).assign(**tag))
    tables['scores-distribution.csv'] = [distribution_scores(run)]
    for name, frames in tables.items():
        pd.concat(frames, ignore_index=True).to_csv(run / name, index=False)
    version.write_text(CACHE_VERSION + '\n')



def hub_tables(relative, pool, config, destination):
    """Hub-relative WIS ratios and the pairwise pool, averaged over seeds and draws."""
    relative.groupby([*config, 'hub', 'season', 'target', 'scale', 'geography']).wis_ratio.mean().reset_index() \
        .to_csv(destination / 'hub-relative-scores.csv', index=False, float_format='%.6g')
    pool = pool[pool.deployed | pool.ours]
    ours = pool[pool.ours].groupby([*config, 'hub', 'season', 'target', 'scale']).agg(
        relative_wis=('relative_wis', 'mean'), rank=('rank', 'mean'), models=('models', 'first')).reset_index()
    hub = pool[~pool.ours].groupby(['model', 'hub', 'season', 'target', 'scale']).agg(
        relative_wis=('relative_wis', 'mean'), google=('google', 'first')).reset_index()
    pd.concat([ours.assign(ours=True), hub.assign(ours=False)], ignore_index=True) \
        .to_csv(destination / 'hub-pairwise-scores.csv', index=False, float_format='%.6g')



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
    # October WIS (user request 2026-10-10): reference dates in October, horizons 0-3, mean over
    # seeds and reporting draws, then seasons equally. Reported next to the headline; it does not rank.
    details = read('scores-details.csv')
    october = details[details.dimension.eq('month') & details.value.astype(str).str[5:7].eq('10')]
    october = october.groupby(['name', 'history', 'season', 'target', 'scale', 'geography']).wis.mean() \
        .groupby(['name', 'history', 'target', 'scale', 'geography']).mean().rename('october').reset_index()
    summary = summary.merge(october, on=['name', 'history', 'target', 'scale', 'geography'], how='left')
    summary.to_csv(destination / 'headline-rankings.csv', index=False, float_format='%.6g')
    # Compact tables only (user rule, 2026-10-09): seed and draw averages; per-location and
    # per-seed detail stays in each run's cached scores-*.csv.
    config = ['name', 'history', 'deployed']
    raw.groupby([*config, 'season', 'target', 'scale', 'geography'])[METRICS].mean().reset_index() \
        .to_csv(destination / 'headline-season-scores.csv', index=False, float_format='%.6g')
    details = details[details.deployed]
    measures = [c for c in details if c == 'wis' or c.startswith('covered_')]
    details.groupby([*config, 'season', 'target', 'scale', 'geography', 'dimension', 'value'])[measures + ['tasks']] \
        .mean().reset_index().to_csv(destination / 'score-details.csv', index=False, float_format='%.6g')
    relative = read('scores-hub-relative.csv')
    if relative.empty:  # the problem declares no Hub comparison
        pd.DataFrame(columns=[*config, 'hub', 'season', 'target', 'scale', 'geography', 'wis_ratio']) \
            .to_csv(destination / 'hub-relative-scores.csv', index=False)
        pd.DataFrame(columns=[*config, 'hub', 'season', 'target', 'scale', 'relative_wis', 'rank', 'models', 'ours',
                              'model', 'google']).to_csv(destination / 'hub-pairwise-scores.csv', index=False)
    else:
        hub_tables(relative, read('scores-pairwise.csv'), config, destination)
    distribution = read('scores-distribution.csv')
    distribution = distribution[distribution.deployed]
    distribution['geography'] = distribution.location.astype(str).eq('US').map({True: 'US', False: 'states_dc'})
    numeric = [c for c in distribution if c.startswith(('weekly_', 'all_four_', 'four_week_'))]
    distribution.groupby([*config, 'season', 'target', 'geography'])[numeric].mean().reset_index() \
        .to_csv(destination / 'distribution-scores.csv', index=False, float_format='%.6g')
    large = [p.name for p in destination.glob('*.csv') if p.stat().st_size > 1_000_000]
    if large:
        print(f'Warning: ranking tables over 1 MB are not copied to the report: {large}', flush=True)
    from chromantis.problem import Problem
    first_manifest = json.loads((Path(runs[0]['path']) / 'manifest.json').read_text())
    problem = Problem.load(first_manifest['problem'])
    headline_signal = problem.dataset.by_name[problem.headline_target]
    headline_name = headline_signal.hub_target or headline_signal.name
    (destination / 'manifest.json').write_text(json.dumps(dict(
        runs=[str(r['path']) for r in runs], definition=__doc__,
        views={f'{r["name"]} s{r["seed"]}': r['view'] for r in runs},
        support_sha256={'/'.join(k): v for k, v in raw.groupby(['target', 'season', 'scale', 'geography'])
                        .support_sha256.first().items()},
        problem_id=problem.id, problem_sha256=problem.hash,
        selection_metric=f'{problem.headline_geography} {problem.headline_scale} {headline_name} WIS in each recipe\'s forecast_view (headline-rankings.csv, '
                         'deployed rows); lower is better'), indent=2) + '\n')
    window = 'October-May, seasons equal' if problem.fold_kind == 'leave_one_season_out' else f'whole {problem.fold_kind} folds, folds equal'
    print(f'Headline: {window}; states/DC and US separate; lower WIS is better.', flush=True)
    print(summary[summary.deployed & (summary.target == headline_name) & (summary.scale == problem.headline_scale) & (summary.geography == problem.headline_geography)]
          .sort_values('rank')[['name', 'history', 'mean', 'std', 'october', 'count', 'rank']].head(20).to_string(index=False), flush=True)
    return summary
