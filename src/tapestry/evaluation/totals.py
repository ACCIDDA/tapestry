"""Total-WIS scores against the official ensemble on frozen hub tasks, and rankings.

`score` writes one run's `totals.csv`: sums of WIS, its components, absolute
median error, and central-interval hits for the model and the ensemble over the
identical frozen tasks, by target, season, geography, and horizon. Sums add
exactly, so rankings take ratios of total WIS rather than averaging per-task or
per-location ratios, whose mean has poor statistical properties.

Per run, a target's score is the mean over its seasons of
total model WIS / total ensemble WIS within each season (each season counts
equally). The combined score weights each admissions target twice and each ED
target once. Configurations average run scores across seeds.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tapestry.models.quantiles import LEVELS
from .hubs import KEY, QCOLS, export_b0
from .scoring import match_forecasts

TARGET_WEIGHTS = {
    'wk inc flu hosp': 2, 'wk inc covid hosp': 2, 'wk inc rsv hosp': 2,
    'wk inc flu prop ed visits': 1, 'wk inc covid prop ed visits': 1, 'wk inc rsv prop ed visits': 1,
}
COVERAGE = (50, 80, 90, 95)
METRICS = ['wis', 'dispersion', 'underprediction', 'overprediction', 'ae_median', *[f'covered_{c}' for c in COVERAGE]]
IDS = ['config_id', 'seed']


def quantile_scores(q, y, levels=LEVELS):
    """Per-task WIS components for quantiles q [tasks, levels], columns in level order.

    WIS = (|y - median| / 2 + sum_k alpha_k / 2 * IS_k) / (K + 1/2) over the K
    central intervals, which equals twice the mean pinball loss over all levels.
    """
    levels = np.asarray(levels, dtype=float)
    q, y = np.asarray(q, dtype=float), np.asarray(y, dtype=float)
    k = len(levels) // 2
    if len(levels) % 2 == 0 or not np.isclose(levels[k], .5) or not np.allclose(levels + levels[::-1], 1):
        raise ValueError('WIS needs a symmetric quantile grid with a median')
    alpha = 2 * levels[:k]
    lower, upper, median = q[:, :k], q[:, ::-1][:, :k], q[:, k]
    denominator = k + .5
    # (alpha / 2) * (2 / alpha) * miss leaves the miss itself.
    result = dict(dispersion=(alpha / 2 * (upper - lower)).sum(1) / denominator,
                  underprediction=(np.maximum(y[:, None] - upper, 0).sum(1) + np.maximum(y - median, 0) / 2) / denominator,
                  overprediction=(np.maximum(lower - y[:, None], 0).sum(1) + np.maximum(median - y, 0) / 2) / denominator,
                  ae_median=np.abs(y - median))
    result['wis'] = result['dispersion'] + result['underprediction'] + result['overprediction']
    for coverage in COVERAGE:
        i = int(np.flatnonzero(np.isclose(levels, (1 - coverage / 100) / 2))[0])
        result[f'covered_{coverage}'] = ((q[:, i] <= y) & (y <= q[:, -1 - i])).astype(float)
    return pd.DataFrame(result)


def case_totals(model, ensemble, case):
    """Sum model and ensemble scores over identical tasks by geography and horizon."""
    model = model.sort_values(KEY).reset_index(drop=True)
    ensemble = ensemble.sort_values(KEY).reset_index(drop=True)
    if not model[KEY].equals(ensemble[KEY]) or not np.allclose(model.observed, ensemble.observed):
        raise ValueError(f"Model and ensemble tasks differ: {case['directory']}")
    y = model.observed.to_numpy()
    table = pd.concat([quantile_scores(frame[QCOLS].to_numpy(), y).add_prefix(f'{who}_')
                       for who, frame in (('model', model), ('ensemble', ensemble))], axis=1)
    table['geography'] = np.where(model.location.eq('US'), 'US', 'states_dc')
    table['horizon'] = model.horizon.to_numpy()
    grouped = table.groupby(['geography', 'horizon'])
    totals = grouped.sum()
    totals.insert(0, 'n', grouped.size())
    return totals.reset_index().assign(target=case['target'], season=case['season'])


def frozen_cases(frozen):
    frozen = Path(frozen)
    manifest = json.loads((frozen / 'manifest.json').read_text())
    if manifest.get('quantiles') != QCOLS:
        raise ValueError(f'{frozen} holds quantiles {manifest.get("quantiles")}; rebuild frozen support for {QCOLS}')
    return [case for case in manifest['cases'] if case['status'] == 'scored']


def score_run(run, frozen):
    """Write `totals.csv` for one saved season-CV run."""
    run, frozen = Path(run), Path(frozen)
    frames = export_b0(run)
    parts = []
    for case in frozen_cases(frozen):
        units = pd.read_parquet(frozen / case['directory'] / 'units.parquet')
        quantiles = pd.read_parquet(frozen / case['directory'] / 'quantiles.parquet')
        model = match_forecasts(frames[(case['season'], case['target'])], units, case['target'])
        ensemble = match_forecasts(quantiles[quantiles.model == case['ensemble']], units, case['target'])
        parts.append(case_totals(model, ensemble, case))
    columns = ['target', 'season', 'geography', 'horizon', 'n', *[f'{who}_{m}' for who in ('model', 'ensemble') for m in METRICS]]
    totals = pd.concat(parts, ignore_index=True)[columns]
    temporary = run / 'totals.tmp'
    totals.to_csv(temporary, index=False)
    temporary.replace(run / 'totals.csv')
    return totals


def season_scores(totals):
    """Total-WIS ratio and coverage per run, target, and season, for all tasks, states/DC, and US."""
    sums = [f'{who}_{m}' for who in ('model', 'ensemble') for m in METRICS]
    tables = []
    for geography in ('all', 'states_dc', 'US'):
        part = totals if geography == 'all' else totals[totals.geography == geography]
        table = part.groupby([*IDS, 'target', 'season'])[['n', *sums]].sum().reset_index()
        table.insert(len(IDS), 'geography', geography)
        tables.append(table)
    table = pd.concat(tables, ignore_index=True)
    table['wis_ratio'] = table.model_wis / table.ensemble_wis
    for who in ('model', 'ensemble'):
        for coverage in COVERAGE:
            table[f'{who}_coverage_{coverage}'] = table[f'{who}_covered_{coverage}'] / table.n
    return table


def run_scores(seasons):
    """Season-equal mean ratio per target, and the weighted combined score, per run and geography."""
    wide = seasons.groupby([*IDS, 'geography', 'target']).wis_ratio.mean().unstack('target')
    wide = wide.reindex(columns=list(TARGET_WEIGHTS))
    weights = pd.Series(TARGET_WEIGHTS)
    # A run missing any target gets no combined score rather than a partial one.
    wide['combined'] = (wide[list(TARGET_WEIGHTS)] * weights).sum(axis=1, min_count=len(weights)) / weights.sum()
    wide.loc[wide[list(TARGET_WEIGHTS)].isna().any(axis=1), 'combined'] = np.nan
    return wide.reset_index()


def configuration_ranking(runs):
    """Mean and seed SD of every score per configuration; ordered by combined score over all tasks."""
    scores = [*TARGET_WEIGHTS, 'combined']
    tables = {}
    for geography, part in runs.groupby('geography'):
        grouped = part.groupby('config_id')[scores]
        table = grouped.mean().add_suffix('_mean').join(grouped.std().add_suffix('_sd'))
        tables[geography] = table
    ranking = tables['all'].join(runs[runs.geography == 'all'].groupby('config_id').seed.count().rename('seeds'))
    for geography in ('states_dc', 'US'):
        if geography in tables:
            ranking[f'{geography}_combined_mean'] = tables[geography]['combined_mean']
    ranking = ranking.sort_values('combined_mean')
    ranking.insert(0, 'rank', ranking.combined_mean.rank(method='min'))
    return ranking.reset_index()


def rank(runs, output):
    """Rank saved runs [{'config_id', 'seed', 'path'}] into `output`."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    totals = pd.concat([pd.read_csv(Path(run['path']) / 'totals.csv').assign(config_id=run['config_id'], seed=run['seed'])
                        for run in runs], ignore_index=True)
    seasons = season_scores(totals)
    scores = run_scores(seasons)
    ranking = configuration_ranking(scores)
    seasons.to_csv(output / 'season_scores.csv', index=False)
    scores.to_csv(output / 'run_scores.csv', index=False)
    ranking.to_csv(output / 'configuration_ranking.csv', index=False)
    (output / 'manifest.json').write_text(json.dumps(dict(
        runs=[dict(run, path=str(run['path'])) for run in runs], quantile_levels=LEVELS.tolist(), target_weights=TARGET_WEIGHTS,
        definition=('Per target and season: total model WIS / total ensemble WIS over identical frozen tasks '
                    '(all locations including US, all reference dates, horizons 0–3). Per target: mean over its '
                    'seasons. Combined: admissions targets weight 2, ED targets weight 1. Configurations: mean of run '
                    'scores across seeds. states_dc and US repeat the same computation on those tasks only.'),
        runs_sha256=hashlib.sha256(json.dumps(sorted(str(run['path']) for run in runs)).encode()).hexdigest()),
        indent=2) + '\n')
    return ranking


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    score = sub.add_parser('score', help='Write totals.csv for one saved season-CV run')
    score.add_argument('--run', type=Path, required=True)
    score.add_argument('--frozen', type=Path, required=True)
    ranked = sub.add_parser('rank', help='Rank saved runs that already have totals.csv')
    ranked.add_argument('--runs', type=Path, nargs='+', required=True)
    ranked.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == 'score':
        totals = score_run(args.run, args.frozen)
        print(json.dumps(dict(run=str(args.run), rows=len(totals))), flush=True)
    else:
        from .configurations import identify
        runs = [dict(config_id=record['config_id'], seed=record['seed'], path=path)
                for path in args.runs for record in [identify(path)]]
        print(rank(runs, args.output).head(20).to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
