"""Location-relative WIS: states/DC 80%, US 20%, then targets within equal seasons.

Keep native-unit score sums by location and horizon. Divide model by ensemble
total WIS within each target/season/location, never within individual tasks.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tapestry.models.quantiles import LEVELS
from tapestry.models.objective import US_WEIGHT
from .hubs import KEY, QCOLS, export
from .scoring import match_forecasts

TARGET_WEIGHTS = {
    'wk inc flu hosp': 1., 'wk inc covid hosp': 1., 'wk inc rsv hosp': 1.,
    'wk inc flu prop ed visits': .5, 'wk inc covid prop ed visits': .5, 'wk inc rsv prop ed visits': .5,
}
SCORE_VERSION = 'location-relative-season-first-us20-v1'
SCORE_DEFINITION = ('Per target/season/location: total native-unit model WIS / total ensemble WIS '
                    'on identical tasks, all eligible dates and horizons 0-3. States/DC ratios '
                    'average equally with 80% weight; US ratio gets 20%. Absent geography groups '
                    'renormalize over available groups. Within season: weighted mean of available '
                    'targets, admissions 1 and ED .5. Combined: equal mean of seasons. '
                    'Configurations: mean and SD across seeds. Nonpositive ensemble denominators '
                    'raise an error; missing run support is not silently dropped.')
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
    """Sum model and ensemble scores over identical tasks by location and horizon."""
    model = model.sort_values(KEY).reset_index(drop=True)
    ensemble = ensemble.sort_values(KEY).reset_index(drop=True)
    if not model[KEY].equals(ensemble[KEY]) or not np.allclose(model.observed, ensemble.observed):
        raise ValueError(f"Model and ensemble tasks differ: {case['directory']}")
    table = case_cells(model, ensemble, case)
    grouped = table.groupby(['geography', 'location', 'horizon'])
    totals = grouped[[f'{who}_{m}' for who in ('model', 'ensemble') for m in METRICS]].sum()
    totals.insert(0, 'n', grouped.size())
    return totals.reset_index().assign(target=case['target'], season=case['season'])


def case_cells(model, ensemble, case):
    """The same scores as case_totals, retaining forecast origins for paired blocks."""
    model = model.sort_values(KEY).reset_index(drop=True)
    ensemble = ensemble.sort_values(KEY).reset_index(drop=True)
    if not model[KEY].equals(ensemble[KEY]) or not np.allclose(model.observed, ensemble.observed):
        raise ValueError(f"Model and ensemble tasks differ: {case['directory']}")
    y = model.observed.to_numpy()
    table = pd.concat([quantile_scores(frame[QCOLS].to_numpy(), y).add_prefix(f'{who}_')
                       for who, frame in (('model', model), ('ensemble', ensemble))], axis=1)
    for key in KEY:
        table[key] = model[key].to_numpy()
    table['geography'] = np.where(model.location.eq('US'), 'US', 'states_dc')
    return table.assign(target=case['target'], season=case['season'])


def forecast_cells(run, frozen, stress='natural'):
    """Shared frozen support, reference truth and ensemble for every stress condition."""
    frozen = Path(frozen)
    frames = export(run, stress)
    parts = []
    for case in frozen_cases(frozen):
        units = pd.read_parquet(frozen / case['directory'] / 'units.parquet')
        quantiles = pd.read_parquet(frozen / case['directory'] / 'quantiles.parquet')
        model = match_forecasts(frames[(case['season'], case['target'])], units, case['target'])
        ensemble = match_forecasts(quantiles[quantiles.model == case['ensemble']], units, case['target'])
        parts.append(case_cells(model, ensemble, case))
    return pd.concat(parts, ignore_index=True)


def cells_totals(cells):
    keys = ['target', 'season', 'geography', 'location', 'horizon']
    metrics = [f'{who}_{m}' for who in ('model', 'ensemble') for m in METRICS]
    grouped = cells.groupby(keys)
    result = grouped[metrics].sum()
    result.insert(0, 'n', grouped.size())
    return result.reset_index()


def frozen_cases(frozen):
    frozen = Path(frozen)
    manifest = json.loads((frozen / 'manifest.json').read_text())
    if manifest.get('quantiles') != QCOLS:
        raise ValueError(f'{frozen} holds quantiles {manifest.get("quantiles")}; rebuild frozen support for {QCOLS}')
    return [case for case in manifest['cases'] if case['status'] == 'scored']


def score_run(run, frozen):
    """Write `totals.csv` for one saved season-CV run, B0 or B1.

    A B1 run additionally writes `nowcast-totals.csv` for its two recent weeks,
    which have no Hub ensemble and are scored against preliminary persistence.
    """
    run, frozen = Path(run), Path(frozen)
    totals = cells_totals(forecast_cells(run, frozen))
    if json.loads((run / 'manifest.json').read_text()).get('model') == 'B1':
        stress_parts = []
        for stress in ('natural', 'recent', 'gap', 'outage'):
            cells = forecast_cells(run, frozen, stress)
            cells.to_parquet(run / f'forecast-cells-{stress}.parquet', index=False)
            stress_parts.append(cells_totals(cells).assign(stress=stress))
        pd.concat(stress_parts, ignore_index=True).to_csv(run / 'stress-totals.csv', index=False)
    temporary = run / 'totals.tmp'
    totals.to_csv(temporary, index=False)
    temporary.replace(run / 'totals.csv')
    if json.loads((run / 'manifest.json').read_text()).get('model') == 'B1':
        from .nowcast import score_nowcasts
        score_nowcasts(run)
    return totals


def season_scores(totals):
    """Location-relative scores per target/season, with pooled sums as diagnostics."""
    if 'location' not in totals:
        raise ValueError('totals.csv lacks locations; rerun totals score on saved forecasts before ranking')
    sums = [f'{who}_{m}' for who in ('model', 'ensemble') for m in METRICS]
    keys = [*IDS, 'target', 'season']
    locations = totals.groupby([*keys, 'location'])[['n', *sums]].sum().reset_index()
    if ((locations.ensemble_wis <= 0) | ~np.isfinite(locations.ensemble_wis) |
            ~np.isfinite(locations.model_wis) | (locations.n <= 0)).any():
        raise ValueError('Relative WIS needs positive finite ensemble total WIS and valid model totals at every location')
    rows = []
    for values, part in locations.groupby(keys):
        for geography in ('all', 'states_dc', 'US'):
            group = part if geography == 'all' else part[part.location.eq('US') == (geography == 'US')]
            if group.empty:
                continue
            us = group.location.eq('US').to_numpy()
            weights = np.zeros(len(group))
            if (~us).any():
                weights[~us] = (1 - US_WEIGHT) / (~us).sum()
            if us.any():
                weights[us] = US_WEIGHT / us.sum()
            weights /= weights.sum()
            row = dict(zip(keys, values), geography=geography, locations=len(group),
                       us_weight_used=float(weights[us].sum()), **group[['n', *sums]].sum().to_dict())
            row['wis_ratio'] = float(np.dot(weights, group.model_wis / group.ensemble_wis))
            row['pooled_wis_ratio'] = row['model_wis'] / row['ensemble_wis']
            for who in ('model', 'ensemble'):
                for coverage in COVERAGE:
                    row[f'{who}_coverage_{coverage}'] = float(np.dot(weights, group[f'{who}_covered_{coverage}'] / group.n))
            rows.append(row)
    return pd.DataFrame(rows)


def season_composites(seasons):
    """Targets average within each season; unavailable hub targets get no weight."""
    keys = [*IDS, 'geography', 'season']
    wide = seasons.pivot(index=keys, columns='target', values='wis_ratio').reindex(columns=list(TARGET_WEIGHTS))
    # Common support is shared by all runs, not chosen per model.
    support = seasons[['season', 'target']].drop_duplicates().groupby('season').target.agg(list)
    wide['combined'] = np.nan
    for label, targets in support.items():
        selected = wide.index.get_level_values('season') == label
        weights = pd.Series({target: TARGET_WEIGHTS[target] for target in targets})
        wide.loc[selected, 'combined'] = (wide.loc[selected, targets] * weights).sum(axis=1, min_count=len(targets)) / weights.sum()
    return wide.reset_index()


def run_scores(seasons):
    """Equal mean of season composites; per-target season means are diagnostics."""
    wide = seasons.groupby([*IDS, 'geography', 'target']).wis_ratio.mean().unstack('target')
    wide = wide.reindex(columns=list(TARGET_WEIGHTS))
    composites = season_composites(seasons)
    expected_seasons = seasons.season.nunique()
    wide['combined'] = composites.groupby([*IDS, 'geography']).combined.agg(
        lambda values: values.mean() if len(values) == expected_seasons and values.notna().all() else np.nan)
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
    if 'location' not in totals or totals.location.isna().any():
        raise ValueError('totals.csv lacks locations; rerun totals score on saved forecasts before ranking')
    support_keys = ['target', 'season', 'location', 'horizon', 'n']
    supports = [part[support_keys].sort_values(support_keys).reset_index(drop=True)
                for _, part in totals.groupby(IDS)]
    if any(not part.equals(supports[0]) for part in supports[1:]):
        raise ValueError('Runs have different frozen target/season/location/horizon support; rescore on identical tasks')
    seasons = season_scores(totals)
    scores = run_scores(seasons)
    if not np.isfinite(scores.loc[scores.geography == 'all', 'combined']).all():
        raise ValueError('Incomplete or invalid combined run scores; do not average over missing seeds')
    ranking = configuration_ranking(scores)
    seasons.to_csv(output / 'season_scores.csv', index=False)
    season_composites(seasons).to_csv(output / 'season_composite_scores.csv', index=False)
    scores.to_csv(output / 'run_scores.csv', index=False)
    ranking.to_csv(output / 'configuration_ranking.csv', index=False)
    (output / 'manifest.json').write_text(json.dumps(dict(
        runs=[dict(run, path=str(run['path'])) for run in runs], quantile_levels=LEVELS.tolist(), target_weights=TARGET_WEIGHTS,
        definition=SCORE_DEFINITION, score_version=SCORE_VERSION, us_weight=US_WEIGHT,
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
