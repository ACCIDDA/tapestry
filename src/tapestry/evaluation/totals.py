"""The one score: location-relative WIS ratio to the hub ensemble (user decision 2026-09-22).

Per target and season: each location's total model WIS / total ensemble WIS on
identical frozen tasks; states/DC ratios average equally and share 1 - w, the US
ratio gets w (`us_weight`, default 0.2). Within a season, targets combine with
weights `admissions_weight` (default 1) and `ed_weight` (default .5), i.e.
(2 x admissions + ED) / 9; seasons count equally. The three weights are rank-time
options (`planner rank --us-weight --admissions-weight --ed-weight`, 2026-09-22),
recorded in the ranking manifest and part of the ranking folder name; they are
not scenario or experiment settings. Native-unit WIS sums by location and horizon are kept in
`totals.csv` and in `season_scores.csv` as raw totals only (`model_wis`,
`ensemble_wis`); a pooled total-WIS ratio is not computed or ranked.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .quantiles import LEVELS
from .hubs import CHANNEL, KEY, QCOLS, export

TARGETS = tuple(CHANNEL)  # hub target names in panel channel order
US_SCORE_WEIGHT, ADMISSIONS_WEIGHT, ED_WEIGHT = .2, 1., .5


def target_weights(admissions=ADMISSIONS_WEIGHT, ed=ED_WEIGHT):
    return {target: ed if 'prop ed' in target else admissions for target in TARGETS}


TARGET_WEIGHTS = target_weights()
SCORE_VERSION = 'location-relative-season-first-v2'
SCORE_DEFINITION = ('Per target/season/location: total native-unit model WIS / total ensemble WIS '
                    'on identical tasks, all eligible dates and horizons 0-3. States/DC ratios '
                    'average equally sharing 1 - us_weight; the US ratio gets us_weight (default 0.2). '
                    'Absent geography groups renormalize over available groups. Within season: weighted '
                    'mean of available targets, admissions_weight (default 1) and ed_weight (default .5). '
                    'Combined: equal mean of seasons. '
                    'Configurations: mean and SD across seeds. Nonpositive ensemble denominators '
                    'raise an error; missing run support is not silently dropped. Summed WIS columns '
                    'are raw totals, not a score.')
COVERAGE = (50, 80, 90, 95)
METRICS = ['wis', 'dispersion', 'underprediction', 'overprediction', 'ae_median', *[f'covered_{c}' for c in COVERAGE]]
IDS = ['config_id', 'seed']
COMPONENTS = ('dispersion', 'underprediction', 'overprediction')


def match_forecasts(predictions, units, target):
    """Require one valid forecast for every frozen evaluation task."""
    wide = units[KEY + ['observed']].merge(predictions[KEY + QCOLS], on=KEY, validate='one_to_one')
    if len(wide) != len(units):
        raise ValueError(f'Missing frozen tasks: {target}')
    q = wide[QCOLS].to_numpy()
    if wide.duplicated(KEY).any() or not np.isfinite(q).all() or (q < 0).any() or (np.diff(q, axis=1) < 0).any():
        raise ValueError(f'Invalid forecast units/quantiles: {target}')
    if 'prop' in target and (q > 1).any():
        raise ValueError('ED forecasts must be proportions')
    delta = (pd.to_datetime(wide.target_end_date) - pd.to_datetime(wide.reference_date)).dt.days
    if not wide.horizon.isin(range(4)).all() or not delta.eq(wide.horizon * 7).all():
        raise ValueError('Expected hub horizons 0-3 with exact weekly target dates')
    return wide


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


def forecast_cells(run, frozen):
    """Shared frozen support, reference truth and ensemble."""
    frozen = Path(frozen)
    frames = export(run)
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
    """Write `totals.csv` for one saved season-CV run."""
    run, frozen = Path(run), Path(frozen)
    totals = cells_totals(forecast_cells(run, frozen))
    temporary = run / 'totals.tmp'
    totals.to_csv(temporary, index=False)
    temporary.replace(run / 'totals.csv')
    return totals


def season_scores(totals, us_weight=US_SCORE_WEIGHT):
    """Location-relative scores per target/season; summed WIS columns are raw totals only."""
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
                weights[~us] = (1 - us_weight) / (~us).sum()
            if us.any():
                weights[us] = us_weight / us.sum()
            weights /= weights.sum()
            row = dict(zip(keys, values), geography=geography, locations=len(group),
                       us_weight_used=float(weights[us].sum()), **group[['n', *sums]].sum().to_dict())
            row['wis_ratio'] = float(np.dot(weights, group.model_wis / group.ensemble_wis))
            for who in ('model', 'ensemble'):  # WIS components / ensemble WIS: model's sum to wis_ratio
                for component in COMPONENTS:
                    row[f'{who}_{component}_ratio'] = float(np.dot(weights, group[f'{who}_{component}'] / group.ensemble_wis))
            for who in ('model', 'ensemble'):
                for coverage in COVERAGE:
                    row[f'{who}_coverage_{coverage}'] = float(np.dot(weights, group[f'{who}_covered_{coverage}'] / group.n))
            rows.append(row)
    return pd.DataFrame(rows)


def season_composites(seasons, weights=TARGET_WEIGHTS, value='wis_ratio'):
    """Targets average within each season; unavailable hub targets get no weight."""
    keys = [*IDS, 'geography', 'season']
    wide = seasons.pivot(index=keys, columns='target', values=value).reindex(columns=list(weights))
    # Common support is shared by all runs, not chosen per model.
    support = seasons[['season', 'target']].drop_duplicates().groupby('season').target.agg(list)
    wide['combined'] = np.nan
    for label, targets in support.items():
        selected = wide.index.get_level_values('season') == label
        used = pd.Series({target: weights[target] for target in targets})
        wide.loc[selected, 'combined'] = (wide.loc[selected, targets] * used).sum(axis=1, min_count=len(targets)) / used.sum()
    return wide.reset_index()


def run_scores(seasons, weights=TARGET_WEIGHTS, value='wis_ratio'):
    """Equal mean of season composites; per-target season means are diagnostics."""
    wide = seasons.groupby([*IDS, 'geography', 'target'])[value].mean().unstack('target')
    wide = wide.reindex(columns=list(weights))
    composites = season_composites(seasons, weights, value)
    expected_seasons = seasons.season.nunique()
    wide['combined'] = composites.groupby([*IDS, 'geography']).combined.agg(
        lambda values: values.mean() if len(values) == expected_seasons and values.notna().all() else np.nan)
    return wide.reset_index()


def configuration_ranking(runs):
    """Mean and seed SD of every score per configuration; ordered by combined score over all tasks."""
    scores = [*TARGETS, 'combined']
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


def rank(runs, output, us_weight=US_SCORE_WEIGHT, admissions_weight=ADMISSIONS_WEIGHT, ed_weight=ED_WEIGHT):
    """Rank saved runs [{'config_id', 'seed', 'path'}] into `output` by the location-relative score."""
    weights = target_weights(admissions_weight, ed_weight)
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
    seasons = season_scores(totals, us_weight)
    scores = run_scores(seasons, weights)
    if not np.isfinite(scores.loc[scores.geography == 'all', 'combined']).all():
        raise ValueError('Incomplete or invalid combined run scores; do not average over missing seeds')
    ranking = configuration_ranking(scores)
    seasons.to_csv(output / 'season_scores.csv', index=False)
    season_composites(seasons, weights).to_csv(output / 'season_composite_scores.csv', index=False)
    scores.to_csv(output / 'run_scores.csv', index=False)
    ranking.to_csv(output / 'configuration_ranking.csv', index=False)
    (output / 'manifest.json').write_text(json.dumps(dict(
        runs=[dict(run, path=str(run['path'])) for run in runs], quantile_levels=LEVELS.tolist(), target_weights=weights,
        definition=SCORE_DEFINITION, score_version=SCORE_VERSION, us_weight=us_weight,
        admissions_weight=admissions_weight, ed_weight=ed_weight,
        runs_sha256=hashlib.sha256(json.dumps(sorted(str(run['path']) for run in runs)).encode()).hexdigest()),
        indent=2) + '\n')
    return ranking

