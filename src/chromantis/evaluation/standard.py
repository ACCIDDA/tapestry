"""The scorer: WIS, Hub-relative WIS, pairwise relative WIS and raw WIS (user specification 2026-10-05).

Every forecast run is scored on Wednesday-report inputs (`dataset.episodes`,
`input_mode='reported'`), each pathogen at its own Hub's deadline, with 512 samples.
For each run this module writes, on identical tasks:

1. **Hub-relative WIS** (`hub_relative`): where a Hub ran, each location's total WIS
   divided by the Hub ensemble's on the frozen Hub tasks; location ratios averaged
   equally within states/DC and reported separately for the US (the 80% states / 20% US
   combination was dropped on 2026-10-08, user decision). On the natural scale for all
   targets and on the log scale, log(x + 1), for admission counts only. ED proportions
   are never transformed (user decision).
2. **Official-style pairwise relative WIS** (`pairwise`), following the CDC FluSight
   2025–26 evaluation: states/DC only (no US, no Puerto Rico); models qualify with
   forecasts for at least 75% of the Hub's tasks; for each pair of qualifying models,
   the ratio of their mean WIS on the tasks both forecast; each model's geometric mean
   of its ratios against every qualifying model (itself included, ratio 1); divided
   by the Hub baseline's. Lower is better, 1 = baseline. One Chromantis run joins the
   Hub pool at a time (each seed separately). Natural scale for all targets; log scale
   for admissions only. The CDC report does not state its zero offset; log(x + 1) is
   our assumption. The same method is applied to the COVID and RSV Hubs.
   Google's models are any Hub model named `Google_*`.
3. **Raw WIS** (`raw_wis`), for every problem target in every scored season, with or
   without a Hub, on Hub horizons 0-3 also for longer forecasts
   (`raw_score_details` has every horizon): mean WIS per task, separately for states/DC and the US, natural
   scale (and log for admissions). Truth is the panel's finalized value saved with the
   forecasts. Window: every Saturday from the first to the last reference date of that
   season's Hub admissions tasks for the pathogen; without such a Hub, CDC epiweeks 40-20.
   Problems whose folds are not CDC seasons (periods, rolling origins) score every
   reference date of the fold instead of October-May.
4. **Input fills** (`input_fills`): share of available context inputs that received a
   finalized value because no report was archived by the Hub deadline: the target's
   own history (newest week, all weeks) and every input history the forecast
   read, at that Hub's deadline. Reports star every figure with these shares.
"""
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .hubs import KEY, QCOLS, export
from .quantiles import LEVELS

HUB_OF = {'flu': 'flusight', 'covid': 'covid', 'rsv': 'rsv'}
BASELINES = {'flusight': 'FluSight-baseline', 'covid': 'CovidHub-baseline', 'rsv': 'RSVHub-baseline'}
QUALIFYING_SHARE = .75
EXCLUDED_FROM_RANKING = ('US', '72')  # national and Puerto Rico, as in the CDC evaluation
OUR_MODEL = 'Chromantis'
SCALES = ('natural', 'log')
COVERAGE = (50, 80, 90, 95)
METRICS = ['wis', 'dispersion', 'underprediction', 'overprediction', 'ae_median', *[f'covered_{c}' for c in COVERAGE]]


def frozen_cases(frozen):
    """Scored (season, target) cases of the frozen Hub support (tasks, truth, Hub quantiles)."""
    frozen = Path(frozen)
    manifest = json.loads((frozen / 'manifest.json').read_text())
    if manifest.get('quantiles') != QCOLS:
        raise ValueError(f'{frozen} holds quantiles {manifest.get("quantiles")}; rebuild frozen support for {QCOLS}')
    return [case for case in manifest['cases'] if case['status'] == 'scored']


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


def cells_totals(cells):
    keys = ['target', 'season', 'geography', 'location', 'horizon']
    metrics = [f'{who}_{m}' for who in ('model', 'ensemble') for m in METRICS]
    grouped = cells.groupby(keys)
    result = grouped[metrics].sum()
    result.insert(0, 'n', grouped.size())
    return result.reset_index()


def season_scores(totals):
    """Location-relative WIS ratio per target/season: states/DC location ratios averaged equally; US alone."""
    keys = ['target', 'season']
    locations = totals.groupby([*keys, 'location'])[['n', 'model_wis', 'ensemble_wis']].sum().reset_index()
    if ((locations.ensemble_wis <= 0) | ~np.isfinite(locations.ensemble_wis) |
            ~np.isfinite(locations.model_wis) | (locations.n <= 0)).any():
        raise ValueError('Relative WIS needs positive finite ensemble total WIS and valid model totals at every location')
    rows = []
    for values, part in locations.groupby(keys):
        for geography in ('states_dc', 'US'):
            group = part[part.location.eq('US') == (geography == 'US')]
            if len(group):
                rows.append(dict(zip(keys, values), geography=geography, n=int(group.n.sum()),
                                 wis_ratio=float((group.model_wis / group.ensemble_wis).mean())))
    return pd.DataFrame(rows)


def pathogen(target):
    return target.split()[2]


def admissions(target):
    return 'prop ed' not in target


def scales_for(target):
    """ED proportions are always untransformed; admission counts get both scales."""
    return SCALES if admissions(target) else ('natural',)


def transform(values, scale):
    values = np.asarray(values, dtype=float)
    if scale == 'log' and not (values >= 0).all():
        raise ValueError('log(x + 1) scoring needs finite nonnegative values')
    return np.log1p(values) if scale == 'log' else values


def scored(q, y, scale):
    return quantile_scores(transform(q, scale), transform(y, scale))


def hub_relative(run, frozen):
    """Location-relative WIS / Hub ensemble per target, season and scale (states/DC, US, all)."""
    frozen = Path(frozen)
    frames = export(run)
    seasons = set(json.loads((Path(run) / 'manifest.json').read_text())['folds'])
    rows = []
    for case in frozen_cases(frozen):
        if case['season'] not in seasons or (case['season'],case['target']) not in frames:
            continue
        units = pd.read_parquet(frozen / case['directory'] / 'units.parquet')
        quantiles = pd.read_parquet(frozen / case['directory'] / 'quantiles.parquet')
        model = match_forecasts(frames[(case['season'], case['target'])], units, case['target'])
        ensemble = match_forecasts(quantiles[quantiles.model == case['ensemble']], units, case['target'])
        model, ensemble = (f.sort_values(KEY).reset_index(drop=True) for f in (model, ensemble))
        if not model[KEY].equals(ensemble[KEY]):
            raise ValueError(f'Model and ensemble tasks differ: {case["directory"]}')
        for scale in scales_for(case['target']):
            y = model.observed.to_numpy()
            cells = pd.concat([scored(f[QCOLS].to_numpy(), y, scale).add_prefix(f'{who}_')
                               for who, f in (('model', model), ('ensemble', ensemble))], axis=1)
            for key in KEY:
                cells[key] = model[key].to_numpy()
            cells['geography'] = np.where(model.location.eq('US'), 'US', 'states_dc')
            totals = cells_totals(cells.assign(target=case['target'], season=case['season']))
            ratios = season_scores(totals)
            for row in ratios.itertuples():
                rows.append(dict(hub=case['hub'], target=case['target'], season=case['season'], scale=scale,
                                 geography=row.geography, ensemble=case['ensemble'], wis_ratio=row.wis_ratio,
                                 tasks=int(row.n)))
    return pd.DataFrame(rows)


def relative_wis(scores, baseline):
    """CDC pairwise relative WIS from per-task scores [task, model] (NaN = no forecast).

    Each pair compares mean WIS on the tasks both models forecast; a model's theta is
    the geometric mean of its ratios against every model (itself included, ratio 1);
    the result is theta divided by the baseline's theta."""
    models = list(scores.columns)
    values = scores.to_numpy()
    present = np.isfinite(values)
    theta = {}
    for i, a in enumerate(models):
        ratios = []
        for j, b in enumerate(models):
            shared = present[:, i] & present[:, j]
            if not shared.any():
                raise ValueError(f'{a} and {b} share no tasks')
            ratios.append(values[shared, i].mean() / values[shared, j].mean())
        theta[a] = float(np.exp(np.mean(np.log(ratios))))
    if baseline not in theta:
        raise ValueError(f'Baseline {baseline} does not qualify')
    return pd.Series({m: theta[m] / theta[baseline] for m in models})


def pairwise(run, frozen):
    """Official-style relative WIS of this run among the qualifying Hub models, with ranks."""
    frozen = Path(frozen)
    frames = export(run)
    seasons = set(json.loads((Path(run) / 'manifest.json').read_text())['folds'])
    rows = []
    for case in frozen_cases(frozen):
        if case['season'] not in seasons or (case['season'],case['target']) not in frames:
            continue
        units = pd.read_parquet(frozen / case['directory'] / 'units.parquet')
        units = units[~units.location.isin(EXCLUDED_FROM_RANKING)].reset_index(drop=True)
        quantiles = pd.read_parquet(frozen / case['directory'] / 'quantiles.parquet')
        quantiles = quantiles[~quantiles.model.str.startswith('Chromantis')]  # frozen copies of old runs
        ours = match_forecasts(frames[(case['season'], case['target'])], units, case['target']).assign(model=OUR_MODEL)
        pool = pd.concat([quantiles[quantiles.model.ne(OUR_MODEL)], ours[[*KEY, *QCOLS, 'model']]], ignore_index=True)
        pool = units[KEY + ['observed']].merge(pool[KEY + QCOLS + ['model']], on=KEY, validate='one_to_many')
        if pool.duplicated([*KEY, 'model']).any():
            raise ValueError(f'Duplicate model tasks in {case["directory"]}')
        counts = pool.groupby('model').size()
        qualifying = counts[counts >= QUALIFYING_SHARE * len(units)].index
        for model in counts.index.difference(qualifying):  # published but not rankable under the 75% rule
            if model.startswith('Google_') or model == OUR_MODEL:
                for scale in scales_for(case['target']):
                    rows.append(dict(hub=case['hub'], target=case['target'], season=case['season'], scale=scale,
                                     model=model, relative_wis=np.nan, rank=np.nan, models=len(qualifying),
                                     tasks=int(counts[model]), hub_tasks=len(units), qualifies=False,
                                     google=model.startswith('Google_'), ours=model == OUR_MODEL))
        pool = pool[pool.model.isin(qualifying)]
        q = pool[QCOLS].to_numpy()
        if not np.isfinite(q).all() or (np.diff(q, axis=1) < 0).any():
            raise ValueError(f'Invalid quantiles in {case["directory"]}')
        hub = HUB_OF[pathogen(case['target'])]
        for scale in scales_for(case['target']):
            wis = scored(q, pool.observed.to_numpy(), scale)['wis'].to_numpy()
            table = pool[KEY].assign(model=pool.model.to_numpy(), wis=wis).pivot_table(
                index=KEY, columns='model', values='wis', aggfunc='first')
            relative = relative_wis(table, BASELINES[hub]).sort_values()
            ranks = relative.rank(method='min')
            for model, value in relative.items():
                rows.append(dict(hub=case['hub'], target=case['target'], season=case['season'], scale=scale,
                                 model=model, relative_wis=float(value), rank=int(ranks[model]),
                                 models=len(relative), tasks=int(table[model].notna().sum()),
                                 hub_tasks=len(units), qualifies=True, google=model.startswith('Google_'),
                                 ours=model == OUR_MODEL))
    return pd.DataFrame(rows)


def epiweek(day):
    """CDC (MMWR) week number of the Sunday-Saturday week ending on Saturday `day`."""
    sunday = date.fromisoformat(str(day)[:10]) - timedelta(days=6)
    first = lambda year: date(year, 1, 4) - timedelta(days=(date(year, 1, 4).weekday() + 1) % 7)
    year = sunday.year + 1 if sunday >= first(sunday.year + 1) else sunday.year if sunday >= first(sunday.year) else sunday.year - 1
    return (sunday - first(year)).days // 7 + 1


def epiweek_window(day):
    """True when Saturday `day` ends CDC epiweek 40-53 or 1-20 (the fallback season window)."""
    week = epiweek(day)
    return week >= 40 or week <= 20


def hub_windows(frozen):
    """{(pathogen, season): every Saturday from the first to the last Hub admissions reference date}.

    The frozen Hub tasks are a subset (complete ensemble and old B0 overlap; e.g. RSV
    2025-26 jumps from 2025-09-27 to 2025-11-22). Raw WIS scores outside the Hub, so it
    keeps only the season's first and last Hub dates and every Saturday in between."""
    frozen = Path(frozen)
    windows = {}
    for case in frozen_cases(frozen):
        if admissions(case['target']):
            units = pd.read_parquet(frozen / case['directory'] / 'units.parquet', columns=['reference_date'])
            first, last = (date.fromisoformat(d) for d in (units.reference_date.astype(str).min(),
                                                             units.reference_date.astype(str).max()))
            windows[(pathogen(case['target']), case['season'])] = {
                (first + timedelta(weeks=k)).isoformat() for k in range((last - first).days // 7 + 1)}
    return windows


def check_raw_tasks(frame, target, where):
    """Raw WIS has no frozen Hub check: require one valid forecast and finite truth per task."""
    q, y = frame[QCOLS].to_numpy(float), frame.model_original_truth.to_numpy(float)
    if frame.duplicated(KEY).any():
        raise ValueError(f'Duplicate raw-WIS tasks: {where}')
    if not np.isfinite(q).all() or (q < 0).any() or (np.diff(q, axis=1) < 0).any():
        raise ValueError(f'Invalid forecast quantiles (non-finite, negative or decreasing): {where}')
    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError(f'Invalid finalized truth: {where}')
    if not admissions(target) and ((q > 1).any() or (y > 1).any()):
        raise ValueError(f'ED values must be proportions: {where}')


def scored_window(run):
    """(description, filter on reference dates): October-May for CDC-season folds, else the whole fold."""
    from chromantis.problem import Problem
    problem = Problem.load(json.loads((Path(run) / 'manifest.json').read_text())['problem'])
    if problem.fold_kind == 'leave_one_season_out':
        return ('Every October-May reference date, independent of Hub forecast support',
                lambda r: r.str[5:7].isin(('10','11','12','01','02','03','04','05')))
    return f'Every reference date of the {problem.fold_kind} fold', lambda r: r.notna()


def raw_wis(run, frozen, horizon=None):
    """Mean WIS per task, its three components and interval coverage, for every predicted target;
    states/DC and US, natural (and log for admissions). Coverage is identical on both scales."""
    frames = export(run)
    source, window = scored_window(run)
    rows = []
    for (season, target), frame in frames.items():
        if horizon is not None:
            if horizon not in range(4):
                raise ValueError('Expected horizon 0–3')
            frame = frame[frame.horizon.eq(horizon)]
        else:  # the headline stays on Hub horizons 0-3 for models forecasting further ahead
            frame = frame[frame.horizon.lt(4)]
        frame = frame[frame.model_original_mask.astype(bool)]
        frame = frame[window(frame.reference_date)]
        if frame.empty:
            raise ValueError(f'No raw-WIS tasks for {target} {season}')
        check_raw_tasks(frame, target, f'{run} {target} {season}')
        support = hashlib.sha256(frame.sort_values(KEY)[[*KEY, 'model_original_truth']]
                                 .to_csv(index=False).encode()).hexdigest()
        for scale in scales_for(target):
            wis = scored(frame[QCOLS].to_numpy(), frame.model_original_truth.to_numpy(), scale)
            us = frame.location.eq('US').to_numpy()
            for geography, part in (('states_dc', ~us), ('US', us)):
                # WIS = dispersion + underprediction + overprediction (per-task means, same tasks).
                parts = {f'mean_{c}': float(wis[c][part].mean()) for c in ('dispersion', 'underprediction', 'overprediction')}
                parts.update({f'covered_{c}': float(wis[f'covered_{c}'][part].mean()) for c in COVERAGE})
                rows.append(dict(target=target, season=season, scale=scale, geography=geography,
                                 mean_wis=float(wis.wis[part].mean()), tasks=int(part.sum()), window=source,
                                 support_sha256=support, **parts,
                                 first_reference=frame.reference_date.min(), last_reference=frame.reference_date.max()))
    return pd.DataFrame(rows)


def raw_score_details(run):
    """October–May (or whole-fold, see `scored_window`) WIS/coverage by month, horizon and individual location; no US mixture.

    The horizon breakdown covers every forecast week; month and location use horizons 0-3,
    as the headline does, so longer forecasts stay comparable with four-week ones."""
    rows=[]
    _,window=scored_window(run)
    for (season,target),frame in export(run).items():
        frame=frame[frame.model_original_mask.astype(bool) & window(frame.reference_date)].copy()
        check_raw_tasks(frame,target,str(run))
        for scale in scales_for(target):
            scores=scored(frame[QCOLS].to_numpy(),frame.model_original_truth.to_numpy(),scale).reset_index(drop=True)
            scores['geography']=np.where(frame.location.eq('US'),'US','states_dc')
            scores['month']=frame.reference_date.str[:7].to_numpy()
            scores['horizon']=frame.horizon.to_numpy()
            scores['location']=frame.location.to_numpy()
            metrics=[c for c in scores if c=='wis' or c.startswith('covered_')]
            for dimension in ('month','horizon','location'):
                part=scores if dimension=='horizon' else scores[scores.horizon<4]
                grouped=part.groupby(['geography',dimension])
                table=grouped[metrics].mean().reset_index()
                table['tasks']=grouped.size().to_numpy()
                table=table.rename(columns={dimension:'value'})
                table['value']=table.value.astype(str)
                rows.append(table.assign(season=season,target=target,scale=scale,dimension=dimension))
    return pd.concat(rows,ignore_index=True)


def input_fills(run, frozen):
    """Share of available inputs supplied as finalized values, per season and target channel.

    Counted over the forecasts that are scored: origins whose reference date lies in the
    raw-WIS window of the target's pathogen (its Hub dates, else CDC epiweeks 40-20)."""
    windows = hub_windows(frozen)
    rows = []
    manifest = json.loads((Path(run) / 'manifest.json').read_text())
    from chromantis.problem import Problem
    problem = Problem.load(manifest['problem'])
    for season in manifest['folds']:
        with np.load(Path(run) / f'eval_{season}' / 'forecasts.npz', allow_pickle=False) as data:
            if 'filled' not in data:
                raise ValueError(f'{run}/eval_{season} was not scored on standard reported inputs; refit it')
            hubs = {h: (data[f'filled_{h}'], data[f'available_{h}'])
                    for h in set(problem.target_hubs) if h and f'filled_{h}' in data}
            covariates = {k: data[k] for k in data.files if k.startswith('covariates_')}
            references = [(date.fromisoformat(str(d)) + timedelta(days=7)).isoformat() for d in data['context_end']]
            input_names = list(map(str, data['input_names']))

        def scored(pathogen_):
            window = windows.get((pathogen_, season))
            return np.array([r in window if window else epiweek_window(r) for r in references])

        for signal in problem.target_signals:
            target = signal.hub_target
            if not target or signal.name not in input_names:
                continue
            c = input_names.index(signal.name)
            keep = scored(signal.group)
            filled, available = (a[keep] for a in hubs[signal.hub])
            newest = available[:, -1, c].sum()
            rows.append(dict(season=season, target=target,
                             newest_week_share=float(filled[:, -1, c].sum() / newest) if newest else np.nan,
                             all_weeks_share=float(filled[:, :, c].sum() / max(available[:, :, c].sum(), 1)),
                             all_targets_share=float(filled.sum() / max(available.sum(), 1))))
        for key, value in covariates.items():
            if key.startswith('covariates_filled_'):
                hub = key.removeprefix('covariates_filled_')
                keep = scored({v: k for k, v in HUB_OF.items()}[hub])
                total = covariates[f'covariates_available_{hub}'][keep].sum()
                rows.append(dict(season=season, target=f'covariates ({hub} deadline)', newest_week_share=np.nan,
                                 all_weeks_share=float(value[keep].sum() / total) if total else np.nan))
    return pd.DataFrame(rows)


def star_note(fills):
    """One footnote line: the finalized-value shares behind every figure (means over runs)."""
    columns = [c for c in ('newest_week_share', 'all_weeks_share', 'all_targets_share') if c in fills]
    means = fills.groupby(['season', 'target'], sort=False)[columns].mean()
    parts = []
    for (season, target), row in means.iterrows():
        if row.all_weeks_share > 0 or row.get('all_targets_share', 0) > 0:
            label = target.removeprefix('wk inc ')
            newest = '' if np.isnan(row.newest_week_share) else f'newest week {row.newest_week_share:.0%}, '
            every = '' if np.isnan(row.get('all_targets_share', np.nan)) else f', all model input histories {row.all_targets_share:.0%}'
            parts.append(f'{season} {label}: {newest}own history {row.all_weeks_share:.0%}{every}')
    if not parts:
        return '★ No input needed a finalized value: every scheduled input was archived by its Hub deadline.'
    return ('★ Finalized values were supplied where the schedule assumes availability but no report was archived '
            'by the Hub deadline (share of available inputs): ' + '; '.join(parts) + '.')
