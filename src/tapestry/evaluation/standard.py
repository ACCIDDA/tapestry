"""The standard evaluation report tables (user specification 2026-10-05).

Every forecast run is scored on Wednesday-report inputs (`dataset.episodes`,
`input_mode='reported'`), each pathogen at its own Hub's deadline, with 512 samples.
For each run this module writes, on identical tasks:

1. **Hub-relative WIS** (`hub_relative`): where a Hub ran, each location's total WIS
   divided by the Hub ensemble's on the frozen Hub tasks (`totals.py` rule: states/DC
   averaged equally, 80%; US 20%). On the natural scale for all targets and on the
   log scale, log(x + 1), for admission counts only. ED proportions are never
   transformed (user decision).
2. **Official-style pairwise relative WIS** (`pairwise`), following the CDC FluSight
   2025–26 evaluation: states/DC only (no US, no Puerto Rico); models qualify with
   forecasts for at least 75% of the Hub's tasks; for each pair of qualifying models,
   the ratio of their mean WIS on the tasks both forecast; each model's geometric mean
   of its ratios against every qualifying model (itself included, ratio 1); divided
   by the Hub baseline's. Lower is better, 1 = baseline. One Tapestry run joins the
   Hub pool at a time (each seed separately). Natural scale for all targets; log scale
   for admissions only. The CDC report does not state its zero offset; log(x + 1) is
   our assumption. The same method is applied to the COVID and RSV Hubs.
   Google's models are any Hub model named `Google_*`.
3. **Raw WIS** (`raw_wis`), for all six targets in every scored season, with or
   without a Hub: mean WIS per task, separately for states/DC and the US, natural
   scale (and log for admissions). Truth is the panel's finalized value saved with the
   forecasts. Window: every Saturday from the first to the last reference date of that
   season's Hub admissions tasks for the pathogen; without such a Hub, CDC epiweeks 40-20.
4. **Input fills** (`input_fills`): share of available context inputs that received a
   finalized value because no report was archived by the Hub deadline: the target's
   own history (newest week, all weeks) and all six target histories the forecast
   read, at that Hub's deadline. Reports star every figure with these shares.
"""
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .hubs import CHANNEL, KEY, QCOLS, export
from .totals import (frozen_cases, match_forecasts, quantile_scores, cells_totals, season_scores,
                     TARGETS, TARGET_WEIGHTS)

HUB_OF = {'flu': 'flusight', 'covid': 'covid', 'rsv': 'rsv'}
BASELINES = {'flusight': 'FluSight-baseline', 'covid': 'CovidHub-baseline', 'rsv': 'RSVHub-baseline'}
QUALIFYING_SHARE = .75
EXCLUDED_FROM_RANKING = ('US', '72')  # national and Puerto Rico, as in the CDC evaluation
OUR_MODEL = 'Tapestry'
SCALES = ('natural', 'log')


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
        if case['season'] not in seasons:
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
            ratios = season_scores(totals.assign(config_id='', seed=0))
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
        if case['season'] not in seasons:
            continue
        units = pd.read_parquet(frozen / case['directory'] / 'units.parquet')
        units = units[~units.location.isin(EXCLUDED_FROM_RANKING)].reset_index(drop=True)
        quantiles = pd.read_parquet(frozen / case['directory'] / 'quantiles.parquet')
        quantiles = quantiles[~quantiles.model.str.startswith('Tapestry')]  # frozen copies of old runs
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


def raw_wis(run, frozen):
    """Mean WIS per task for all six targets, states/DC and US, natural (and log for admissions)."""
    frames = export(run)
    windows = hub_windows(frozen)
    rows = []
    for (season, target), frame in frames.items():
        frame = frame[frame.model_original_mask.astype(bool)]
        window = windows.get((pathogen(target), season))
        source = 'Hub admissions date range' if window else 'CDC epiweeks 40-20'
        keep = frame.reference_date.isin(window) if window else frame.reference_date.map(epiweek_window)
        frame = frame[keep]
        if frame.empty:
            raise ValueError(f'No raw-WIS tasks for {target} {season}')
        check_raw_tasks(frame, target, f'{run} {target} {season}')
        support = hashlib.sha256(frame.sort_values(KEY)[[*KEY, 'model_original_truth']]
                                 .to_csv(index=False).encode()).hexdigest()
        for scale in scales_for(target):
            wis = scored(frame[QCOLS].to_numpy(), frame.model_original_truth.to_numpy(), scale)
            us = frame.location.eq('US').to_numpy()
            for geography, part in (('states_dc', ~us), ('US', us)):
                rows.append(dict(target=target, season=season, scale=scale, geography=geography,
                                 mean_wis=float(wis.wis[part].mean()), tasks=int(part.sum()), window=source,
                                 support_sha256=support,
                                 first_reference=frame.reference_date.min(), last_reference=frame.reference_date.max()))
    return pd.DataFrame(rows)


def input_fills(run, frozen):
    """Share of available inputs supplied as finalized values, per season and target channel.

    Counted over the forecasts that are scored: origins whose reference date lies in the
    raw-WIS window of the target's pathogen (its Hub dates, else CDC epiweeks 40-20)."""
    names = list(CHANNEL)
    windows = hub_windows(frozen)
    rows = []
    manifest = json.loads((Path(run) / 'manifest.json').read_text())
    for season in manifest['folds']:
        with np.load(Path(run) / f'eval_{season}' / 'forecasts.npz', allow_pickle=False) as data:
            if 'filled' not in data:
                raise ValueError(f'{run}/eval_{season} was not scored on standard reported inputs; refit it')
            hubs = {h: (data[f'filled_{h}'], data[f'available_{h}']) for h in HUB_OF.values()}
            covariates = {k: data[k] for k in data.files if k.startswith('covariates_')}
            references = [(date.fromisoformat(str(d)) + timedelta(days=7)).isoformat() for d in data['context_end']]

        def scored(pathogen_):
            window = windows.get((pathogen_, season))
            return np.array([r in window if window else epiweek_window(r) for r in references])

        for c, target in enumerate(names):
            keep = scored(pathogen(target))
            filled, available = (a[keep] for a in hubs[HUB_OF[pathogen(target)]])
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


def score(runs, frozen, output):
    """All standard tables for runs [{'config_id', 'seed', 'path'}] into `output`/standard_*.csv."""
    output = Path(output)
    tables = {'hub_relative': hub_relative, 'pairwise': pairwise, 'raw_wis': raw_wis}
    written = {}
    for name, function in tables.items():
        frame = pd.concat([function(run['path'], frozen).assign(config_id=run['config_id'], seed=run['seed'])
                           for run in runs], ignore_index=True)
        if name == 'raw_wis':  # every run on the same tasks and truth, as the Hub path enforces
            supports = frame.groupby(['target', 'season']).support_sha256.nunique()
            if (supports > 1).any() or frame.groupby(['config_id', 'seed']).size().nunique() > 1:
                raise ValueError('Runs differ in raw-WIS tasks or truth: '
                                 f'{supports[supports > 1].index.tolist()}; refit on one panel')
        frame.to_csv(output / f'standard_{name}.csv', index=False)
        written[name] = frame
    fills = pd.concat([input_fills(run['path'], frozen).assign(config_id=run['config_id'], seed=run['seed'])
                       for run in runs], ignore_index=True)
    fills.to_csv(output / 'standard_input_fills.csv', index=False)
    written['input_fills'] = fills
    return written


def star_note(fills):
    """One footnote line: the finalized-value shares behind every figure (means over runs)."""
    columns = [c for c in ('newest_week_share', 'all_weeks_share', 'all_targets_share') if c in fills]
    means = fills.groupby(['season', 'target'], sort=False)[columns].mean()
    parts = []
    for (season, target), row in means.iterrows():
        if row.all_weeks_share > 0 or row.get('all_targets_share', 0) > 0:
            label = target.removeprefix('wk inc ')
            newest = '' if np.isnan(row.newest_week_share) else f'newest week {row.newest_week_share:.0%}, '
            every = '' if np.isnan(row.get('all_targets_share', np.nan)) else f', all six target histories {row.all_targets_share:.0%}'
            parts.append(f'{season} {label}: {newest}own history {row.all_weeks_share:.0%}{every}')
    if not parts:
        return '★ No input needed a finalized value: every scheduled input was archived by its Hub deadline.'
    return ('★ Finalized values were supplied where the schedule assumes availability but no report was archived '
            'by the Hub deadline (share of available inputs): ' + '; '.join(parts) + '.')


def summary_tables(written, labels):
    """Markdown tables for the labelled configurations (seed means), plus Google rows."""
    lines = []
    keep = set(labels)
    rel = written['hub_relative']
    rel = rel[rel.config_id.isin(keep)]
    if not rel.empty:
        lines += ['### Relative to the Hub ensemble', '',
                  'Total WIS divided by the Hub ensemble\'s on identical Hub tasks, per location, then states/DC '
                  'averaged (80%) and the US (20%). Lower is better; 1 = ensemble. Seed means. '
                  'Log = log(count + 1), admissions only; ED is never transformed.', '',
                  '| Config | Target | Season | Scale | States/DC | US | Combined |', '|---|---|---|---|---:|---:|---:|']
        table = rel.groupby(['config_id', 'target', 'season', 'scale', 'geography']).wis_ratio.mean().unstack('geography')
        for (config, target, season, scale), row in table.iterrows():
            lines.append(f'| {labels[config]} | {target.removeprefix("wk inc ")} | {season} | {scale} | '
                         f'{row.get("states_dc", np.nan):.3f} | {row.get("US", np.nan):.3f} | {row.get("all", np.nan):.3f} |')
        lines.append('')
    pair = written['pairwise']
    if not pair.empty:
        lines += ['### Official-style ranking among Hub models', '',
                  'Pairwise relative WIS (CDC FluSight method: states/DC, models with at least 75% of tasks, '
                  'geometric mean of pairwise mean-WIS ratios, divided by the Hub baseline). Lower is better; '
                  '1 = baseline. Rank 1 is best. Our value is the seed mean; each seed joins the Hub pool alone. '
                  'Google rows are its own submissions (mean over our seed pools); a Google model with fewer '
                  'than 75% of the Hub tasks is listed but, as in the CDC evaluation, not ranked.', '',
                  '| Target | Season | Scale | Model | Relative WIS | Rank (seeds) | Models |', '|---|---|---|---|---:|---|---:|']
        ours = pair[pair.ours & pair.config_id.isin(keep)]
        google = pair[pair.google & pair.config_id.isin(keep)]
        for (target, season, scale), part in pair.groupby(['target', 'season', 'scale']):
            rows = []
            for config in labels:
                mine = ours[(ours.config_id == config) & (ours.target == target) & (ours.season == season) & (ours.scale == scale)]
                if len(mine):
                    rows.append((labels[config], mine.relative_wis.mean(), ', '.join(map(str, mine['rank'])), int(mine.models.iloc[0])))
            g = google[(google.target == target) & (google.season == season) & (google.scale == scale)]
            for model, gpart in g.groupby('model'):
                if gpart.qualifies.astype(bool).all():
                    rank = ', '.join(sorted({str(int(r)) for r in gpart['rank']}))
                else:
                    rank = f'did not qualify ({gpart.tasks.iloc[0] / gpart.hub_tasks.iloc[0]:.1%} of tasks)'
                rows.append((model, gpart.relative_wis.mean(), rank, int(gpart.models.iloc[0])))
            for name, value, rank, count in rows:
                shown = '' if np.isnan(value) else f'{value:.3f}'
                lines.append(f'| {target.removeprefix("wk inc ")} | {season} | {scale} | {name} | {shown} | {rank} | {count} |')
        lines.append('')
    raw = written['raw_wis']
    raw = raw[raw.config_id.isin(keep)]
    if not raw.empty:
        lines += ['### Raw WIS on every target', '',
                  'Mean WIS per task in native units (admissions counts; ED as a proportion) and, for admissions, '
                  'on log(count + 1). Seed means. Lower is better; compare configurations within a row group only. '
                  'Truth: finalized panel values. Window: every Saturday between the season\'s first and last '
                  'Hub admissions reference dates for the pathogen, otherwise CDC epiweeks 40-20.', '',
                  '| Config | Target | Season | Scale | States/DC | US | Window |', '|---|---|---|---|---:|---:|---|']
        table = raw.groupby(['config_id', 'target', 'season', 'scale', 'window', 'geography']).mean_wis.mean().unstack('geography')
        for (config, target, season, scale, window), row in table.iterrows():
            lines.append(f'| {labels[config]} | {target.removeprefix("wk inc ")} | {season} | {scale} | '
                         f'{row.get("states_dc", np.nan):.4g} | {row.get("US", np.nan):.4g} | {window} |')
        lines.append('')
    return '\n'.join(lines)
