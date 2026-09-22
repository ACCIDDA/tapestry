"""Build `data/processed/finalized.npz` and `data/processed/vintaged.npz`.

Replaces `model_data/finalized.py` + `model_data/wednesday.py` + `model_data/b2.py`
(three separate builders, three covariate conventions) with one module producing
exactly two arrays, restricted to Wednesday-issuance/Saturday-target episodes,
state+national geography, and the two wastewater indices already promoted into
production (`wval_like`, `pct_rank`); see docs/design/restructure-2026-unified.md §3.

`COVARIATE_GROUPS` is the single source of truth for covariate column order; a
`Scenario.covariate_set` string (`model.scenario`) expands through
`covariate_names_for` into the exact columns of `covariates`/`covariates_national`.
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from tapestry.data.geography import STATE_NAMES
from .extract import (TARGET_SOURCES, CLAIMS_SOURCES, NWSS_INDICES, NATIONAL_ONLY,
                       VintageArchive, _target_archive, _claims_archive, _nwss_frame,
                       _latest_snapshot, cutoff_time, saturday)

CALENDAR_START = '2023-09-02'  # First modelled Saturday.
CHANNELS = tuple(TARGET_SOURCES)  # nhsn_*_admissions, nssp_*_proportion, in that order.
COVARIATE_GROUPS = {
    'inpatient': ('inpatient_flu', 'inpatient_covid'),
    'outpatient': ('outpatient_flu', 'outpatient_covid'),
    'ww_wval_like': ('nwss_flu_wval_like', 'nwss_covid_wval_like', 'nwss_rsv_wval_like'),
    'ww_pct_rank': ('nwss_flu_pct_rank', 'nwss_covid_pct_rank', 'nwss_rsv_pct_rank'),
    'kinsa': ('kinsa_ili',),
}
SOURCE_GROUPS = tuple(COVARIATE_GROUPS)
COVARIATE_NAMES = tuple(name for group in COVARIATE_GROUPS.values() for name in group)
STATE_COVARIATE_NAMES = tuple(n for n in COVARIATE_NAMES if n not in NATIONAL_ONLY)
NATIONAL_COVARIATE_NAMES = tuple(n for n in COVARIATE_NAMES if n in NATIONAL_ONLY)
DEFAULT_LOOKBACK = 12
DEFAULT_HORIZONS = (1, 2, 3, 4)
FINALIZED_DATASET = 'data/processed/finalized.npz'
VINTAGED_DATASET = 'data/processed/vintaged.npz'


def covariate_names_for(covariate_set):
    """Expand a `+`-joined subset of `SOURCE_GROUPS` into its ordered column names.

    `''` means no covariates. Raises on an unknown group name so a scenario
    string typo fails fast instead of silently training without covariates.
    """
    if not covariate_set:
        return ()
    groups = covariate_set.split('+')
    unknown = set(groups) - set(SOURCE_GROUPS)
    if unknown:
        raise ValueError(f'Unknown covariate group(s): {sorted(unknown)}')
    selected = set(groups)
    return tuple(name for group, names in COVARIATE_GROUPS.items() if group in selected for name in names)


def _weekly_calendar(start, end):
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    return tuple((first + timedelta(weeks=i)).isoformat() for i in range((last - first).days // 7 + 1))


def _wednesdays(start, end):
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    first -= timedelta(days=(first.weekday() - 2) % 7)
    return tuple((first + timedelta(weeks=i)).isoformat() for i in range((last - first).days // 7 + 1))


def _locations(explicit=None):
    return tuple(explicit) if explicit else (*sorted(STATE_NAMES), 'US')


def _covariate_archives(data_root):
    """Every archive-backed covariate source, built once and reused across issuances."""
    return {name: _claims_archive(data_root, name) for name in CLAIMS_SOURCES}


def _target_panel(data_root, dates, locations, cutoff):
    """[T, L, 6] resolved as of `cutoff` (a date or Wednesday-cutoff string)."""
    panel = np.full((len(dates), len(locations), len(CHANNELS)), np.nan, np.float32)
    for c, name in enumerate(CHANNELS):
        archive = _target_archive(data_root, name)
        values, available = archive.panel(dates, locations, archive.resolve(cutoff))
        panel[:, :, c] = np.where(available, values, np.nan)
    return panel


def _claims_panel(data_root, dates, locations, cutoff, archives=None):
    archives = archives or _covariate_archives(data_root)
    panel = np.full((len(dates), len(locations), len(CLAIMS_SOURCES)), np.nan, np.float32)
    for c, name in enumerate(CLAIMS_SOURCES):
        values, available = archives[name].panel(dates, locations, archives[name].resolve(cutoff))
        panel[:, :, c] = np.where(available, values, np.nan)
    return panel


def _nwss_panel(frame, dates, locations, cutoff):
    panel = np.full((len(dates), len(locations), len(NWSS_INDICES)), np.nan, np.float32)
    visible = frame[frame.report_time <= cutoff]
    for c, name in enumerate(NWSS_INDICES):
        metric = 'pct_rank' if name.endswith('pct_rank') else 'wval_like'
        pathogen = name.removeprefix('nwss_').removesuffix('_' + metric)
        part = visible[visible.pathogen.eq(pathogen)].sort_values('report_time')
        part = part.dropna(subset=[metric]).drop_duplicates(['geo_value', 'reference_time'], keep='last')
        lookup = {(row.reference_time, row.geo_value): row[metric] for row in part.itertuples()}
        for t, day in enumerate(dates):
            for l, loc in enumerate(locations):
                if (day, loc) in lookup:
                    panel[t, l, c] = lookup[(day, loc)]
    return panel


def _kinsa_weekly(data_root, truth_cutoff):
    """Saturday-ending weekly means of daily Kinsa values at every real report date."""
    snapshot = _latest_snapshot(data_root, 'pophive_kinsa_ili')
    path = snapshot / 'archive.csv.gz'
    frame = pd.read_csv(path, dtype={'geo_type': str, 'geo_value': str})
    frame = frame[frame['geo_value'].eq('US')].copy()
    frame['report_time'] = frame['report_time'].astype(str).str[:10]
    frame['reference_time'] = frame['reference_time'].astype(str).str[:10]
    frame = frame[frame['report_time'] <= truth_cutoff]
    frame['value'] = pd.to_numeric(frame['kinsa_cough_cold_flu'], errors='coerce')
    frame.loc[~np.isfinite(frame['value']) | (frame['value'] < 0), 'value'] = np.nan
    if frame.empty:
        return pd.DataFrame(columns=['report_time', 'geo_value', 'reference_time', 'value']), snapshot.name
    frame = frame.sort_values(['reference_time', 'report_time']).drop_duplicates(
        ['reference_time', 'report_time'], keep='last')
    days = {day: (g['report_time'].to_numpy(str), g['value'].to_numpy(float))
            for day, g in frame.groupby('reference_time')}
    saturdays = sorted({(pd.Timestamp(day) + pd.Timedelta(days=(5 - pd.Timestamp(day).weekday()) % 7)).strftime('%Y-%m-%d')
                        for day in days})
    rows = []
    for saturday_ in saturdays:
        week = [(pd.Timestamp(saturday_) - pd.Timedelta(days=k)).strftime('%Y-%m-%d') for k in range(6, -1, -1)]
        if not all(day in days for day in week):
            continue
        previous = np.nan
        for release in sorted({r for day in week for r in days[day][0]}):
            values = []
            for day in week:
                reports, daily = days[day]
                index = np.searchsorted(reports, release, side='right') - 1
                values.append(daily[index] if index >= 0 else np.nan)
            value = float(np.mean(values)) if np.isfinite(values).all() else np.nan
            if not (value == previous or (np.isnan(value) and np.isnan(previous))):
                rows.append((release, 'US', saturday_, value))
            previous = value
    return pd.DataFrame(rows, columns=['report_time', 'geo_value', 'reference_time', 'value']), snapshot.name


def _national_panel(kinsa, dates, cutoff):
    panel = np.full((len(dates), 1), np.nan, np.float32)
    visible = kinsa[kinsa.report_time <= cutoff].sort_values('report_time')
    visible = visible.dropna(subset=['value']).drop_duplicates('reference_time', keep='last')
    lookup = dict(zip(visible.reference_time, visible.value))
    for t, day in enumerate(dates):
        if day in lookup:
            panel[t, 0] = lookup[day]
    return panel


def build_finalized(data_root='data', *, start=CALENDAR_START, end=None, locations=None):
    """Truth-only array: no revision structure, latest resolved value per cell."""
    locations = _locations(locations)
    end = end or date.today().isoformat()
    dates = _weekly_calendar(start, end)
    cutoff = date.today().isoformat()
    targets = _target_panel(data_root, dates, locations, cutoff)
    claims_archives = _covariate_archives(data_root)
    claims = _claims_panel(data_root, dates, locations, cutoff, claims_archives)
    nwss = _nwss_panel(_nwss_frame(data_root), dates, locations, cutoff)
    kinsa, kinsa_snapshot = _kinsa_weekly(data_root, cutoff)
    national = _national_panel(kinsa, dates, cutoff)
    covariates = np.concatenate([claims, nwss], axis=2)
    # Finalized values are known-final by definition wherever they are available.
    known_final = ~np.isnan(targets)
    return dict(dates=np.array(dates, dtype='datetime64[D]'), locations=np.array(locations),
                targets=targets, target_names=np.array(CHANNELS), known_final=known_final,
                covariates=covariates, covariate_mask=~np.isnan(covariates),
                covariate_names=np.array(STATE_COVARIATE_NAMES),
                covariates_national=national, covariate_national_names=np.array(NATIONAL_COVARIATE_NAMES),
                metadata=json.dumps(dict(version=1, kind='finalized', start=start, end=end,
                                          channels=list(CHANNELS), covariate_groups={k: list(v) for k, v in COVARIATE_GROUPS.items()},
                                          kinsa_snapshot=kinsa_snapshot)))


def build_vintaged(data_root='data', *, start, end, lookback=DEFAULT_LOOKBACK,
                    horizons=DEFAULT_HORIZONS, locations=None):
    """One entry per historical Wednesday issuance, as-of-issuance visible values."""
    locations = _locations(locations)
    issuances = _wednesdays(start, end)
    window = lookback + len(horizons)
    target_archives = {name: _target_archive(data_root, name) for name in CHANNELS}
    claims_archives = _covariate_archives(data_root)
    nwss_frame = _nwss_frame(data_root)
    kinsa, kinsa_snapshot = _kinsa_weekly(data_root, cutoff_time(end))
    n_state, n_national = len(STATE_COVARIATE_NAMES), len(NATIONAL_COVARIATE_NAMES)
    all_dates = np.empty((len(issuances), window), dtype='datetime64[D]')
    targets = np.full((len(issuances), window, len(locations), len(CHANNELS)), np.nan, np.float32)
    covariates = np.full((len(issuances), window, len(locations), n_state), np.nan, np.float32)
    national = np.full((len(issuances), window, n_national), np.nan, np.float32)
    for i, issuance in enumerate(issuances):
        end_of_context = (date.fromisoformat(issuance) - timedelta(days=4)).isoformat()
        window_dates = tuple((date.fromisoformat(end_of_context) - timedelta(weeks=w)).isoformat()
                              for w in reversed(range(lookback))) + \
            tuple((date.fromisoformat(end_of_context) + timedelta(weeks=h)).isoformat() for h in horizons)
        all_dates[i] = np.array(window_dates, dtype='datetime64[D]')
        cutoff = cutoff_time(issuance)
        for c, name in enumerate(CHANNELS):
            values, available = target_archives[name].panel(window_dates, locations, target_archives[name].resolve(cutoff))
            targets[i, :, :, c] = np.where(available, values, np.nan)
        claims = _claims_panel(data_root, window_dates, locations, cutoff, claims_archives)
        nwss = _nwss_panel(nwss_frame, window_dates, locations, cutoff)
        covariates[i] = np.concatenate([claims, nwss], axis=2)
        national[i] = _national_panel(kinsa, window_dates, cutoff)
    # Context weeks older than the two most recent are pinned reference finals by
    # assumption; the two most recent context weeks carry their real Wednesday
    # vintage and are not known-final. Horizon (forecast) weeks are never context.
    known_final = np.zeros_like(targets, dtype=bool)
    known_final[:, :lookback - 2] = ~np.isnan(targets[:, :lookback - 2])
    return dict(issuance_dates=np.array(issuances, dtype='datetime64[D]'), dates=all_dates,
                locations=np.array(locations), targets=targets, target_names=np.array(CHANNELS),
                known_final=known_final,
                covariates=covariates, covariate_mask=~np.isnan(covariates),
                covariate_names=np.array(STATE_COVARIATE_NAMES),
                covariates_national=national, covariate_national_names=np.array(NATIONAL_COVARIATE_NAMES),
                metadata=json.dumps(dict(version=1, kind='vintaged', start=start, end=end, lookback=lookback,
                                          horizons=list(horizons), channels=list(CHANNELS),
                                          covariate_groups={k: list(v) for k, v in COVARIATE_GROUPS.items()},
                                          kinsa_snapshot=kinsa_snapshot)))


def save(arrays, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        np.savez_compressed(stream, **arrays)


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('build', help='Build both finalized.npz and vintaged.npz')
    build.add_argument('--data-root', default='data')
    build.add_argument('--start', default=CALENDAR_START)
    build.add_argument('--vintaged-end', help='Last Wednesday issuance; defaults to the most recent Wednesday')
    build.add_argument('--lookback', type=int, default=DEFAULT_LOOKBACK)
    build.add_argument('--finalized-output', default=FINALIZED_DATASET)
    build.add_argument('--vintaged-output', default=VINTAGED_DATASET)
    show = sub.add_parser('show', help='Summarize a built array')
    show.add_argument('--dataset', required=True)
    args = parser.parse_args(argv)
    if args.command == 'build':
        vintaged_end = args.vintaged_end or _wednesdays(args.start, date.today().isoformat())[-1]
        finalized = build_finalized(args.data_root, start=args.start, locations=None)
        save(finalized, args.finalized_output)
        vintaged = build_vintaged(args.data_root, start=args.start, end=vintaged_end, lookback=args.lookback)
        save(vintaged, args.vintaged_output)
        print(json.dumps(dict(finalized=dict(output=args.finalized_output, shape=list(finalized['targets'].shape)),
                               vintaged=dict(output=args.vintaged_output, shape=list(vintaged['targets'].shape)))))
    else:
        arrays = load(args.dataset)
        print(json.dumps({k: (list(v.shape) if hasattr(v, 'shape') and v.dtype != object else str(v))
                           for k, v in arrays.items() if k != 'metadata'}, indent=2, default=str))


if __name__ == '__main__':
    main()
