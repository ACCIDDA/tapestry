"""Build the one dataset array, `data/processed/panel.npz` (user decisions 2026-09-22).

One weekly Saturday calendar carries the retrospective truth panel and, for every
historical Wednesday issuance, the exact value visible at its cutoff for every
calendar week (targets, state covariates and national covariates alike), so any
issuance's inputs can be reconstructed exactly as they were. Episodes (any
lookback, any `Scenario.asof_weeks`) are cut from it by `dataset.episodes`; see
docs/architecture.md for the layout and every choice below.

In memory (`build`, `load`) the as-of arrays are dense and indexed by calendar week:
`asof_targets[w, t]` is week t as visible at issuance w's FluSight submission deadline,
NaN when nothing was visible, and NaN by definition for weeks after the issuance's
context end (the Saturday four days earlier).

Deadlines (2026-10-05, user decision: per Hub, never a common earliest deadline): each
Hub's deadline is Wednesday 23:00 America/New_York, except the holiday extensions in
`HOLIDAY_DEADLINES`, read from each Hub's `hub-config/tasks.json` Git history. The
Wednesday identifier and its context Saturday never move; only the information
cutoff does. The main as-of arrays use the FluSight deadline (flu targets, and the
inputs of every training episode). For each other Hub, issuances whose deadline
differs are stored separately (`hub_<hub>_issuances`, `hub_<hub>_asof_*`) and
`for_hub` swaps them in, so COVID and RSV forecasts are scored on the inputs visible
at their own Hub's deadline. `issuance_cutoffs_utc[h, w]` records every deadline. On disk (`save`) each as-of array is stored
only where it differs from the truth panel: a bool mask `<name>_revised [W, T, ...]`
plus the differing values `<name>_values [N]` (NaN = visible in truth but not at
the cutoff), in `np.savez_compressed`. `load` rebuilds the dense arrays exactly.

Values are float32, like the truth panel. float16 was considered (user suggestion)
and rejected: it represents integers exactly only up to 2048 and has a maximum of
65504 (step 32 near 50,000), so US weekly influenza admission counts would be
rounded or overflow.

The truth panel is resolved at the end of the build day (`truth_day`, default today,
recorded in metadata). Sources are read in parallel, one process per source.
The wastewater indices are a derived raw snapshot, rebuilt from Delphi NWSS by
`python -m chromantis.dataset.build nwss-indices` (`dataset/nwss.py`) before `build`.
`python -m chromantis.dataset.build check` verifies the stored panel against direct
`extract.extract(...)` resolution at sampled issuances and at the truth day.

`COVARIATE_GROUPS` is the single source of truth for covariate column order; a
`Scenario.covariate_set` string expands through `covariate_names_for`.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from datetime import date, datetime, time as clock, timedelta
import os
from zoneinfo import ZoneInfo
from pathlib import Path
import time

import numpy as np

from .extract import (TARGET_SOURCES, DELPHI_COVARIATES, NWSS_INDICES, NATIONAL_ONLY, LOCATIONS,
                      revisions, resolve, resolve_reports, nwss_frame, kinsa_frame, extract, _latest_snapshot)

CALENDAR_START = '2022-05-14'  # Twelve context weeks before the 2022–23 season.
ASOF_ARRAYS = {'asof_targets': 'targets', 'asof_covariates': 'covariates',
               'asof_covariates_national': 'covariates_national'}  # as-of array -> its truth array
CHANNELS = tuple(TARGET_SOURCES)  # nhsn_*_admissions, nssp_*_proportion, in that order.
COVARIATE_GROUPS = {
    'inpatient': ('inpatient_flu', 'inpatient_covid'),
    'outpatient': ('outpatient_flu', 'outpatient_covid'),
    'ww_wval_like': ('nwss_flu_wval_like', 'nwss_covid_wval_like', 'nwss_rsv_wval_like'),
    'ww_pct_rank': ('nwss_flu_pct_rank', 'nwss_covid_pct_rank', 'nwss_rsv_pct_rank'),
    'kinsa': ('kinsa_ili',),
    'ilinet': ('ilinet_ili',),
    'clinical_lab': ('clinical_lab_flu_pct_positive',),
    'flusurv': ('flusurv_flu_rate',),
}
SOURCE_GROUPS = tuple(COVARIATE_GROUPS)
# Selection aliases do not change the stored panel's covariate column order.
COVARIATE_SELECTORS = {**COVARIATE_GROUPS,
    'outpatient_flu': ('outpatient_flu',),
    'ww_flu': ('nwss_flu_wval_like',),
    'ww_flu_pct_rank': ('nwss_flu_pct_rank',)}
COVARIATE_NAMES = tuple(name for group in COVARIATE_GROUPS.values() for name in group)
STATE_COVARIATE_NAMES = tuple(n for n in COVARIATE_NAMES if n not in NATIONAL_ONLY)
NATIONAL_COVARIATE_NAMES = tuple(n for n in COVARIATE_NAMES if n in NATIONAL_ONLY)
LAG_ONE_COVARIATES = frozenset({'ilinet_ili', 'clinical_lab_flu_pct_positive', 'flusurv_flu_rate'})
PANEL_DATASET = 'data/processed/panel.npz'
HUBS = ('flusight', 'covid', 'rsv')
CHANNEL_HUBS = ('flusight', 'covid', 'rsv', 'flusight', 'covid', 'rsv')  # Hub of each CHANNELS entry
# Deadline day per Hub forecast reference date (the Saturday after the Wednesday).
# FluSight: tasks.json submissions_due end -2 (Thursday) 6da755c7 2024-12-26, reverted
# b791af8e 2025-01-07; end +3 (Tuesday) 4dbd6833 2025-12-28; +2 (Monday) 2b92e8db
# 2025-12-31; reverted fc07d821 2026-01-05. COVID: end -2 5e96cfd 2024-12-23, reverted
# ca66561 2025-01-03; README holiday schedule 0dbd496 2025-12-29 (Dec 29, Jan 4), reverted
# c07049a 2026-01-05. RSV: same 2025-26 schedule 49854c8/9aba0e4, reverted da5a003; the
# RSV Hub repository starts 2025-08-14, so 2024-25 RSV keeps the plain Wednesday.
HOLIDAY_DEADLINES = {
    'flusight': {'2024-12-28': '2024-12-26', '2025-01-04': '2025-01-02',
                 '2025-12-27': '2025-12-30', '2026-01-03': '2026-01-05'},
    'covid': {'2024-12-28': '2024-12-26', '2025-01-04': '2025-01-02',
              '2025-12-27': '2025-12-29', '2026-01-03': '2026-01-04'},
    'rsv': {'2025-12-27': '2025-12-29', '2026-01-03': '2026-01-04'},
}
DEADLINE_ZONE, DEADLINE_HOUR = ZoneInfo('America/New_York'), 23


def deadline(issuance, hub):
    """Aware submission deadline of Wednesday `issuance` for `hub` (23:00 Eastern, holiday extensions)."""
    nominal = date.fromisoformat(str(issuance)[:10])
    day = HOLIDAY_DEADLINES[hub].get((nominal + timedelta(days=3)).isoformat(), nominal.isoformat())
    return datetime.combine(date.fromisoformat(day), clock(DEADLINE_HOUR), DEADLINE_ZONE)


def utc(moment):
    return moment.astimezone(ZoneInfo('UTC')).strftime('%Y-%m-%dT%H:%M:%SZ')
SNAPSHOT_DATASETS = sorted({d for spec in TARGET_SOURCES.values() for d in spec[:2] if d} |
                           {spec[0] for spec in DELPHI_COVARIATES.values()} |
                           {'derived_nwss_state_indices', 'pophive_kinsa_ili'})


def covariate_names_for(covariate_set):
    """Expand a `+`-joined subset of `SOURCE_GROUPS` into its ordered column names.

    `''` means no covariates. Raises on an unknown group name so a scenario
    string typo fails fast instead of silently training without covariates.
    """
    if not covariate_set:
        return ()
    groups = covariate_set.split('+')
    unknown = set(groups) - set(COVARIATE_SELECTORS)
    if unknown:
        raise ValueError(f'Unknown covariate group(s): {sorted(unknown)}')
    selected = {name for group in groups for name in COVARIATE_SELECTORS[group]}
    return tuple(name for name in COVARIATE_NAMES if name in selected)


def weekly_calendar(start, end):
    """Saturdays from `start` through the last one on or before `end`."""
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    return tuple((first + timedelta(weeks=i)).isoformat() for i in range((last - first).days // 7 + 1))


def wednesdays(start, end):
    """Wednesday issuances from the one on or before `start` through the last on or before `end`."""
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    first -= timedelta(days=(first.weekday() - 2) % 7)
    return tuple((first + timedelta(weeks=i)).isoformat() for i in range((last - first).days // 7 + 1))


def context_end(issuance):
    """The Saturday four days before a Wednesday issuance: its last context week."""
    return (date.fromisoformat(str(issuance)[:10]) - timedelta(days=4)).isoformat()


def visible_weeks(dates, issuances):
    """[W, T] bool: calendar week t is at or before issuance w's context end."""
    ends = np.array([context_end(str(i)) for i in issuances], dtype='datetime64[D]')
    return np.asarray(dates, dtype='datetime64[D]')[None, :] <= ends[:, None]


def finalized_admissions(name, values, dates, data_root):
    """Prefer finite CDC finalized admission counts for retrospective truth only."""
    if not name.startswith('nhsn_'):
        return values
    import pandas as pd
    column = TARGET_SOURCES[name][3]
    frame = pd.read_json(_latest_snapshot(data_root, 'cdc_nhsn_final') / 'data.ndjson.gz', lines=True)
    frame['day'] = frame.weekendingdate.astype(str).str[:10]
    frame['location'] = frame.jurisdiction.replace({'USA': 'US'})
    frame['value'] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame[frame.day.isin(dates) & frame.location.isin(LOCATIONS) & np.isfinite(frame.value) & frame.value.ge(0)]
    if frame.duplicated(['day', 'location']).any():
        raise ValueError('Duplicate finalized NHSN location/week')
    result = values.copy()
    ti, li = {d:i for i,d in enumerate(dates)}, {l:i for i,l in enumerate(LOCATIONS)}
    for row in frame.itertuples():
        result[ti[row.day], li[row.location], ...] = row.value
    return result


def _source(task):
    """One process per source: truth, FluSight-deadline as-of [W, T, L, K], and other Hubs' differing rows."""
    name, data_root, dates, issuances, truth_day = task
    started = time.perf_counter()
    if name == 'nwss':
        frame, names = nwss_frame(data_root), list(NWSS_INDICES)
        at = lambda day, days: resolve_reports(frame, day, days, LOCATIONS, names)
    elif name == 'kinsa_ili':
        frame, names = kinsa_frame(data_root), ['kinsa_ili']
        at = lambda day, days: resolve_reports(frame, day, days, LOCATIONS, names)
    else:
        archive = revisions(name, data_root)
        at = lambda day, days: resolve(archive, day, days, LOCATIONS)[:, :, None]
    truth = finalized_admissions(name, at(truth_day, dates), dates, data_root)
    visible = visible_weeks(dates, issuances)

    def asof_rows(rows, hub):
        result = np.full((len(rows), *truth.shape), np.nan)
        for i, w in enumerate(rows):
            weeks = int(visible[w].sum())  # visible weeks are a calendar prefix
            if weeks:
                result[i, :weeks] = at(deadline(issuances[w], hub), dates[:weeks])
        return result.astype(np.float32)

    asof = asof_rows(range(len(issuances)), HUBS[0])
    others = {}
    for hub in HUBS[1:]:
        rows = [w for w, i in enumerate(issuances) if deadline(i, hub) != deadline(i, HUBS[0])]
        others[hub] = (np.array(rows, int), asof_rows(rows, hub))
    return name, truth.astype(np.float32), asof, others, time.perf_counter() - started


def build(data_root='data', *, start=CALENDAR_START, truth_day=None, workers=None, sources=None):
    """The panel as a dict of dense arrays; see the module docstring and the design doc."""
    truth_day = truth_day or date.today().isoformat()
    dates = weekly_calendar(start, truth_day)
    issuances = wednesdays(start, truth_day)
    names = (*CHANNELS, *DELPHI_COVARIATES, 'nwss', 'kinsa_ili')
    selected = set(names if sources is None else sources)
    if not selected or selected - set(names):
        raise ValueError('Unknown or empty build source selection')
    tasks = [(name, data_root, dates, issuances, truth_day) for name in names if name in selected]
    with ProcessPoolExecutor(max_workers=workers or min(len(tasks), os.cpu_count() or 1)) as pool:
        results = {name: (truth, asof, others, seconds) for name, truth, asof, others, seconds in pool.map(_source, tasks)}
    for name in set(names) - selected:
        truth = np.full((len(dates), len(LOCATIONS), len(NWSS_INDICES) if name == 'nwss' else 1), np.nan, dtype=np.float32)
        asof = np.full((len(issuances), *truth.shape), np.nan, dtype=np.float32)
        others = {}
        for hub in HUBS[1:]:
            rows = np.array([w for w, i in enumerate(issuances) if deadline(i, hub) != deadline(i, HUBS[0])], int)
            others[hub] = (rows, asof[rows])
        results[name] = (truth, asof, others, 0.)
    timings = {name: round(seconds, 1) for name, (*_, seconds) in results.items()}
    targets = np.concatenate([results[n][0] for n in CHANNELS], axis=2)
    asof_targets = np.concatenate([results[n][1] for n in CHANNELS], axis=3)
    columns = {n: (results[n][0], results[n][1]) for n in DELPHI_COVARIATES}
    columns |= {n: (results['nwss'][0][..., k:k + 1], results['nwss'][1][..., k:k + 1])
                for k, n in enumerate(NWSS_INDICES)}
    covariates = np.concatenate([columns[n][0] for n in STATE_COVARIATE_NAMES], axis=2)
    asof_covariates = np.concatenate([columns[n][1] for n in STATE_COVARIATE_NAMES], axis=3)
    national = results['kinsa_ili'][0][:, LOCATIONS.index('US'), :]
    asof_national = results['kinsa_ili'][1][:, :, LOCATIONS.index('US'), :]
    hub_arrays = {}
    for hub in HUBS[1:]:
        rows = results[CHANNELS[0]][2][hub][0]
        hub_arrays[f'hub_{hub}_issuances'] = rows
        hub_arrays[f'hub_{hub}_asof_targets'] = np.concatenate([results[n][2][hub][1] for n in CHANNELS], axis=3)
        columns_ = {n: results[n][2][hub][1] for n in DELPHI_COVARIATES}
        columns_ |= {n: results['nwss'][2][hub][1][..., k:k + 1] for k, n in enumerate(NWSS_INDICES)}
        hub_arrays[f'hub_{hub}_asof_covariates'] = np.concatenate([columns_[n] for n in STATE_COVARIATE_NAMES], axis=3)
        hub_arrays[f'hub_{hub}_asof_covariates_national'] = results['kinsa_ili'][2][hub][1][:, :, LOCATIONS.index('US'), :]
    cutoffs = np.array([[utc(deadline(i, hub)) for i in issuances] for hub in HUBS])
    snapshots = {key: _latest_snapshot(data_root, key).name for key in [*SNAPSHOT_DATASETS, 'cdc_nhsn_final']}
    metadata = dict(finalized_admissions_policy='Finite CDC finalized NHSN counts take precedence in retrospective truth; archive-resolved values remain where the finalized source has no finite value. Historical as-of arrays never receive this override.', version=4, kind='panel', start=start, end=dates[-1], truth_day=truth_day,
                    asof='asof_*[w, t]: calendar week t as visible at issuance w\'s FluSight deadline '
                         '(Wednesday 23:00 America/New_York, holiday extensions in HOLIDAY_DEADLINES); '
                         'date-only release labels count at the end of their UTC day. NaN after '
                         'context_end(w) = issuance - 4 days. Stored as <name>_revised mask + '
                         '<name>_values where it differs from the truth panel. hub_<hub>_asof_* hold the '
                         'rows hub_<hub>_issuances at that Hub\'s own deadline (dataset.build.for_hub).',
                    hubs=list(HUBS), holiday_deadlines=HOLIDAY_DEADLINES,
                    channels=list(CHANNELS), locations=list(LOCATIONS),
                    covariate_groups={k: list(v) for k, v in COVARIATE_GROUPS.items()},
                    snapshots=snapshots, source_seconds=timings,
                    available_sources=sorted(selected), omitted_sources=sorted(set(names)-selected))
    return dict(dates=np.array(dates, dtype='datetime64[D]'), locations=np.array(LOCATIONS),
                target_names=np.array(CHANNELS), targets=targets,
                covariate_names=np.array(STATE_COVARIATE_NAMES), covariates=covariates,
                covariate_national_names=np.array(NATIONAL_COVARIATE_NAMES), covariates_national=national,
                issuance_dates=np.array(issuances, dtype='datetime64[D]'), asof_targets=asof_targets,
                asof_covariates=asof_covariates, asof_covariates_national=asof_national,
                hub_names=np.array(HUBS), issuance_cutoffs_utc=cutoffs, **hub_arrays,
                metadata=json.dumps(metadata))


def for_hub(panel, hub):
    """Shallow copy of `panel` whose as-of arrays are visible at `hub`'s own deadlines.

    Only the rows `hub_<hub>_issuances` differ from the FluSight deadline; they are
    swapped in (already masked like the main arrays by `cv.masked`).
    `forecast_cutoff_utc` records each issuance's deadline for that Hub."""
    if 'hub_names' not in panel:
        raise ValueError('This panel predates per-Hub deadlines (2026-10-05); rebuild it with '
                         '`python -m chromantis.dataset.build build` before fitting or evaluating')
    hubs = [str(h) for h in panel['hub_names']]
    out = dict(panel, forecast_cutoff_utc=panel['issuance_cutoffs_utc'][hubs.index(hub)])
    if hub == HUBS[0]:
        return out
    rows = panel[f'hub_{hub}_issuances']
    for name in ASOF_ARRAYS:
        if len(rows):
            out[name] = panel[name].copy()
            out[name][rows] = panel[f'hub_{hub}_{name}']
    return out


def _visible_like(arrays, name):
    visible = visible_weeks(arrays['dates'], arrays['issuance_dates'])
    return visible.reshape(visible.shape + (1,) * (arrays[name].ndim - 2))


def encode(arrays):
    """Dense panel -> on-disk arrays: each as-of array only where it differs from the truth."""
    out = {k: v for k, v in arrays.items() if k not in ASOF_ARRAYS}
    for name, truth_name in ASOF_ARRAYS.items():
        asof, truth, visible = arrays[name], arrays[truth_name][None], _visible_like(arrays, name)
        if not np.isnan(asof[~np.broadcast_to(visible, asof.shape)]).all():
            raise ValueError(f'{name} holds values after an issuance context end')
        same = (asof == truth) | (np.isnan(asof) & np.isnan(truth))
        revised = visible & ~same
        out[f'{name}_revised'], out[f'{name}_values'] = revised, asof[revised].astype(np.float32)
    return out


def decode(stored):
    """On-disk arrays -> dense panel (inverse of `encode`)."""
    arrays = {k: v for k, v in stored.items() if not k.endswith(('_revised', '_values'))}
    for name, truth_name in ASOF_ARRAYS.items():
        revised = stored[f'{name}_revised']
        visible = np.broadcast_to(_visible_like(arrays | {name: revised}, name), revised.shape)
        dense = np.where(visible, arrays[truth_name][None], np.nan).astype(np.float32)
        dense[revised] = stored[f'{name}_values']
        arrays[name] = dense
    return arrays


def save(arrays, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npz')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **encode(arrays))
    temporary.replace(path)


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return decode({k: data[k] for k in data.files})


def _check_source(task):
    """Direct `extract` resolution of one source at each sampled day -> max mismatch count."""
    name, data_root, days, dates = task
    mismatches = {}
    for index, (day, weeks, expected) in enumerate(days):
        frame = extract(name, day, data_root=data_root, dates=dates[:weeks])
        direct = np.full((weeks, len(LOCATIONS)), np.nan, np.float32)
        direct[frame.date.map({d: i for i, d in enumerate(dates)}).to_numpy(int),
               frame.location.map({l: i for i, l in enumerate(LOCATIONS)}).to_numpy(int)] = frame.value
        if index == len(days) - 1:
            direct = finalized_admissions(name, direct, dates[:weeks], data_root)
        stored = expected[:weeks]
        equal = (direct == stored) | (np.isnan(direct) & np.isnan(stored))
        mismatches[str(day)] = int((~equal).sum())
    return name, mismatches


def check(path=PANEL_DATASET, data_root='data', samples=4, seed=0, workers=None):
    """Stored panel == direct `extract(...)` at `samples` random issuances plus the truth day, per source."""
    arrays = load(path)
    metadata = json.loads(str(arrays['metadata']))
    dates = [str(d) for d in arrays['dates']]
    issuances = [str(d) for d in arrays['issuance_dates']]
    picked = sorted(np.random.default_rng(seed).choice(len(issuances), size=min(samples, len(issuances)), replace=False))
    visible = visible_weeks(dates, issuances)
    locations = list(arrays['locations'])
    tasks = []
    stored = [str(n) for n in arrays['covariate_names']]
    for name in (*CHANNELS, *DELPHI_COVARIATES, *NWSS_INDICES, 'kinsa_ili'):
        if name in CHANNELS:
            k, truth, asof = list(CHANNELS).index(name), arrays['targets'], arrays['asof_targets']
        elif name in stored:
            k, truth, asof = stored.index(name), arrays['covariates'], arrays['asof_covariates']
        else:  # national: compare on the US column, NaN elsewhere
            k = list(NATIONAL_COVARIATE_NAMES).index(name)
            truth = np.full((len(dates), len(locations), 1), np.nan, np.float32)
            truth[:, locations.index('US'), 0] = arrays['covariates_national'][:, k]
            asof = np.full((len(issuances), len(dates), len(locations), 1), np.nan, np.float32)
            asof[:, :, locations.index('US'), 0] = arrays['asof_covariates_national'][:, :, k]
            k = 0
        days = [(deadline(issuances[w], HUBS[0]), int(visible[w].sum()), asof[w, :, :, k]) for w in picked]
        days.append((metadata['truth_day'], len(dates), truth[:, :, k]))
        tasks.append((name, data_root, days, dates))
    with ProcessPoolExecutor(max_workers=workers or min(len(tasks), os.cpu_count() or 1)) as pool:
        results = dict(pool.map(_check_source, tasks))
    return dict(issuances=[issuances[w] for w in picked], truth_day=metadata['truth_day'],
                mismatched_cells=results, exact=not any(v for r in results.values() for v in r.values()))


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    build_parser = sub.add_parser('build', help='Build panel.npz')
    build_parser.add_argument('--data-root', default='data')
    build_parser.add_argument('--start', default=CALENDAR_START)
    build_parser.add_argument('--truth-day', help='Resolve truth at the end of this UTC day (default: today)')
    build_parser.add_argument('--workers', type=int, help='Parallel source processes (default: one per source)')
    build_parser.add_argument('--output', default=PANEL_DATASET)
    build_parser.add_argument('--sources', nargs='+', help='Operational subset; omitted columns remain NaN and are recorded in metadata')
    show = sub.add_parser('show', help='Summarize a built panel')
    show.add_argument('--dataset', default=PANEL_DATASET)
    checked = sub.add_parser('check', help='Compare the stored panel with direct extract() at sampled issuances')
    checked.add_argument('--dataset', default=PANEL_DATASET)
    checked.add_argument('--data-root', default='data')
    checked.add_argument('--samples', type=int, default=4, help='Random issuances checked (plus the truth day)')
    checked.add_argument('--seed', type=int, default=0)
    indices = sub.add_parser('nwss-indices', help='Rebuild the derived_nwss_state_indices snapshot from '
                             'delphi_nwss + delphi_nwss_aux (run before build; see dataset/nwss.py)')
    indices.add_argument('--data-root', default='data')
    args = parser.parse_args(argv)
    if args.command == 'nwss-indices':
        from .nwss import register
        started = time.perf_counter()
        result = register(args.data_root)
        print(json.dumps(dict(dataset_key=result['dataset_key'], snapshot_id=result['snapshot_id'],
                              rows=result['rows'], report_time_range=result['report_time_range'],
                              seconds=round(time.perf_counter() - started, 1))))
    elif args.command == 'build':
        started = time.perf_counter()
        arrays = build(args.data_root, start=args.start, truth_day=args.truth_day, workers=args.workers, sources=args.sources)
        save(arrays, args.output)
        print(json.dumps(dict(output=args.output, targets=list(arrays['targets'].shape),
                              issuances=len(arrays['issuance_dates']), seconds=round(time.perf_counter() - started, 1),
                              megabytes=round(Path(args.output).stat().st_size / 1e6, 2),
                              source_seconds=json.loads(str(arrays['metadata']))['source_seconds'])))
    elif args.command == 'check':
        result = check(args.dataset, args.data_root, args.samples, args.seed)
        print(json.dumps(result, indent=2))
        if not result['exact']:
            raise SystemExit(1)
    else:
        arrays = load(args.dataset)
        print(json.dumps({k: (list(v.shape) if k != 'metadata' else json.loads(str(v))) for k, v in arrays.items()},
                         indent=2, default=str))


if __name__ == '__main__':
    main()
