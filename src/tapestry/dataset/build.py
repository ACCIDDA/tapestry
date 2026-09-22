"""Build the one dataset array, `data/processed/panel.npz` (user decisions 2026-09-22).

One weekly Saturday calendar carries the retrospective truth panel and, for every
historical Wednesday issuance, the exact value visible at its cutoff for every
calendar week (targets, state covariates and national covariates alike), so any
issuance's inputs can be reconstructed exactly as they were. Episodes (any
lookback, any `Scenario.asof_weeks`) are cut from it by `dataset.episodes`; see
docs/design/restructure-2026-unified.md §3 for the layout and every choice below.

In memory (`build`, `load`) the as-of arrays are dense and indexed by calendar week:
`asof_targets[w, t]` is week t as visible at the end of issuance day w, NaN when
nothing was visible, and NaN by definition for weeks after the issuance's context
end (the Saturday four days earlier). On disk (`save`) each as-of array is stored
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
`python -m tapestry.dataset.build nwss-indices` (`dataset/nwss.py`) before `build`.
`python -m tapestry.dataset.build check` verifies the stored panel against direct
`extract.extract(...)` resolution at sampled issuances and at the truth day.

`COVARIATE_GROUPS` is the single source of truth for covariate column order; a
`Scenario.covariate_set` string expands through `covariate_names_for`.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from datetime import date, timedelta
import os
from pathlib import Path
import time

import numpy as np

from .extract import (TARGET_SOURCES, CLAIMS_SOURCES, NWSS_INDICES, NATIONAL_ONLY, LOCATIONS,
                      revisions, resolve, resolve_reports, nwss_frame, kinsa_frame, extract, _latest_snapshot)

CALENDAR_START = '2023-09-02'  # First modelled Saturday.
ASOF_ARRAYS = {'asof_targets': 'targets', 'asof_covariates': 'covariates',
               'asof_covariates_national': 'covariates_national'}  # as-of array -> its truth array
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
PANEL_DATASET = 'data/processed/panel.npz'
SNAPSHOT_DATASETS = sorted({d for spec in TARGET_SOURCES.values() for d in spec[:2] if d} |
                           {spec[0] for spec in CLAIMS_SOURCES.values()} |
                           {'derived_nwss_state_indices', 'pophive_kinsa_ili'})


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


def _source(task):
    """One process per source: its truth panel and its dense as-of array [W, T, L, K]."""
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
    truth = at(truth_day, dates)
    asof = np.full((len(issuances), *truth.shape), np.nan)
    for w, seen in enumerate(visible_weeks(dates, issuances)):
        weeks = int(seen.sum())  # visible weeks are a calendar prefix
        if weeks:
            asof[w, :weeks] = at(issuances[w], dates[:weeks])
    return name, truth.astype(np.float32), asof.astype(np.float32), time.perf_counter() - started


def build(data_root='data', *, start=CALENDAR_START, truth_day=None, workers=None):
    """The panel as a dict of dense arrays; see the module docstring and the design doc."""
    truth_day = truth_day or date.today().isoformat()
    dates = weekly_calendar(start, truth_day)
    issuances = wednesdays(start, truth_day)
    names = (*CHANNELS, *CLAIMS_SOURCES, 'nwss', 'kinsa_ili')
    tasks = [(name, data_root, dates, issuances, truth_day) for name in names]
    with ProcessPoolExecutor(max_workers=workers or min(len(tasks), os.cpu_count() or 1)) as pool:
        results = {name: (truth, asof, seconds) for name, truth, asof, seconds in pool.map(_source, tasks)}
    timings = {name: round(seconds, 1) for name, (_, _, seconds) in results.items()}
    targets = np.concatenate([results[n][0] for n in CHANNELS], axis=2)
    asof_targets = np.concatenate([results[n][1] for n in CHANNELS], axis=3)
    covariates = np.concatenate([results[n][0] for n in CLAIMS_SOURCES] + [results['nwss'][0]], axis=2)
    asof_covariates = np.concatenate([results[n][1] for n in CLAIMS_SOURCES] + [results['nwss'][1]], axis=3)
    national = results['kinsa_ili'][0][:, LOCATIONS.index('US'), :]
    asof_national = results['kinsa_ili'][1][:, :, LOCATIONS.index('US'), :]
    assert list(CLAIMS_SOURCES) + list(NWSS_INDICES) == list(STATE_COVARIATE_NAMES)
    snapshots = {key: _latest_snapshot(data_root, key).name for key in SNAPSHOT_DATASETS}
    metadata = dict(version=3, kind='panel', start=start, end=dates[-1], truth_day=truth_day,
                    asof='asof_*[w, t]: calendar week t as visible by 23:59:59.999999 UTC on issuance day w; '
                         'NaN after context_end(w) = issuance - 4 days. Stored as <name>_revised mask + '
                         '<name>_values where it differs from the truth panel.',
                    channels=list(CHANNELS), locations=list(LOCATIONS),
                    covariate_groups={k: list(v) for k, v in COVARIATE_GROUPS.items()},
                    snapshots=snapshots, source_seconds=timings)
    return dict(dates=np.array(dates, dtype='datetime64[D]'), locations=np.array(LOCATIONS),
                target_names=np.array(CHANNELS), targets=targets,
                covariate_names=np.array(STATE_COVARIATE_NAMES), covariates=covariates,
                covariate_national_names=np.array(NATIONAL_COVARIATE_NAMES), covariates_national=national,
                issuance_dates=np.array(issuances, dtype='datetime64[D]'), asof_targets=asof_targets,
                asof_covariates=asof_covariates, asof_covariates_national=asof_national,
                metadata=json.dumps(metadata))


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
    for day, weeks, expected in days:
        frame = extract(name, day, data_root=data_root, dates=dates[:weeks])
        direct = np.full((weeks, len(LOCATIONS)), np.nan, np.float32)
        direct[frame.date.map({d: i for i, d in enumerate(dates)}).to_numpy(int),
               frame.location.map({l: i for i, l in enumerate(LOCATIONS)}).to_numpy(int)] = frame.value
        stored = expected[:weeks]
        equal = (direct == stored) | (np.isnan(direct) & np.isnan(stored))
        mismatches[day] = int((~equal).sum())
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
    for name in (*CHANNELS, *CLAIMS_SOURCES, *NWSS_INDICES, 'kinsa_ili'):
        if name in CHANNELS:
            k, truth, asof = list(CHANNELS).index(name), arrays['targets'], arrays['asof_targets']
        elif name in STATE_COVARIATE_NAMES:
            k, truth, asof = list(STATE_COVARIATE_NAMES).index(name), arrays['covariates'], arrays['asof_covariates']
        else:  # national: compare on the US column, NaN elsewhere
            k = list(NATIONAL_COVARIATE_NAMES).index(name)
            truth = np.full((len(dates), len(locations), 1), np.nan, np.float32)
            truth[:, locations.index('US'), 0] = arrays['covariates_national'][:, k]
            asof = np.full((len(issuances), len(dates), len(locations), 1), np.nan, np.float32)
            asof[:, :, locations.index('US'), 0] = arrays['asof_covariates_national'][:, :, k]
            k = 0
        days = [(issuances[w], int(visible[w].sum()), asof[w, :, :, k]) for w in picked]
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
        arrays = build(args.data_root, start=args.start, truth_day=args.truth_day, workers=args.workers)
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
