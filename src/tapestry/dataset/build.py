"""Build the one dataset array, `data/processed/panel.npz` (user decision 2026-09-22).

Replaces `finalized.npz` + `vintaged.npz`. One weekly Saturday calendar carries the
retrospective truth panel; an as-of overlay records, for every historical Wednesday
issuance, what was visible at its cutoff. Episodes (any lookback) are cut from it by
`dataset.episodes`; see docs/design/restructure-2026-unified.md §3 for the layout
and every choice below.

Build constants (documented in the design doc):
- `CALENDAR_START`: first Saturday of the calendar. Nothing earlier exists in the
  panel; episodes pad earlier context weeks as unavailable.
- `ASOF_TARGET_WEEKS` (R=2): the overlay keeps each issuance's as-of target values
  for the R most recent context weeks, as the previous vintaged builder did.
- `ASOF_COVARIATE_WEEKS` (D=52): the overlay keeps as-of covariate values for the D
  most recent context weeks, so a vintaged episode's covariates are as-of for any
  lookback up to D (the previous builder resolved every covariate week as of the
  issuance). A vintaged lookback above D raises.

The truth panel is resolved at the end of the build day (`truth_day`, default today,
recorded in metadata). Sources are read in parallel, one process per source.

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
                      revisions, resolve, resolve_reports, nwss_frame, kinsa_frame, _latest_snapshot)

CALENDAR_START = '2023-09-02'  # First modelled Saturday.
ASOF_TARGET_WEEKS = 2
ASOF_COVARIATE_WEEKS = 52
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


def overlay_dates(issuance, depth):
    """The `depth` reference weeks an issuance's overlay covers, oldest first."""
    end = date.fromisoformat(context_end(issuance))
    return tuple((end - timedelta(weeks=depth - 1 - j)).isoformat() for j in range(depth))


def _source(task):
    """One process per source: its truth panel and its per-issuance as-of overlay."""
    name, data_root, dates, issuances, truth_day, depth = task
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
    overlay = np.full((len(issuances), depth, *truth.shape[1:]), np.nan)
    for w, issuance in enumerate(issuances):
        window = overlay_dates(issuance, depth)
        overlay[w] = at(issuance, window)
        overlay[w, np.array(window) < dates[0]] = np.nan  # nothing exists before the calendar
    return name, truth.astype(np.float32), overlay.astype(np.float32), time.perf_counter() - started


def build(data_root='data', *, start=CALENDAR_START, truth_day=None, workers=None):
    """The panel as a dict of arrays; see the module docstring and the design doc."""
    truth_day = truth_day or date.today().isoformat()
    dates = weekly_calendar(start, truth_day)
    issuances = wednesdays(start, truth_day)
    depth = {name: ASOF_TARGET_WEEKS for name in CHANNELS}
    depth.update({name: ASOF_COVARIATE_WEEKS for name in (*CLAIMS_SOURCES, 'nwss', 'kinsa_ili')})
    tasks = [(name, data_root, dates, issuances, truth_day, d) for name, d in depth.items()]
    with ProcessPoolExecutor(max_workers=workers or min(len(tasks), os.cpu_count() or 1)) as pool:
        results = {name: (truth, overlay, seconds) for name, truth, overlay, seconds in pool.map(_source, tasks)}
    timings = {name: round(seconds, 1) for name, (_, _, seconds) in results.items()}
    targets = np.concatenate([results[n][0] for n in CHANNELS], axis=2)
    asof_targets = np.concatenate([results[n][1] for n in CHANNELS], axis=3)
    covariates = np.concatenate([results[n][0] for n in CLAIMS_SOURCES] + [results['nwss'][0]], axis=2)
    asof_covariates = np.concatenate([results[n][1] for n in CLAIMS_SOURCES] + [results['nwss'][1]], axis=3)
    national = results['kinsa_ili'][0][:, LOCATIONS.index('US'), :]
    asof_national = results['kinsa_ili'][1][:, :, LOCATIONS.index('US'), :]
    assert list(CLAIMS_SOURCES) + list(NWSS_INDICES) == list(STATE_COVARIATE_NAMES)
    snapshots = {key: _latest_snapshot(data_root, key).name for key in SNAPSHOT_DATASETS}
    metadata = dict(version=2, kind='panel', start=start, end=dates[-1], truth_day=truth_day,
                    asof_target_weeks=ASOF_TARGET_WEEKS, asof_covariate_weeks=ASOF_COVARIATE_WEEKS,
                    overlay_reference='overlay[w, j] is reference week context_end(issuance_w) - (depth-1-j) weeks; '
                                      'context_end = issuance - 4 days; values visible by 23:59:59.999999 UTC on the issuance day',
                    channels=list(CHANNELS), locations=list(LOCATIONS),
                    covariate_groups={k: list(v) for k, v in COVARIATE_GROUPS.items()},
                    snapshots=snapshots, source_seconds=timings)
    return dict(dates=np.array(dates, dtype='datetime64[D]'), locations=np.array(LOCATIONS),
                target_names=np.array(CHANNELS), targets=targets,
                covariate_names=np.array(STATE_COVARIATE_NAMES), covariates=covariates,
                covariate_mask=~np.isnan(covariates),
                covariate_national_names=np.array(NATIONAL_COVARIATE_NAMES), covariates_national=national,
                issuance_dates=np.array(issuances, dtype='datetime64[D]'), asof_targets=asof_targets,
                asof_covariates=asof_covariates, asof_covariates_national=asof_national,
                metadata=json.dumps(metadata))


def save(arrays, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.npz')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


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
    args = parser.parse_args(argv)
    if args.command == 'build':
        started = time.perf_counter()
        arrays = build(args.data_root, start=args.start, truth_day=args.truth_day, workers=args.workers)
        save(arrays, args.output)
        print(json.dumps(dict(output=args.output, targets=list(arrays['targets'].shape),
                              issuances=len(arrays['issuance_dates']), seconds=round(time.perf_counter() - started, 1),
                              source_seconds=json.loads(str(arrays['metadata']))['source_seconds'])))
    else:
        arrays = load(args.dataset)
        print(json.dumps({k: (list(v.shape) if k != 'metadata' else json.loads(str(v))) for k, v in arrays.items()},
                         indent=2, default=str))


if __name__ == '__main__':
    main()
