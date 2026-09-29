"""Freeze real historical deadline inputs without changing retrospective truth.

Run from repository root. Wednesday issuance IDs retain the same context Saturday;
forecast_cutoff_utc records the actual (holiday-adjusted) information cutoff.
Targets are actual values from historical public Hub files or dated Delphi reports.
Covariates are resolved independently from each source's native report archive.
No future truth or assumed publication lag supplies an unavailable input.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import sys
import time as timing
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from tapestry.dataset.build import load, save
from tapestry.dataset import extract as ex
from tapestry.dataset.cv import season

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'b0-audit'))
from deadline_availability import CHANNELS, HUBS, delphi_reports, state, git, geos, fips

HOLIDAYS = {
    '2024-12-28': '2024-12-26',
    '2025-01-04': '2025-01-02',
    '2025-12-27': '2025-12-29',
    '2026-01-03': '2026-01-04',
}
# A joint issuance must satisfy all participating Hubs, so take their earliest
# deadline. The FluSight deadlines for the last two rounds are one day later.
HOLIDAY_EVIDENCE = {
    '2024-12-28,2025-01-04': {
        'flusight': '6da755c738e2ac01c99f5d48c0a7a967c68a766e',
        'covid': 'a79500e (tasks.json Thursday extension)',
        'policy': 'reference_date minus 2 days, 23:00 America/New_York',
    },
    '2025-12-27,2026-01-03': {
        'policy': 'Earliest COVID/RSV deadline: Dec29/Jan4 23:00 ET; FluSight Dec30/Jan5',
        'audit': 'docs/results/b1-to-b0-chain/availability_inventory.md',
    },
}


def deadline(issuance):
    nominal = date.fromisoformat(str(issuance))
    reference = nominal + timedelta(days=3)
    day = date.fromisoformat(HOLIDAYS.get(reference.isoformat(), nominal.isoformat()))
    return datetime.combine(day, time(23), ZoneInfo('America/New_York'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def covariate_worker(task):
    name, data_root, dates, issuances, locations = task
    started = timing.monotonic()
    if name == 'nwss':
        # Derived releases are complete states and carry explicit NaNs when a
        # site/state index is withdrawn or no longer eligible. Preserve those
        # rows so an older reported value cannot survive a later null release.
        raw = pd.read_csv(ex._latest_snapshot(data_root, ex.NWSS_DERIVED_DATASET) / 'data.csv.gz')
        names, parts = list(ex.NWSS_INDICES), []
        for cov in names:
            metric = 'pct_rank' if cov.endswith('pct_rank') else 'wval_like'
            pathogen = cov.removeprefix('nwss_').removesuffix('_' + metric)
            part = raw.loc[raw.pathogen.str.lower().eq(pathogen)]
            parts.append(pd.DataFrame(dict(report_time=part.report_time.astype(str).str[:10],
                reference_time=part.reference_time.astype(str).str[:10],
                location=part.geo_value.astype(str).str.upper(), name=cov,
                value=pd.to_numeric(part[metric], errors='coerce'))))
        frame = pd.concat(parts, ignore_index=True)
    elif name == 'kinsa_ili':
        frame, names = ex.kinsa_frame(data_root), ['kinsa_ili']
    else:
        archive = ex.revisions(name, data_root)
        # Covariates have Delphi per-cell revision streams only, no Hub tier.
        frame = archive.rows.rename(columns={'day': 'reference_time'})
        frame['name'] = name
        frame['report_time'] = frame['release']
        names = [name]
    frame = frame.copy()
    # Most archive timestamps are UTC midnight labels. Treat all midnight stamps
    # conservatively as end-of-day; preserve explicit non-midnight timestamps.
    release = pd.to_datetime(frame.report_time, utc=True, format='mixed')
    midnight = release.eq(release.dt.normalize())
    frame['release'] = release + pd.to_timedelta(midnight.astype(int), unit='D') - pd.to_timedelta(midnight.astype(int), unit='ns')
    frame = frame.sort_values('release', kind='stable')
    result = np.full((len(issuances), len(dates), len(locations), len(names)), np.nan, np.float32)
    reported = np.zeros(result.shape, bool)
    ti, li, ni = ({v: i for i, v in enumerate(vals)} for vals in (dates, locations, names))
    first = {}
    for name_ in names:
        subset = frame.loc[frame.name.eq(name_) & frame.value.notna()]
        earliest = subset.drop_duplicates(['reference_time', 'location'], keep='first').copy()
        lag = (earliest.release.dt.normalize() - pd.to_datetime(earliest.reference_time, utc=True)).dt.days
        recent = (earliest.release >= '2025-08-01') & (earliest.release < '2026-08-01')
        operational = lag[recent & lag.between(0, 60)]
        first[name_] = dict(first_release=None if subset.empty else subset.release.min().isoformat(),
            recent_first_releases=int(recent.sum()), operational_first_releases=len(operational),
            first_release_median_days=None if operational.empty else float(operational.median()),
            first_release_p10_days=None if operational.empty else float(operational.quantile(.1)),
            first_release_p90_days=None if operational.empty else float(operational.quantile(.9)),
            excluded_backfill_over60days=int((recent & (lag > 60)).sum()))
    for w, issuance in enumerate(issuances):
        cutoff = pd.Timestamp(deadline(issuance))
        end = (date.fromisoformat(issuance) - timedelta(days=4)).isoformat()
        eligible = frame.loc[(frame.release <= cutoff) & (frame.reference_time <= end)]
        eligible = eligible.drop_duplicates(['reference_time', 'location', 'name'], keep='last')
        t, l, n = eligible.reference_time.map(ti), eligible.location.map(li), eligible.name.map(ni)
        keep = t.notna() & l.notna() & n.notna()
        result[w, t[keep].astype(int), l[keep].astype(int), n[keep].astype(int)] = eligible.value[keep]
        reported[w, t[keep].astype(int), l[keep].astype(int), n[keep].astype(int)] = True
    return name, names, result, reported, first, round(timing.monotonic() - started, 2)


def table(hub, commit, path):
    """Historical file with explicit null statements retained (separate cache)."""
    if not commit:
        return None, None
    blob = git(hub, 'rev-parse', f'{commit}:{path}', check=False).decode().strip()
    if len(blob) != 40 or ':' in blob:
        return None, None
    cache = Path('data/audits/b2/deadline-cache') / f'{blob}.pkl'
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return pd.read_pickle(cache), blob
    content = git(hub, 'show', f'{commit}:{path}')
    raw = pd.read_parquet(io.BytesIO(content)) if path.endswith('parquet') else pd.read_csv(io.BytesIO(content), dtype={'location':str}, low_memory=False)
    if 'target_end_date' not in raw and 'date' in raw:
        raw = raw.rename(columns={'date':'target_end_date'})
    if path == 'target-data/target-hospital-admissions.csv':
        raw = raw.rename(columns={'value':'observation'}).assign(target='wk inc flu hosp')
    if path == 'target-data/target-ed-visits-prop.csv':
        raw = raw.rename(columns={'value':'observation'}).assign(target='wk inc flu prop ed visits')
    if 'as_of' in raw:
        raw = raw.sort_values('as_of').drop_duplicates(['target_end_date','location','target'], keep='last')
    if 'percent_visits_covid' in raw:
        raw = raw.loc[raw['county'].astype(str).str.lower().eq('all')].copy()
        raw['location'] = raw.geography.map(geos)
        raw['week'] = pd.to_datetime(raw.week_end).dt.strftime('%Y-%m-%d')
        frames = []
        for pathogen, col in [('flu','percent_visits_influenza'),('covid','percent_visits_covid'),('rsv','percent_visits_rsv')]:
            if col not in raw:
                continue
            part = raw[['week','location']].copy()
            part['target'] = f'wk inc {pathogen} prop ed visits'
            part['value'] = pd.to_numeric(raw[col], errors='coerce') / 100
            frames.append(part)
        out = pd.concat(frames, ignore_index=True)
    elif {'target_end_date','target','observation','location'} <= set(raw):
        out = pd.DataFrame(dict(week=pd.to_datetime(raw.target_end_date).dt.strftime('%Y-%m-%d'), location=raw.location.astype(str).map(fips), target=raw.target.astype(str), value=pd.to_numeric(raw.observation, errors='coerce')))
    else:
        raise ValueError((hub, path, raw.columns.tolist()))
    out = out.loc[out.location.notna() & out.target.isin(CHANNELS)].drop_duplicates(['week','location','target'])
    out['value'] = out.value.where(np.isfinite(out.value) & (out.value >= 0))
    out.to_pickle(cache)
    return out, blob


@lru_cache(maxsize=None)
def file_release(hub, commit, path):
    """Latest first-parent path-change time, bounded by historical tree state."""
    value = git(hub, 'log', '--first-parent', '-1', '--format=%cI', commit, '--', path).decode().strip()
    return pd.Timestamp(value)


def target_inputs(panel, report_dir):
    dates = panel['dates'].astype(str).tolist()
    locations = panel['locations'].astype(str).tolist()
    issuances = panel['issuance_dates'].astype(str).tolist()
    ti, li, ci = ({v: i for i, v in enumerate(vals)} for vals in (dates, locations, CHANNELS))
    # Preserve explicit latest nulls in archive resolution; these may retract a
    # prior value. Older values are never filled into a latest-null cell.
    delphi = delphi_reports().sort_values('release', kind='stable')
    result = np.full_like(panel['asof_targets'], np.nan)
    reported = np.zeros(result.shape, bool)
    source_rows, coverage = [], []
    for w, issuance in enumerate(issuances):
        cutoff = pd.Timestamp(deadline(issuance))
        end = (date.fromisoformat(issuance) - timedelta(days=4)).isoformat()
        context = set(d for d in dates if d <= end)
        candidates = []
        for hub in HUBS:
            commit = state(hub, cutoff.tz_convert('UTC').isoformat())
            paths = ('target-data/time-series.csv', 'target-data/target-hospital-admissions.csv', 'target-data/target-ed-visits-prop.csv') if hub == 'flusight' else ('target-data/time-series.parquet', 'auxiliary-data/nssp-raw-data/latest.parquet', 'auxiliary-data/nssp-raw-data/latest.csv')
            for path in paths:
                frame, blob = table(hub, commit, path)
                if frame is None:
                    continue
                published = file_release(hub, commit, path)
                if published > cutoff:
                    raise ValueError(f'Future publication {hub} {commit} {path}: {published} > {cutoff}')
                selected = frame.loc[frame.week.isin(context)].copy()
                selected['release'] = published
                selected['source'] = f'{hub}:{path}'
                candidates.append(selected)
                source_rows.append(dict(issuance=issuance, cutoff_utc=cutoff.tz_convert('UTC').isoformat(), hub=hub, commit=commit, path=path, blob=blob, file_publication=published.isoformat()))
        eligible = delphi.loc[delphi.week.isin(context) & (delphi.release <= cutoff)].copy()
        eligible = eligible.drop_duplicates(['week', 'location', 'target'], keep='last')
        eligible['source'] = 'delphi'
        candidates.append(eligible)
        merged = pd.concat(candidates, ignore_index=True).sort_values('release', kind='stable')
        # Every Hub file is its actual historical state. The latest public
        # statement among extant public sources wins per observation. Null
        # Delphi statements remain null; no retrospective numerical values used.
        latest = merged.drop_duplicates(['week', 'location', 'target'], keep='last')
        t, l, c = latest.week.map(ti), latest.location.map(li), latest.target.map(ci)
        keep = t.notna() & l.notna() & c.notna()
        values = pd.to_numeric(latest.value, errors='coerce')
        values = values.where(np.isfinite(values) & (values >= 0))
        result[w, t[keep].astype(int), l[keep].astype(int), c[keep].astype(int)] = values[keep]
        reported[w, t[keep].astype(int), l[keep].astype(int), c[keep].astype(int)] = True
        recent = [ti[d] for d in dates if max(dates[0], (date.fromisoformat(end) - timedelta(weeks=11)).isoformat()) <= d <= end]
        for channel, name in enumerate(panel['target_names']):
            coverage.append(dict(issuance=issuance, season=season(end), target=str(name), context_present=int(np.isfinite(result[w, recent, :, channel]).sum()), context_cells=len(recent)*len(locations), latest_present=int(np.isfinite(result[w, ti[end], :, channel]).sum()) if end in ti else 0))
        if w % 10 == 0:
            print(f'target deadline {w + 1}/{len(issuances)}: {issuance}', flush=True)
    pd.DataFrame(source_rows).to_csv(report_dir / 'target-source-versions.csv', index=False)
    pd.DataFrame(coverage).to_csv(report_dir / 'target-coverage.csv', index=False)
    return result, reported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/processed/panel.npz')
    parser.add_argument('--output', default='data/processed/panel-b2-deadline.npz')
    parser.add_argument('--report-dir', default='docs/data/availability/provenance')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    started = timing.monotonic()
    panel = load(args.source)
    frozen = {n: panel[n].copy() for n in ('targets', 'covariates', 'covariates_national')}
    dates, issuances, locations = (panel[n].astype(str).tolist() for n in ('dates', 'issuance_dates', 'locations'))
    sources = [*ex.DELPHI_COVARIATES, 'nwss', 'kinsa_ili']
    tasks = [(name, 'data', dates, issuances, locations) for name in sources]
    source_details = {}
    panel['known_report_covariates'] = np.zeros(panel['asof_covariates'].shape, bool)
    panel['known_report_covariates_national'] = np.zeros(panel['asof_covariates_national'].shape, bool)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(covariate_worker, task) for task in tasks]
        # Independent Git reconstruction proceeds while the covariate archives load.
        panel['asof_targets'], panel['known_report_targets'] = target_inputs(panel, report_dir)
        for future in as_completed(futures):
            name, names, values, reported, first, seconds = future.result()
            for c, cov in enumerate(names):
                if cov in panel['covariate_names']:
                    out_c = panel['covariate_names'].tolist().index(cov)
                    panel['asof_covariates'][..., out_c] = values[..., c]
                    panel['known_report_covariates'][..., out_c] = reported[..., c]
                else:
                    out_c = panel['covariate_national_names'].tolist().index(cov)
                    panel['asof_covariates_national'][..., out_c] = values[:, :, locations.index('US'), c]
                    panel['known_report_covariates_national'][..., out_c] = reported[:, :, locations.index('US'), c]
            source_details[name] = dict(first_release=first, seconds=seconds)
            print(f'covariate {name}: {seconds}s; first releases {first}', flush=True)
    panel['forecast_cutoff_utc'] = np.array([deadline(i).astimezone(timezone.utc).isoformat() for i in issuances])
    metadata = json.loads(str(panel['metadata']))
    metadata.update(deadline_build=dict(source=args.source, source_sha256=sha256(args.source), builder_sha256=sha256(__file__), covariate_snapshots={dataset: ex._latest_snapshot('data', dataset).name for dataset in sorted({spec[0] for spec in ex.DELPHI_COVARIATES.values()} | {ex.NWSS_DERIVED_DATASET, ex.KINSA_DATASET})}, holiday_dates=HOLIDAYS, holiday_evidence=HOLIDAY_EVIDENCE, target_policy='Actual historical Hub files plus Delphi reports; latest public statement per observation among extant source files. Missing public archive cells remain missing; finalized truth is never substituted.', covariate_policy='Latest native report per observation by the common deadline; all midnight/date-only releases conservatively placed at end of their UTC day. No future value substitution or imputed source lag.', deadline_policy='Wednesday 23:00 America/New_York except documented extensions; earliest participating Hub deadline for joint issuance. Wednesday IDs and preceding Saturday context remain fixed.', source_details=source_details, limitation='Public archive availability is a lower bound on upstream availability. Kinsa first public release is April 2026; NWSS vintage support also starts in 2026. A source deletion in one public Hub does not retract a value still present in another public Hub.', truth_policy='All finalized target and covariate truth arrays retained bit-for-bit from frozen source panel.', seconds=round(timing.monotonic()-started, 2)))
    panel['metadata'] = json.dumps(metadata)
    for name, original in frozen.items():
        np.testing.assert_equal(panel[name], original)
    save(panel, args.output)
    coverage = []
    for w, issuance in enumerate(issuances):
        end = (date.fromisoformat(issuance)-timedelta(days=4)).isoformat()
        recent = [i for i, d in enumerate(dates) if (date.fromisoformat(end)-timedelta(weeks=11)).isoformat() <= d <= end]
        for key, names_key in [('asof_covariates','covariate_names'), ('asof_covariates_national','covariate_national_names')]:
            for c, name in enumerate(panel[names_key]):
                x = panel[key][w, recent, ..., c]
                coverage.append(dict(issuance=issuance, season=season(end), covariate=str(name), present=int(np.isfinite(x).sum()), cells=x.size))
    pd.DataFrame(coverage).to_csv(report_dir/'covariate-coverage.csv',index=False)
    record = dict(metadata['deadline_build'], output=args.output, output_sha256=sha256(args.output), truth_arrays_unchanged=True)
    (report_dir/'deadline-policy.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
