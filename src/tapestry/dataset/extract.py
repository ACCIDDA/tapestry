"""One extraction function, two axes: target vs. covariate, vintaged vs. finalized.

`extract` is the single read path for both array builders (`dataset.build`) and
ad-hoc notebook/analysis code (docs/design/restructure-2026-unified.md §2). It
wraps the same vintage-resolution policy the project has always used (native
Hub/Git full-snapshot coverage first, Delphi report-time revisions outside that
coverage, explicit nulls on same-release conflicts) behind one signature instead
of a bespoke read path per dataset.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from tapestry.data.geography import STATE_NAMES, observation_geography
from tapestry.data.selection import SelectedData, describe, source_signal, _timestamp

# Target channels: NHSN admissions (Hub finals + Delphi fallback) and NSSP ED
# proportions (Delphi only; NSSP has no finality flag, so "finalized" for it
# means the latest reported unsmoothed snapshot).
TARGET_SOURCES = {
    'nhsn_flu_admissions': ('hub_flusight_current', 'delphi_nhsn', 'confirmed_admissions_flu_ew', 'totalconfflunewadm'),
    'nhsn_covid_admissions': ('hub_covid_current', 'delphi_nhsn', 'confirmed_admissions_covid_ew', 'totalconfc19newadm'),
    'nhsn_rsv_admissions': ('hub_rsv_current', 'delphi_nhsn', 'confirmed_admissions_rsv_ew', 'totalconfrsvnewadm'),
    'nssp_flu_proportion': (None, 'delphi_nssp', 'pct_ed_visits_influenza', 'percent_visits_influenza'),
    'nssp_covid_proportion': (None, 'delphi_nssp', 'pct_ed_visits_covid', 'percent_visits_covid'),
    'nssp_rsv_proportion': (None, 'delphi_nssp', 'pct_ed_visits_rsv', 'percent_visits_rsv'),
}
CLAIMS_SOURCES = {
    'inpatient_flu': ('delphi_claims_inpatient', 'claims_inpatient_adm_pct_claims_flu'),
    'inpatient_covid': ('delphi_claims_inpatient', 'claims_inpatient_adm_pct_claims_covid'),
    'outpatient_flu': ('delphi_claims_outpatient', 'claims_outpatient_ov_pct_claims_flu'),
    'outpatient_covid': ('delphi_claims_outpatient', 'claims_outpatient_ov_pct_claims_covid'),
}
NWSS_INDICES = ('nwss_flu_wval_like', 'nwss_covid_wval_like', 'nwss_rsv_wval_like',
                'nwss_flu_pct_rank', 'nwss_covid_pct_rank', 'nwss_rsv_pct_rank')
NWSS_DERIVED_DATASET = 'derived_nwss_state_indices'
KINSA_DATASET = 'pophive_kinsa_ili'
KINSA_SIGNAL = 'kinsa_cough_cold_flu'
COVARIATE_SOURCES = tuple(CLAIMS_SOURCES) + NWSS_INDICES + ('kinsa_ili',)
NATIONAL_ONLY = ('kinsa_ili',)


def saturday(value):
    day = date.fromisoformat(str(value)[:10])
    if day.weekday() != 5:
        raise ValueError(f'Expected a Saturday week end: {value}')
    return day


def _release_time(value):
    return _timestamp(str(value)).isoformat()


def cutoff_time(day):
    return _release_time(str(day) + 'T23:59:59.999999+00:00')


class VintageArchive:
    """Resolve native Hub, Git Hub, and per-observation Delphi vintages.

    Coverage is conservative: once a channel/location appears, its weekly period
    from the earliest event onward is covered, even if later snapshots omit rows.
    Native as_of coverage precedes Git coverage, which precedes Delphi.
    """
    def __init__(self):
        self.hub = {}
        self.delphi = {}
        self.provenance = [dict(source='none', release=None, snapshot_id=None)]
        self._provenance_ids = {}
        self.manifests = []

    def provenance_id(self, **record):
        record['fallback_reason'] = ('absent_hub_historical_coverage'
                                      if record['source'].startswith('delphi_') else None)
        key = tuple(sorted(record.items()))
        if key not in self._provenance_ids:
            self._provenance_ids[key] = len(self.provenance)
            self.provenance.append(record)
        return self._provenance_ids[key]

    def add(self, source, release, day, location, value, snapshot_id='', source_path=''):
        release = _release_time(release)
        saturday(day)
        pid = self.provenance_id(source=source, release=release, snapshot_id=snapshot_id, source_path=source_path)
        cell = (day, location)
        if source.startswith('hub_'):
            rows = self.hub.setdefault(source, {}).setdefault(release, {})
            key = cell
        else:
            rows = self.delphi.setdefault(cell, {})
            key = release
        if key in rows and rows[key][0] != value:
            value = None
        rows[key] = (value, pid)

    def resolve(self, cutoff):
        cutoff = cutoff_time(cutoff)
        hub, coverage, latest = {}, {}, {}
        git, git_coverage, git_latest = {}, {}, {}
        for source, releases in self.hub.items():
            eligible = sorted(r for r in releases if r <= cutoff)
            if not eligible:
                continue
            selected = eligible[-1]
            is_git = source.endswith(':git')
            values, covered, provenance = (git, git_coverage, git_latest) if is_git else (hub, coverage, latest)
            base = source.removesuffix(':git')
            rows = releases[selected]
            provenance[base] = (next(iter(rows.values()))[1] if rows else
                                 self.provenance_id(source=source, release=selected, source_path='empty full snapshot'))
            for release in eligible:
                for day, loc in releases[release]:
                    key = loc
                    covered[key] = min(day, covered.get(key, day))
            values.update(rows)
        delphi = {}
        for cell, revisions in self.delphi.items():
            eligible = [r for r in revisions if r <= cutoff]
            if eligible:
                delphi[cell] = revisions[max(eligible)]
        return hub, coverage, latest, delphi, git, git_coverage, git_latest

    def panel(self, dates, locations, state, start=None):
        hub, coverage, latest, delphi, git, git_coverage, git_latest = state
        values = np.zeros((len(dates), len(locations)), np.float32)
        available = np.zeros_like(values, dtype=bool)
        for t, day in enumerate(dates):
            for l, loc in enumerate(locations):
                if start and day < start:
                    continue
                cell = day, loc
                covered = loc in coverage and day >= coverage[loc]
                if covered:
                    value, _ = hub.get(cell, (None, 0))
                elif loc in git_coverage and day >= git_coverage[loc]:
                    value, _ = git.get(cell, (None, 0))
                else:
                    value, _ = delphi.get(cell, (None, 0))
                if value is not None:
                    values[t, l], available[t, l] = value, True
        return values, available


def _archive_for(data_root, hub_dataset, delphi_dataset, delphi_signal, hub_origin, upper=None):
    archive = VintageArchive()
    selected = SelectedData(data_root)
    for key in filter(None, (hub_dataset, delphi_dataset)):
        for table in selected.selected_tables(dataset_key=key):
            is_git = getattr(table, 'source_path', '') == 'git-history.ndjson.gz'
            source_key = key + ':git' if is_git else key
            if is_git:
                for release in table.release_times:
                    archive.hub.setdefault(source_key, {}).setdefault(_release_time(release), {})
            if key == delphi_dataset:
                signal = source_signal(table.source_path)
                if signal and signal != delphi_signal:
                    continue
            for row in table.iter_rows():
                if key == delphi_dataset:
                    row_signal = source_signal(table.source_path, row)
                    if row_signal != delphi_signal:
                        continue
                    if str(row.get('fill_method', 'none')).lower() not in ('source', 'none', '', 'null'):
                        continue
                else:
                    origin = describe(key, 'observation', table.source_path, row)['origin_column']
                    if origin != hub_origin:
                        continue
                loc = observation_geography(row, table.geographic_resolutions)
                if loc not in STATE_NAMES and loc != 'US':
                    continue
                release = row.get(table.vintage_column)
                if not release:
                    continue
                day = saturday(row.get(table.event_date_column) or row.get('target_end_date') or row.get('date')).isoformat()
                raw = row.get('observation' if key == hub_dataset else 'value')
                try:
                    value = float(raw)
                    cap = (1 if key == hub_dataset else 100) if upper else float('inf')
                    if not np.isfinite(value) or not 0 <= value <= cap:
                        value = None
                except (TypeError, ValueError):
                    value = None
                if upper and key == delphi_dataset and value is not None:
                    value /= 100
                archive.add(source_key, release, day, loc, value, table.snapshot_id, table.source_path)
    return archive


def _target_archive(data_root, name):
    hub_dataset, delphi_dataset, delphi_signal, hub_origin = TARGET_SOURCES[name]
    upper = name.startswith('nssp_') or name.startswith('nhsn_') and False
    return _archive_for(data_root, hub_dataset, delphi_dataset, delphi_signal, hub_origin,
                         upper=name.startswith('nssp_'))


def _claims_archive(data_root, name):
    dataset, signal = CLAIMS_SOURCES[name]
    return _archive_for(data_root, None, dataset, signal, None, upper=False)


def _latest_snapshot(data_root, dataset):
    root = Path(data_root) / 'raw' / dataset
    latest = root / 'latest.json'
    if not latest.is_file():
        raise FileNotFoundError(f'Missing selected raw dataset: {latest}')
    import json
    record = json.loads(latest.read_text())
    snapshot_id = record.get('snapshot_id') or record.get('id')
    if not snapshot_id:
        manifest = record.get('manifest') or record.get('path')
        if manifest:
            snapshot_id = Path(manifest).parent.name
    if not snapshot_id:
        raise ValueError(f'Cannot resolve snapshot id from {latest}')
    return root / 'snapshots' / snapshot_id


def _nwss_frame(data_root):
    path = _latest_snapshot(data_root, NWSS_DERIVED_DATASET) / 'data.csv.gz'
    if not path.is_file():
        raise FileNotFoundError(f'Registered NWSS state index payload is missing: {path}')
    frame = pd.read_csv(path)
    for column in ('report_time', 'reference_time'):
        frame[column] = frame[column].astype(str).str[:10]
    frame['geo_value'] = frame['geo_value'].astype(str).str.upper()
    frame['pathogen'] = frame['pathogen'].astype(str).str.lower()
    return frame


def _kinsa_frame(data_root):
    from tapestry.dataset.build import _kinsa_weekly
    weekly, _ = _kinsa_weekly(data_root, date.today().isoformat())
    return weekly


def _resolve_series(archive, dates, locations, version, as_of):
    """Common vintaged/finalized resolution shared by every archive-backed source."""
    if version == 'finalized':
        cutoff = as_of.isoformat() if as_of else date.today().isoformat()
        values, available = archive.panel(dates, locations, archive.resolve(cutoff))
        rows = [dict(date=d, location=l, value=float(values[i, j]))
                for i, d in enumerate(dates) for j, l in enumerate(locations) if available[i, j]]
        return pd.DataFrame(rows, columns=['date', 'location', 'value'])
    if as_of is not None:
        values, available = archive.panel(dates, locations, archive.resolve(as_of.isoformat()))
        rows = [dict(date=d, location=l, value=float(values[i, j]))
                for i, d in enumerate(dates) for j, l in enumerate(locations) if available[i, j]]
        return pd.DataFrame(rows, columns=['date', 'location', 'value'])
    # Every historical Wednesday issuance: the shape the array builders need.
    first = min(dates) if dates else date.today().isoformat()
    issuances = _wednesdays_through_today(first)
    frames = []
    for issuance in issuances:
        values, available = archive.panel(dates, locations, archive.resolve(issuance))
        for i, d in enumerate(dates):
            for j, l in enumerate(locations):
                if available[i, j]:
                    frames.append((issuance, d, l, float(values[i, j])))
    return pd.DataFrame(frames, columns=['issuance', 'date', 'location', 'value'])


def _wednesdays_through_today(first):
    day = date.fromisoformat(first[:10])
    day -= timedelta(days=(day.weekday() - 2) % 7)  # roll back to a Wednesday
    today = date.today()
    out = []
    while day <= today:
        out.append(day.isoformat())
        day += timedelta(weeks=1)
    return out


def extract(name: str, kind: Literal['target', 'covariate'], version: Literal['vintaged', 'finalized'],
            as_of: date | None = None, *, data_root: str = 'data', dates=None, locations=None) -> pd.DataFrame:
    """The one read path for a single source, at one vintage policy.

    `dates`/`locations` restrict which cells are resolved (required for archive
    lookups); when omitted the caller gets whatever the source naturally covers.
    Returns a long frame with columns `date`, `location`, `value` (`finalized`,
    or `vintaged` with an explicit `as_of`), or additionally `issuance`
    (`vintaged` with `as_of=None`: every historical Wednesday).
    """
    if kind not in ('target', 'covariate'):
        raise ValueError(f'Unknown extract kind: {kind}')
    if version not in ('vintaged', 'finalized'):
        raise ValueError(f'Unknown extract version: {version}')
    dates = list(dates) if dates is not None else []
    locations = list(locations) if locations is not None else (*sorted(STATE_NAMES), 'US')
    if kind == 'target':
        if name not in TARGET_SOURCES:
            raise ValueError(f'Unknown target: {name}')
        archive = _target_archive(data_root, name)
        return _resolve_series(archive, dates, locations, version, as_of)
    if name in CLAIMS_SOURCES:
        archive = _claims_archive(data_root, name)
        return _resolve_series(archive, dates, locations, version, as_of)
    if name in NWSS_INDICES:
        pathogen, metric = name.removeprefix('nwss_').rsplit('_', 1) if name.endswith('rank') else (
            name.removeprefix('nwss_').removesuffix('_wval_like'), 'wval_like')
        metric = 'pct_rank' if name.endswith('pct_rank') else 'wval_like'
        frame = _nwss_frame(data_root)
        frame = frame[frame.pathogen.eq(pathogen)][['report_time', 'geo_value', 'reference_time', metric]]
        frame = frame.rename(columns={'reference_time': 'date', 'geo_value': 'location', metric: 'value'})
        frame['value'] = pd.to_numeric(frame['value'], errors='coerce')
        cutoff = (as_of.isoformat() if as_of else date.today().isoformat())
        if version == 'finalized' or as_of is not None:
            frame = frame[frame.report_time <= cutoff].sort_values('report_time')
            frame = frame.dropna(subset=['value']).drop_duplicates(['location', 'date'], keep='last')
            return frame[['date', 'location', 'value']].reset_index(drop=True)
        frame = frame.dropna(subset=['value'])
        frame = frame.rename(columns={'report_time': 'issuance'})
        return frame[['issuance', 'date', 'location', 'value']].reset_index(drop=True)
    if name == 'kinsa_ili':
        frame = _kinsa_frame(data_root).rename(columns={'reference_time': 'date', 'geo_value': 'location'})
        cutoff = (as_of.isoformat() if as_of else date.today().isoformat())
        if version == 'finalized' or as_of is not None:
            frame = frame[frame.report_time <= cutoff].sort_values('report_time')
            frame = frame.drop_duplicates(['location', 'date'], keep='last')
            return frame[['date', 'location', 'value']].reset_index(drop=True)
        frame = frame.rename(columns={'report_time': 'issuance'})
        return frame[['issuance', 'date', 'location', 'value']].reset_index(drop=True)
    raise ValueError(f'Unknown covariate: {name}')
