"""Read each dataset source once into a revision table, then resolve it as of any cutoff.

The single read path for the dataset builder (`dataset.build`) and ad-hoc analysis
(docs/design/restructure-2026-unified.md §2). Vintage policy, unchanged from the
row-by-row `VintageArchive` it replaces (2026-09-22, rewritten for speed only):

- Archive-backed sources (six targets, four Delphi claims covariates) carry three
  tiers of revisions: `hub` (the Hub's native `as_of` full snapshots), `git` (Hub
  Git-history full snapshots, each registered release is a full snapshot even when
  empty for this target) and `delphi` (per-observation Delphi report-time revisions).
- As of a cutoff, a location is covered by a Hub tier from the earliest reference
  week that appears in any eligible release of that tier. Covered cells take the
  value of the latest eligible full snapshot (absent there = unavailable). Native
  `hub` coverage precedes `git`, which precedes `delphi`. Outside Hub coverage a
  cell takes its latest eligible Delphi revision (a missing/invalid latest revision
  is unavailable; earlier revisions are not used as fallback).
- Rows released at the same time for the same cell with different values
  (including one missing and one present) are unavailable, never resolved by
  file order.
- A release is visible at a cutoff day if released by 23:59:59.999999 UTC that day.
- Values must be finite and nonnegative; NSSP percentages must be at most 100 and
  are divided by 100. Claims archives are daily: only Saturday reference days are
  kept, as the pre-restructure builder did (`model_data/b2.py:_claims_frame`), with
  no weekly averaging. Delphi rows with a non-native `fill_method` are dropped.
- NWSS indices and Kinsa (report-date files, not archives) use the latest report
  dated on or before the cutoff day, after dropping missing values.

Tables are read columnar (pyarrow), and every per-row policy function
(`observation_geography`, `describe`) is evaluated once per unique combination of
the columns it reads, so results equal the row-by-row path.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.json as pajson
import pyarrow.parquet as paparquet

from tapestry.data.geography import STATE_NAMES, observation_geography
from tapestry.data.selection import SelectedData, describe, selected_row

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
NATIONAL_ONLY = ('kinsa_ili',)
ARCHIVE_SOURCES = tuple(TARGET_SOURCES) + tuple(CLAIMS_SOURCES)
TIERS = ('hub', 'git', 'delphi')
LOCATIONS = (*sorted(STATE_NAMES), 'US')

# Columns read by the per-row policy functions; each is evaluated per unique combination.
GEOGRAPHY_COLUMNS = ('county', 'county_fips', 'county_name', 'hsa', 'hsa_nci_id', 'site', 'site_id', 'sewershed',
                     'sewershed_id', 'wwtp_id', 'geo_type', 'geography_type', 'geographic_level', 'level',
                     'jurisdiction', 'state', 'state_code', 'state_abbr', 'state_abbreviation', 'state_territory',
                     'geography', 'geo_value', 'location', 'location_code', 'location_name', 'fips', 'area', 'region')
ORIGIN_COLUMNS = ('target', 'target_variable', 'signal', 'pathogen', 'pathogen_target')
NATIVE_FILL = ('source', 'none', '', 'null')


def cutoff_time(day):
    """End of UTC day `day` (a date or 'YYYY-MM-DD'); anything released that day is visible.

    Accepts only a calendar day, never an already-converted timestamp, so a cutoff
    cannot be converted twice.
    """
    if isinstance(day, datetime) or not isinstance(day, (date, str)):
        raise TypeError(f'Cutoff must be a calendar day, not {day!r}')
    day = day if isinstance(day, date) else date.fromisoformat(day)
    return np.datetime64(f'{day.isoformat()}T23:59:59.999999', 'ns')  # UTC, like every stored release


def _releases(values):
    """Release strings -> naive-UTC datetime64[ns] (naive strings are UTC, as `selection._timestamp`)."""
    stamps = pd.to_datetime(pd.Series(values, dtype=object), utc=True, format='ISO8601')
    return stamps.dt.tz_localize(None).to_numpy('datetime64[ns]')


def _number(raw):
    try:
        return float(raw)
    except (TypeError, ValueError):
        return np.nan


def _numbers(values):
    """Python `float()` semantics per unique raw value; missing or unparsable -> NaN."""
    codes, unique = pd.factorize(pd.Series(values, dtype=object))
    parsed = np.array([_number(u) for u in unique] + [np.nan], dtype=float)
    return parsed[codes]  # code -1 (missing) picks the trailing NaN


def _per_unique(frame, columns, function):
    """Evaluate `function(row_dict)` once per unique combination of `columns` in `frame`.

    Present columns are passed with None for missing cells, as the row-by-row readers did.
    """
    present = [c for c in columns if c in frame]
    if not present:
        return np.full(len(frame), function({}), dtype=object)
    keys = frame[present].astype(object)
    keys = keys.where(keys.notna(), None)
    ids = keys.groupby(present, dropna=False, sort=False).ngroup().to_numpy()
    first = pd.Series(np.arange(len(frame))).groupby(ids).first().sort_index().to_numpy()
    results = np.empty(len(first), dtype=object)
    results[:] = [function(dict(zip(present, row))) for row in keys.iloc[first].itertuples(index=False)]
    return results[ids]


def _read(path, name, columns):
    """Yield string-typed pandas batches of `columns` (missing columns absent) from one table file."""
    name = name.lower()
    if name.endswith('.parquet'):
        table = paparquet.read_table(path)
        yield table.select([c for c in columns if c in table.column_names]).to_pandas()
        return
    if name.endswith(('.ndjson', '.ndjson.gz', '.jsonl', '.jsonl.gz')):
        stream = pa.input_stream(str(path), compression='gzip' if name.endswith('.gz') else None)
        if not stream.read(1).strip():
            return  # an empty history (no rows); its registered releases still count
        stream = pa.input_stream(str(path), compression='gzip' if name.endswith('.gz') else None)
        table = pajson.read_json(stream, read_options=pajson.ReadOptions(block_size=1 << 26))
        table = table.select([c for c in columns if c in table.column_names])
        table = pa.table({c: table[c].cast(pa.string()) for c in table.column_names})
        yield table.to_pandas()
        return
    header = pacsv.open_csv(str(path)).schema.names
    present = [c for c in columns if c in header]
    reader = pacsv.open_csv(str(path), read_options=pacsv.ReadOptions(block_size=1 << 26),
                            convert_options=pacsv.ConvertOptions(include_columns=present,
                                                                 column_types={c: pa.string() for c in present},
                                                                 strings_can_be_null=False, quoted_strings_can_be_null=False))
    for batch in reader:
        yield batch.to_pandas()


@dataclass
class Revisions:
    """Every eligible revision of one source: tier, release (UTC), day (Saturday ISO), location, value.

    `value` is NaN for unavailable (invalid, missing or same-release conflict).
    `git_releases` are all Git-history full-snapshot times, including empty ones.
    Rows are sorted by (day, location, release).
    """
    rows: pd.DataFrame
    git_releases: np.ndarray

    def __post_init__(self):
        self.tiers = {tier: self.rows[self.rows.tier.eq(tier)] for tier in TIERS}


def revisions(name, data_root='data'):
    """Read one archive-backed source (a target or a claims covariate) into `Revisions`."""
    if name in TARGET_SOURCES:
        hub_dataset, delphi_dataset, delphi_signal, hub_origin = TARGET_SOURCES[name]
        upper, saturdays_only = name.startswith('nssp_'), False
    elif name in CLAIMS_SOURCES:
        (delphi_dataset, delphi_signal), hub_dataset, hub_origin = CLAIMS_SOURCES[name], None, None
        upper, saturdays_only = False, True
    else:
        raise ValueError(f'Not an archive-backed source: {name}')
    if not selected_row(delphi_dataset, f'signal={delphi_signal}', {}):
        raise ValueError(f'{delphi_signal} is not a selected {delphi_dataset} signal')
    parts, git_releases = [], []
    selected = SelectedData(data_root)
    for key in filter(None, (hub_dataset, delphi_dataset)):
        hub = key == hub_dataset
        for table in selected.selected_tables(dataset_key=key):
            is_git = table.source_path == 'git-history.ndjson.gz'
            if is_git:
                git_releases.extend(table.release_times)
            path_signal = re.search(r'(?:^|/)signal=([^/]+)', table.source_path)
            path_signal = path_signal.group(1) if path_signal else ''
            if not hub and path_signal and path_signal != delphi_signal:
                continue
            path_geo = re.search(r'(?:^|/)geo_type=([^/]+)', table.source_path)
            path_geo = path_geo.group(1).lower() if path_geo else None
            day_columns = [c for c in (table.event_date_column, 'target_end_date', 'date') if c]
            value_column = 'observation' if hub else 'value'
            columns = {*GEOGRAPHY_COLUMNS, *ORIGIN_COLUMNS, *day_columns, table.vintage_column,
                       value_column, 'fill_method'}
            for frame in _read(table.path, table.source_path.split('!/')[-1], sorted(columns)):
                if not hub:
                    signal = frame['signal'] if 'signal' in frame else pd.Series('', index=frame.index)
                    signal = signal.fillna('').astype(str).where(lambda s: s != '', path_signal)
                    keep = signal.eq(delphi_signal)
                    if 'fill_method' in frame:
                        keep &= frame['fill_method'].fillna('None').astype(str).str.lower().isin(NATIVE_FILL)
                    frame = frame[keep]
                day = pd.Series('', index=frame.index, dtype=object)
                for column in reversed(day_columns):
                    if column in frame:
                        value = frame[column].astype(object)
                        day = value.where(value.notna() & (value.astype(str) != ''), day)
                release = frame[table.vintage_column] if table.vintage_column in frame else pd.Series(None, index=frame.index)
                keep = release.notna() & (release.astype(str) != '')
                if saturdays_only:
                    keep &= pd.to_datetime(day.astype(str).str[:10], errors='coerce').dt.weekday.eq(5)
                frame, day, release = frame[keep], day[keep], release[keep]
                if frame.empty:
                    continue
                if hub:
                    origin = _per_unique(frame, ORIGIN_COLUMNS,
                                         lambda row: describe(key, 'observation', table.source_path, row)['origin_column'])
                    frame, day, release = frame[origin == hub_origin], day[origin == hub_origin], release[origin == hub_origin]
                resolutions = table.geographic_resolutions
                location = _per_unique(frame, GEOGRAPHY_COLUMNS, lambda row: (
                    observation_geography(row, resolutions, path_geo) and observation_geography(row, resolutions)))
                keep = pd.Series(location, index=frame.index).isin([*STATE_NAMES, 'US']).to_numpy()
                if not keep.any():
                    continue
                frame, day, release, location = frame[keep], day[keep], release[keep], location[keep]
                days = day.astype(str).str[:10]
                weekdays = pd.to_datetime(days, format='%Y-%m-%d').dt.weekday
                if not weekdays.eq(5).all():
                    raise ValueError(f'Expected Saturday week ends in {key} {table.source_path}: '
                                     f'{sorted(set(days[weekdays != 5]))[:5]}')
                values = _numbers(frame[value_column]) if value_column in frame else np.full(len(frame), np.nan)
                cap = (1 if hub else 100) if upper else np.inf
                values[~np.isfinite(values) | (values < 0) | (values > cap)] = np.nan
                if upper and not hub:
                    values = values / 100
                parts.append(pd.DataFrame(dict(
                    tier='git' if is_git else 'hub' if hub else 'delphi', release=_releases(release.to_numpy()),
                    day=days.to_numpy(), location=location.astype(str), value=values)))
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        dict(tier=[], release=np.array([], 'datetime64[ns]'), day=[], location=[], value=[]))
    return Revisions(_without_conflicts(rows), np.sort(_releases(git_releases)))


def _without_conflicts(rows):
    """One row per (tier, release, day, location); disagreeing values (NaN included) -> NaN."""
    keys = ['tier', 'release', 'day', 'location']
    grouped = rows.groupby(keys, sort=False).value
    summary = grouped.agg(['min', 'max', 'count', 'size']).reset_index()
    agree = (summary['count'].eq(0) | (summary['count'].eq(summary['size']) & summary['min'].eq(summary['max'])))
    summary['value'] = summary['min'].where(agree)
    return summary[keys + ['value']].sort_values(['day', 'location', 'release'], kind='stable').reset_index(drop=True)


def resolve(revisions_, day, dates, locations=LOCATIONS):
    """[len(dates), len(locations)] float values visible at the end of `day` (NaN = unavailable).

    `dates` are Saturday ISO strings. Coverage is computed from every eligible Hub
    row, not only `dates`, exactly as the full-archive resolution does.
    """
    cutoff = cutoff_time(day)
    dates, locations = [str(d) for d in dates], list(locations)
    t_index, l_index = {d: i for i, d in enumerate(dates)}, {l: i for i, l in enumerate(locations)}
    result = np.full((len(dates), len(locations)), np.nan)
    chosen = np.zeros(result.shape, dtype=bool)
    if not dates:
        return result
    for tier in ('hub', 'git'):
        part = revisions_.tiers[tier]
        part = part[part.release.to_numpy() <= cutoff]
        releases = part.release.to_numpy()
        if tier == 'git':
            releases = np.concatenate([releases, revisions_.git_releases[revisions_.git_releases <= cutoff]])
        if not len(releases):
            continue
        start = part.groupby('location').day.min().reindex(locations).fillna('9999-12-31')
        covered = (np.array(dates, dtype='U10')[:, None] >= start.to_numpy().astype('U10')[None, :]) & ~chosen
        values = np.full(result.shape, np.nan)
        _fill(values, part[part.release.to_numpy() == releases.max()], t_index, l_index)
        result[covered] = values[covered]
        chosen |= covered
    delphi = revisions_.tiers['delphi']
    days = delphi.day.to_numpy()
    delphi = delphi.iloc[np.searchsorted(days, min(dates), 'left'):np.searchsorted(days, max(dates), 'right')]
    delphi = delphi[delphi.release.to_numpy() <= cutoff]
    if len(delphi):
        latest = delphi[~delphi.duplicated(['day', 'location'], keep='last')]  # sorted by release within a cell
        values = np.full(result.shape, np.nan)
        _fill(values, latest, t_index, l_index)
        result[~chosen] = values[~chosen]
    return result


def _fill(values, rows, t_index, l_index):
    t = rows.day.map(t_index)
    l = rows.location.map(l_index)
    keep = t.notna() & l.notna()
    values[t[keep].astype(int).to_numpy(), l[keep].astype(int).to_numpy()] = rows.value[keep].to_numpy()


def _latest_snapshot(data_root, dataset):
    root = Path(data_root) / 'raw' / dataset
    latest = root / 'latest.json'
    if not latest.is_file():
        raise FileNotFoundError(f'Missing selected raw dataset: {latest}')
    record = json.loads(latest.read_text())
    snapshot_id = record.get('snapshot_id') or record.get('id')
    if not snapshot_id:
        manifest = record.get('manifest') or record.get('path')
        if manifest:
            snapshot_id = Path(manifest).parent.name
    if not snapshot_id:
        raise ValueError(f'Cannot resolve snapshot id from {latest}')
    return root / 'snapshots' / snapshot_id


def nwss_frame(data_root='data'):
    """Long NWSS index table: report_time, reference_time, location, name, value (NaN rows dropped)."""
    path = _latest_snapshot(data_root, NWSS_DERIVED_DATASET) / 'data.csv.gz'
    if not path.is_file():
        raise FileNotFoundError(f'Registered NWSS state index payload is missing: {path}')
    frame = pd.read_csv(path)
    for column in ('report_time', 'reference_time'):
        frame[column] = frame[column].astype(str).str[:10]
    frame['location'] = frame['geo_value'].astype(str).str.upper()
    frame['pathogen'] = frame['pathogen'].astype(str).str.lower()
    parts = []
    for name in NWSS_INDICES:
        metric = 'pct_rank' if name.endswith('pct_rank') else 'wval_like'
        pathogen = name.removeprefix('nwss_').removesuffix('_' + metric)
        part = frame[frame.pathogen.eq(pathogen)]
        parts.append(pd.DataFrame(dict(report_time=part.report_time, reference_time=part.reference_time,
                                       location=part.location, name=name,
                                       value=pd.to_numeric(part[metric], errors='coerce'))))
    return pd.concat(parts, ignore_index=True).dropna(subset=['value'])


def kinsa_frame(data_root='data'):
    """Saturday-ending weekly means of daily national Kinsa values at every real report date.

    A week is reported at a release only once all seven daily values are visible;
    a row is emitted when the weekly value changes. Returns report_time,
    reference_time, location ('US'), value (NaN rows dropped).
    """
    snapshot = _latest_snapshot(data_root, KINSA_DATASET)
    frame = pd.read_csv(snapshot / 'archive.csv.gz', dtype={'geo_type': str, 'geo_value': str})
    frame = frame[frame['geo_value'].eq('US')].copy()
    frame['report_time'] = frame['report_time'].astype(str).str[:10]
    frame['reference_time'] = frame['reference_time'].astype(str).str[:10]
    frame['value'] = pd.to_numeric(frame[KINSA_SIGNAL], errors='coerce')
    frame.loc[~np.isfinite(frame['value']) | (frame['value'] < 0), 'value'] = np.nan
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
                rows.append((release, saturday_, value))
            previous = value
    weekly = pd.DataFrame(rows, columns=['report_time', 'reference_time', 'value']).assign(location='US', name='kinsa_ili')
    return weekly.dropna(subset=['value'])


def resolve_reports(frame, day, dates, locations=LOCATIONS, names=None):
    """[len(dates), len(locations), len(names)] latest value reported on or before `day`."""
    names = list(names if names is not None else frame.name.unique())
    visible = frame[frame.report_time <= date.fromisoformat(str(day)).isoformat()]
    visible = visible[visible.reference_time.isin(set(dates))]
    visible = visible.sort_values('report_time', kind='stable').drop_duplicates(
        ['name', 'location', 'reference_time'], keep='last')
    result = np.full((len(dates), len(locations), len(names)), np.nan)
    t = visible.reference_time.map({d: i for i, d in enumerate(dates)})
    l = visible.location.map({x: i for i, x in enumerate(locations)})
    k = visible.name.map({n: i for i, n in enumerate(names)})
    keep = t.notna() & l.notna() & k.notna()
    result[t[keep].astype(int), l[keep].astype(int), k[keep].astype(int)] = visible.value[keep].to_numpy()
    return result


def extract(name, as_of=None, *, data_root='data', dates=(), locations=LOCATIONS):
    """One source as a long frame `date, location, value`, visible at the end of day `as_of`.

    `as_of=None` means the build date (today): the retrospective truth. `dates`
    are the Saturday week ends to resolve.
    """
    day = as_of or date.today()
    dates = [str(d) for d in dates]
    if name in ARCHIVE_SOURCES:
        values = resolve(revisions(name, data_root), day, dates, locations)[:, :, None]
    elif name in NWSS_INDICES:
        values = resolve_reports(nwss_frame(data_root), day, dates, locations, [name])
    elif name == 'kinsa_ili':
        values = resolve_reports(kinsa_frame(data_root), day, dates, locations, [name])
    else:
        raise ValueError(f'Unknown source: {name}')
    t, l = np.nonzero(~np.isnan(values[:, :, 0]))
    return pd.DataFrame(dict(date=np.array(dates, dtype=object)[t], location=np.array(locations)[l],
                             value=values[t, l, 0]))
