"""Rebuild `derived_nwss_state_indices` (the two wastewater indices) from Delphi NWSS.

    python -m chromantis.dataset.build nwss-indices --data-root data

Reads the selected `delphi_nwss` snapshot (per-sample `*_avg_conc_lin` sewershed
archives for flu, covid, rsv) and the selected `delphi_nwss_aux` snapshot (whose
`state_territory` and `major_lab_method` are themselves versioned), and registers
a new `derived_nwss_state_indices` raw snapshot (`data.csv.gz`, `derivation.json`)
that `dataset.extract.nwss_frame` reads. Run it after pulling new NWSS data and
before `build build`. It is a separate command, not part of `build build`, because
it streams the 1.2 GB auxiliary archive (several minutes) while the NWSS inputs
change only when re-pulled.

Policy (recorded in each snapshot's `derivation.json`):

- Report times before `ARCHIVE_SERVICE_START` (2026-02-25, the first Delphi
  archive vintage) are dropped: earlier states are not reconstructable.
- At every report time R (union of signal and aux report times), the publisher
  state is resolved as the latest row per sample key reported on or before R
  (so retractions/null revisions count); then values must be finite and > 0 with
  reference_time <= R, and the sample must have a visible aux row with a state.
  Same-release conflicting duplicates become missing, never resolved by row order.
- group g = (sewershed, nwss_source, pcr_target, major_lab_method); x = ln(value).
  A group is eligible with >= `MIN_WEEKS` distinct weeks and sd(x) > 0, using only
  samples visible at R (origin-safe baselines).
- `wval_like` = exp((x - p10_g) / sd_g); `pct_rank` = empirical percentile of x
  within g. Mean over a group's samples in a Saturday-ending week, median over
  groups at a site, median over sites in the state (>= `MIN_SITES` sites) and
  nationally (`US`, same >= `MIN_SITES` rule).
- Each report time is a complete derived state: a state-week present at an earlier
  report time and absent now gets an explicit null row, so consumers do not carry
  an obsolete value through a retraction.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .extract import NWSS_DERIVED_DATASET, _latest_snapshot

ARCHIVE_SERVICE_START = '2026-02-25'
MIN_WEEKS = 26
MIN_SITES = 3
PATHOGENS = ('flu', 'covid', 'rsv')
JOIN = ['geo_value', 'nwss_source', 'reference_time', 'sample_index', 'pcr_target']


def score_wval(d):
    """exp((x - p10_g) / sd_g) of each sample against its group's visible history."""
    grp = d.groupby('g', sort=False)['x']
    return np.exp((d['x'] - d['g'].map(grp.quantile(0.10))) / d['g'].map(grp.std()))


def score_pct_rank(d):
    """Empirical percentile of each sample's x within its group's visible history."""
    return d.groupby('g', sort=False)['x'].rank(pct=True)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _read_archive(path, columns):
    parts = []
    for chunk in pd.read_csv(path, usecols=columns, chunksize=750_000, dtype_backend='pyarrow'):
        for column in JOIN:
            chunk[column] = chunk[column].astype('string[pyarrow]')
        chunk['report_time'] = chunk['report_time'].astype(str).str[:10]
        chunk['reference_time'] = chunk['reference_time'].astype(str).str[:10]
        yield chunk[chunk['report_time'] >= ARCHIVE_SERVICE_START]


def _signal(path):
    """One pathogen's sample revisions; same-release conflicting duplicates -> NaN."""
    d = pd.concat([c[['report_time', *JOIN, 'value']] for c in _read_archive(path, None) if len(c)],
                  ignore_index=True).sort_values('report_time')
    key = ['report_time', *JOIN]
    duplicates = d[d.duplicated(key, keep=False)]
    if len(duplicates):
        conflicts = duplicates.groupby(key, sort=False)['value'].nunique(dropna=False)
        bad = conflicts[conflicts > 1].index
        if len(bad):
            indexed = d.set_index(key)
            indexed.loc[bad, 'value'] = np.nan
            d = indexed.reset_index().sort_values('report_time')
        d = d.drop_duplicates(key, keep='last')
    return d


def _aux(path, wanted):
    """Aux revisions of the wanted sample keys; conflicting (state, lab method) -> NA."""
    columns = ['report_time', *JOIN, 'state_territory', 'major_lab_method']
    parts = [c[pd.MultiIndex.from_frame(c[JOIN]).isin(wanted)] for c in _read_archive(path, columns)]
    aux = pd.concat([p for p in parts if len(p)], ignore_index=True).sort_values('report_time')
    key = ['report_time', *JOIN]
    duplicates = aux[aux.duplicated(key, keep=False)].copy()
    if len(duplicates):
        signature = (duplicates['state_territory'].fillna('').astype(str) + '|' +
                     duplicates['major_lab_method'].fillna('').astype(str))
        conflicts = duplicates.assign(_signature=signature).groupby(key, sort=False)['_signature'].nunique()
        bad = conflicts[conflicts > 1].index
        if len(bad):
            indexed = aux.set_index(key)
            indexed.loc[bad, ['state_territory', 'major_lab_method']] = pd.NA
            aux = indexed.reset_index().sort_values('report_time')
        aux = aux.drop_duplicates(key, keep='last')
    return aux


def _at_release(d, aux, release):
    """State and national indices of one pathogen as computable at report time `release`."""
    visible = d[d['report_time'] <= release].drop_duplicates(JOIN, keep='last')
    visible['value'] = pd.to_numeric(visible['value'], errors='coerce')
    visible = visible[np.isfinite(visible['value']) & (visible['value'] > 0) &
                      (visible['reference_time'] <= release)].copy()
    aux_visible = aux[aux['report_time'] <= release].drop_duplicates(JOIN, keep='last')
    visible = visible.merge(aux_visible[JOIN + ['state_territory', 'major_lab_method']],
                            on=JOIN, how='inner', validate='one_to_one')
    visible = visible[visible['state_territory'].notna()].copy()
    visible['state'] = visible['state_territory'].astype(str).str.upper()
    visible['major_lab_method'] = visible['major_lab_method'].fillna('').astype(str)
    visible['reference_time'] = pd.to_datetime(visible['reference_time'])
    visible['week_end'] = visible['reference_time'] + pd.to_timedelta(
        (5 - visible['reference_time'].dt.weekday) % 7, unit='D')
    visible['g'] = (visible['geo_value'].astype(str) + '|' + visible['nwss_source'].astype(str) + '|' +
                    visible['pcr_target'].astype(str) + '|' + visible['major_lab_method'])
    visible['x'] = np.log(visible['value'])
    groups = visible.groupby('g', sort=False)
    weeks, spread = groups['week_end'].nunique(), groups['x'].std()
    visible = visible[visible['g'].isin(weeks[weeks >= MIN_WEEKS].index.intersection(spread[spread > 0].index))].copy()
    if visible.empty:
        return None
    visible['wval_like'] = score_wval(visible)
    visible['pct_rank'] = score_pct_rank(visible)
    group_week = visible.groupby(['state', 'geo_value', 'g', 'week_end'], as_index=False).agg(
        wval_like=('wval_like', 'mean'), pct_rank=('pct_rank', 'mean'))
    site_week = group_week.groupby(['state', 'geo_value', 'week_end'], as_index=False).agg(
        wval_like=('wval_like', 'median'), pct_rank=('pct_rank', 'median'))
    state = site_week.groupby(['state', 'week_end'], as_index=False).agg(
        wval_like=('wval_like', 'median'), pct_rank=('pct_rank', 'median'), n_sites=('geo_value', 'nunique'))
    nation = site_week.groupby('week_end', as_index=False).agg(
        wval_like=('wval_like', 'median'), pct_rank=('pct_rank', 'median'), n_sites=('geo_value', 'nunique'))
    nation['state'] = 'US'
    out = pd.concat([state[state['n_sites'] >= MIN_SITES], nation[nation['n_sites'] >= MIN_SITES]], ignore_index=True)
    return out.rename(columns={'state': 'geo_value', 'week_end': 'reference_time'})


def indices(data_root='data'):
    """(frame, derivation metadata): every report time's complete derived state."""
    signal_root = _latest_snapshot(data_root, 'delphi_nwss')
    aux_root = _latest_snapshot(data_root, 'delphi_nwss_aux')
    aux_candidates = list(aux_root.glob('*.csv.gz')) + list(aux_root.glob('*.csv'))
    if not aux_candidates:
        raise FileNotFoundError(f'No NWSS auxiliary payload in {aux_root}')
    aux_path = aux_candidates[0]
    signal_paths = {p: signal_root / f'signal={p}_avg_conc_lin' / 'geo_type=sewershed' / 'archive.csv.gz'
                    for p in PATHOGENS}
    signals = {p: _signal(path) for p, path in signal_paths.items()}
    wanted = pd.MultiIndex.from_frame(pd.concat([d[JOIN].drop_duplicates() for d in signals.values()],
                                                ignore_index=True).drop_duplicates())
    aux = _aux(aux_path, wanted)
    outputs = []
    for pathogen, d in signals.items():
        for release in sorted(set(d['report_time']) | set(aux['report_time'])):
            state = _at_release(d, aux, release)
            if state is not None:
                outputs.append(state.assign(report_time=release, pathogen=pathogen))
    if not outputs:
        raise ValueError('No eligible origin-safe NWSS state indices were produced')
    columns = ['report_time', 'geo_value', 'reference_time', 'pathogen', 'wval_like', 'pct_rank', 'n_sites']
    result = pd.concat(outputs, ignore_index=True)[columns]
    result['reference_time'] = result['reference_time'].dt.strftime('%Y-%m-%d')
    complete = []  # explicit null rows for state-weeks that disappear at a later report time
    for pathogen, values in result.groupby('pathogen'):
        cells = values[['geo_value', 'reference_time']].drop_duplicates()
        for release in sorted(values['report_time'].unique()):
            expanded = cells.merge(values[values['report_time'].eq(release)],
                                   on=['geo_value', 'reference_time'], how='left')
            complete.append(expanded.assign(report_time=release, pathogen=pathogen))
    result = pd.concat(complete, ignore_index=True)[columns]
    result.insert(1, 'geo_type', np.where(result['geo_value'].eq('US'), 'nation', 'state'))
    metadata = dict(
        version=1, rows=len(result), report_time_range=[result.report_time.min(), result.report_time.max()],
        archive_service_start=ARCHIVE_SERVICE_START, observed_archive_report_time_start=result.report_time.min(),
        availability_policy=('Native signal and auxiliary archive states resolved at each report_time; '
                             'no estimated availability or backfilled vintage.'),
        baseline_policy=('At each report_time, group p10, sd, empirical rank and 26-week eligibility '
                         'use only samples with report_time and reference_time no later than that origin.'),
        aggregation_policy=('Mean within group-week; median groups within site; median sites within '
                            'state/nation; at least 3 sites.'),
        inputs=[dict(dataset='delphi_nwss', snapshot_id=signal_root.name,
                     files=[dict(path=str(p), sha256=_sha256(p)) for p in signal_paths.values()]),
                dict(dataset='delphi_nwss_aux', snapshot_id=aux_root.name,
                     files=[dict(path=str(aux_path), sha256=_sha256(aux_path))])])
    return result, metadata


def register(data_root='data'):
    """Compute the indices and commit them as the new selected `derived_nwss_state_indices` snapshot."""
    from chromantis.data.catalog import CATALOG
    from chromantis.data.repository import RawDataRepository
    result, metadata = indices(data_root)
    repository = RawDataRepository(data_root)
    repository.initialize(CATALOG)
    with repository.begin_snapshot(CATALOG[NWSS_DERIVED_DATASET]) as snapshot:
        destination = snapshot.path('data.csv.gz', rows=len(result), media_type='text/csv+gzip')
        result.to_csv(destination, index=False, compression='gzip')
        snapshot.write_json('derivation.json', metadata)
        manifest = snapshot.commit(selector={'pathogens': list(PATHOGENS), 'indices': ['wval_like', 'pct_rank']},
                                   source_state={'inputs': metadata['inputs'], 'origin_safe': True})
    return dict(metadata, dataset_key=manifest.dataset_key, snapshot_id=manifest.snapshot_id)
