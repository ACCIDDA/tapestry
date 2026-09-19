"""Cutoff-pinned forward panels using the existing canonical vintage resolver."""
import argparse
from datetime import date, timedelta
import json
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from tapestry.data.geography import STATE_FIPS
from tapestry.data.sources.hubverse import HubMirror
from .wednesday import WednesdayDataset, output_dates, cutoff_time, read_archive
from .finalized import CHANNELS

CUTOFF = '2025-07-26'
TEST_START, TEST_END = '2025-07-30', '2026-07-29'
TARGET_START, TARGET_END = '2025-08-02', '2026-08-01'
TRUTH_CUTOFF = '2026-09-16'
# Operational definition, not a claim that the provider certifies finality.
MATURITY_DAYS = 28
DATASET = 'data/processed/forward_2025.npz'
POPULATIONS = 'data/metadata/forward_2025_locations.csv'



def load_archive():
    """Reuse the existing resolver; cache only parsed, manifest-pinned sources."""
    hub_sources = ('hub_flusight_current', 'hub_covid_current', 'hub_rsv_current')
    root = Path('tmp/forward-benchmark')
    root.mkdir(parents=True, exist_ok=True)
    manifests = sorted(Path('data/raw').glob('*/snapshots/*/manifest.json'))
    relevant = [p for p in manifests if p.parts[2] in (*hub_sources, 'delphi_nhsn', 'delphi_nssp')]
    missing = set((*hub_sources, 'delphi_nhsn', 'delphi_nssp')) - {p.parts[2] for p in relevant}
    if missing:
        raise ValueError(f'Missing raw acquisitions: {sorted(missing)}')
    signature = hashlib.sha256(b''.join(p.read_bytes() for p in relevant)).hexdigest()
    cache = root / f'full-archive-{signature}.pkl'
    if cache.exists():
        with cache.open('rb') as stream: return pickle.load(stream)
    # A cached Delphi parse can be extended by the same reader without
    # re-reading millions of unchanged source rows or remapping provenance IDs.
    delphi = [p for p in relevant if p.parts[2].startswith('delphi_')]
    dsig = hashlib.sha256(b''.join(p.read_bytes() for p in delphi)).hexdigest()
    previous = root / f'archive-{dsig}.pkl'
    if previous.exists():
        with previous.open('rb') as stream: archive = pickle.load(stream)
        archive = read_archive('data', sources=hub_sources, archive=archive)
    else:
        archive = read_archive('data')
    if not archive.delphi or not archive.hub:
        raise ValueError('Acquire both Delphi archives and the three raw Hub exports first')
    with cache.open('wb') as stream: pickle.dump(archive, stream)
    return archive


def asof_panel(archive, days, locations, cutoff):
    state = archive.resolve(cutoff)
    values, available, provenance, reasons = archive.panel(days, locations, state)
    covered = available | (reasons == 2) | (reasons == 3)
    # A Delphi source active for this channel/location by issuance establishes
    # missing-report support after its first observation, not before its archive.
    earliest = {}
    for (day, c, loc) in state[3]:
        earliest[c, loc] = min(day, earliest.get((c, loc), day))
    for c in range(6):
        for l, loc in enumerate(locations):
            if (c, loc) in earliest:
                covered[:, c, l] |= (np.asarray(days) >= earliest[c, loc]) & (reasons[:, c, l] != 4)
    return values, available, covered, provenance, reasons


def recent_eligibility(days, available, covered):
    eligible = covered.copy()
    eligible[:, :3] &= (np.asarray(days) >= '2024-11-01')[:, None, None]
    return eligible & available, eligible & ~available


def build(output=DATASET, mirrors='data/mirrors'):
    archive = load_archive()
    print("Loaded canonical vintage archive", flush=True)
    pop_source = 'hub_flusight_current'
    mirror = HubMirror(Path(mirrors) / f'{pop_source}.git')
    pop_commit = mirror.commit_at('HEAD', CUTOFF)
    Path(POPULATIONS).parent.mkdir(parents=True, exist_ok=True)
    Path(POPULATIONS).write_bytes(mirror.read_file(pop_commit, 'auxiliary-data/locations.csv'))
    locations = tuple(sorted(STATE_FIPS.values())) + ('US',)
    first, end = date(2023, 9, 6), date.fromisoformat(TEST_END)
    issuances = [(first + timedelta(weeks=i)).isoformat() for i in range((end-first).days//7+1)]
    all_days = sorted({d for issue in issuances for d in sum(output_dates(issue), ())})
    train_values, train_valid, _, train_pid, _ = asof_panel(archive, all_days, locations, CUTOFF)
    truth, truth_valid, _, truth_pid, _ = asof_panel(archive, all_days, locations, TRUTH_CUTOFF)
    index = {d:i for i,d in enumerate(all_days)}
    mature_end = (date.fromisoformat(CUTOFF) - timedelta(days=MATURITY_DAYS)).isoformat()
    rows = []
    for issue in issuances:
        print(f'Reconstructing {issue}', flush=True)
        context, targets = output_dates(issue)
        ci, ti = [index[d] for d in context], [index[d] for d in targets]
        x, a, coverage, xp, xr = asof_panel(archive, context, locations, issue)
        # Actual reports at each Wednesday, including old history. No reference filling.
        # Publishers do not supply reliable final flags: no cell is certified final.
        final = np.zeros_like(a)
        training = issue <= CUTOFF
        y, valid = (train_values[ti].copy(), train_valid[ti].copy()) if training else (truth[ti].copy(), truth_valid[ti].copy())
        reference_available = valid.copy()
        if training:
            valid &= (np.asarray(targets) <= mature_end)[:, None, None]
        else:
            valid[2:] &= ((np.asarray(targets[2:]) >= TARGET_START) & (np.asarray(targets[2:]) <= TARGET_END))[:,None,None]
        revision, missing = recent_eligibility(targets[:2], a[-2:], coverage[-2:])
        # Forecast-only training uses cutoff-final proxies throughout context;
        # all candidates share those same older-context values during fitting.
        finalized = train_values[ci].copy()
        fv = train_valid[ci] & (np.asarray(context) <= mature_end)[:,None,None]
        if training:
            x[:-2], a[:-2], final[:-2] = finalized[:-2], fv[:-2], fv[:-2]
            xp[:-2] = train_pid[ci][:-2]
        rows.append(dict(issuance_dates=issue, context_dates=context, target_dates=targets,
            X_values=x, X_available=a, X_final=final,
            X_provenance=xp, X_reason=np.where(a,0,np.where(coverage,2,1)).astype(np.uint8),
            X_cutoff_final=finalized, X_cutoff_valid=fv,
            revision_eligible=revision, reconstruction_eligible=missing,
            Y_available=reference_available,
            Y_recent=y[:2],Y_recent_valid=valid[:2],Y_future=y[2:],Y_future_valid=valid[2:],
            Y_provenance=train_pid[ti] if training else truth_pid[ti],Y_reason=np.where(valid,0,1).astype(np.uint8)))
    arrays = {k:np.stack([r[k] for r in rows]) for k in rows[0]}
    arrays['locations'] = np.asarray(locations)
    metadata = dict(version=1, view='model_calendar', protocol='forward_2025_v1', history_mode='cutoff_final_train_older_asof_test',
        channels=CHANNELS, training_cutoff=CUTOFF, maturity_days=MATURITY_DAYS, mature_training_end=mature_end,
        test_start=TEST_START,test_end=TEST_END,target_start=TARGET_START,target_end=TARGET_END,truth_cutoff=TRUTH_CUTOFF,
        population_commit=pop_commit, population_file=POPULATIONS, provenance=archive.provenance, source_manifests=archive.manifests,
        limitation='Native Hub as_of and Delphi report_time are taken as advertised historical availability; Git timestamps are publication proxies for unversioned files. Delphi fallback outside Hub coverage; absent archive coverage excluded from recent supervision. Final means latest at cutoff aged >=28 days, not certified finality.',
        final_flags='Training supplied cutoff-final proxies flagged; test reports have no certified final status, hence false.',
        older_history='Matched across candidates: cutoff-final older history for fitting; actual Wednesday older history at deployment. This train/deployment mismatch is explicit.',
        cutoff_policy='End of day UTC; no intraday submission claim.')
    if not arrays['Y_future_valid'][arrays['issuance_dates'] <= CUTOFF].any():
        raise ValueError('No historically available training labels')
    ds = WednesdayDataset(arrays, metadata)
    ds.save(output)
    write_support(ds, output)
    export_recent(ds, output)
    print(json.dumps({k:v for k,v in metadata.items() if k != 'provenance'},indent=2))


def export_recent(ds, output):
    """Only two observation weeks per Wednesday; never invent a missing revision."""
    a = ds.arrays
    rows = []
    for i, issue in enumerate(a['issuance_dates']):
        training = issue <= CUTOFF
        for h, age in enumerate((11, 4)):
            day = str(a['target_dates'][i, h])
            for c, target in enumerate(CHANNELS):
                for l, loc in enumerate(ds.locations):
                    visible = bool(a['X_available'][i, -2+h, c, l])
                    reference = bool(a['Y_available'][i, h, c, l])
                    preliminary = float(a['X_values'][i, -2+h, c, l]) if visible else np.nan
                    value = float(a['Y_recent'][i, h, c, l]) if reference else np.nan
                    eligible = bool(a['Y_recent_valid'][i, h, c, l])
                    revision = bool(a['revision_eligible'][i, h, c, l])
                    missing = bool(a['reconstruction_eligible'][i, h, c, l])
                    xp = ds.metadata['provenance'][int(a['X_provenance'][i,-2+h,c,l])]
                    yp = ds.metadata['provenance'][int(a['Y_provenance'][i,h,c,l])]
                    rows.append(dict(issuance_date=str(issue), observation_week=day,
                        report_age_days=age, target=target, location=loc,
                        partition='train' if training else 'test',
                        preliminary_available=visible, preliminary_value=preliminary,
                        reference_available=reference, reference_value=value,
                        preliminary_release=xp.get('release') if visible else None,
                        preliminary_source=xp.get('source') if visible else None,
                        reference_release=yp.get('release') if reference else None,
                        reference_source=yp.get('source') if reference else None,
                        reference_cutoff=CUTOFF if training else TRUTH_CUTOFF,
                        archive_covered=bool(a['X_reason'][i,-2+h,c,l] != 1),
                        regime_eligible=c>=3 or day>='2024-11-01',
                        visible_revision_eligible=eligible and revision,
                        missing_reconstruction_eligible=eligible and missing,
                        revision=value-preliminary if reference and visible else np.nan))
    table = pd.DataFrame(rows)
    table.to_parquet(Path(output).with_suffix('.recent.parquet'), index=False)
    # Numerical integrity audit: these checks guard data units/alignment/leakage.
    assert set(table.report_age_days) == {4, 11}
    ages = (pd.to_datetime(table.issuance_date)-pd.to_datetime(table.observation_week)).dt.days
    assert np.array_equal(ages, table.report_age_days)
    assert table.loc[~table.preliminary_available, 'revision'].isna().all()
    assert not table.duplicated(['issuance_date','observation_week','target','location']).any()
    releases = np.asarray([p.get('release') or '' for p in ds.metadata['provenance']])
    for i, issue in enumerate(a['issuance_dates']):
        cutoff = CUTOFF if issue <= CUTOFF else str(issue)
        present = a['X_available'][i]
        assert np.all(releases[a['X_provenance'][i][present]] <= cutoff_time(cutoff))
        recent = a['X_available'][i,-2:]
        assert np.all(releases[a['X_provenance'][i,-2:][recent]] <= cutoff_time(str(issue)))
        if issue <= CUTOFF:
            assert np.all(releases[a['Y_provenance'][i][a['Y_available'][i]]] <= cutoff_time(CUTOFF))
    Path(output).with_suffix('.audit.json').write_text(json.dumps(dict(
        rows=len(table), issuances=table.issuance_date.nunique(), ages=[4,11],
        dataset_sha256=hashlib.sha256(Path(output).read_bytes()).hexdigest(),
        recent_sha256=hashlib.sha256(Path(output).with_suffix('.recent.parquet').read_bytes()).hexdigest(),
        no_future_input_releases=True, no_future_training_reference_releases=True,
        missing_reports_have_no_revision=True, unique_aligned_cells=True),indent=2)+'\n')


def write_support(ds, output):
    arrays = ds.arrays
    counts = []
    for part, select in [('train',arrays['issuance_dates'] <= CUTOFF),('test',arrays['issuance_dates'] >= TEST_START)]:
        for c,name in enumerate(CHANNELS):
            for h,age in enumerate((11,4)):
                labels=arrays['Y_recent_valid'][select,h,c]
                seen = (arrays['revision_eligible'][select,h,c] & labels).any(1)
                issues = arrays['issuance_dates'][select][seen]
                counts.append(dict(partition=part,target=name,age_days=age,
                    first_visible_issuance=str(min(issues)) if len(issues) else None,
                    last_visible_issuance=str(max(issues)) if len(issues) else None,
                    revision=int((arrays['revision_eligible'][select,h,c]&labels).sum()),
                    reconstruction=int((arrays['reconstruction_eligible'][select,h,c]&labels).sum()),
                    archive_unknown=int(((arrays['X_reason'][select,-2+h,c]==1)&labels).sum())))
    pd.DataFrame(counts).to_csv(Path(output).with_suffix('.support.csv'),index=False)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default=DATASET)
    parser.add_argument('--mirrors',default='data/mirrors')
    args=parser.parse_args()
    build(args.output,args.mirrors)
