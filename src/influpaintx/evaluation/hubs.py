"""Normalize saved B0 quantiles and pinned local hub Git blobs into task tables."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from influpaintx.data.geography import STATE_FIPS
from influpaintx.model_data.finalized import season
from influpaintx.models.season_cv import LEVELS, SEASONS

B0_NAME = 'InfluPaintX-B0-finalized-CV'
KEY = ['reference_date', 'target_end_date', 'location', 'horizon']
QCOLS = [f'q{q:g}' for q in LEVELS]
HUBS = {
    'flusight': {'ensemble': 'FluSight-ensemble', 'url': 'https://github.com/cdcepi/FluSight-forecast-hub',
                'truth': 'target-data/time-series.csv', 'targets': {'wk inc flu hosp': 0, 'wk inc flu prop ed visits': 3}},
    'covid': {'ensemble': 'CovidHub-ensemble', 'url': 'https://github.com/CDCgov/covid19-forecast-hub',
              'truth': 'target-data/time-series.parquet', 'targets': {'wk inc covid hosp': 1, 'wk inc covid prop ed visits': 4}},
    'rsv': {'ensemble': 'RSVHub-ensemble', 'url': 'https://github.com/CDCgov/rsv-forecast-hub',
            'truth': 'target-data/time-series.parquet', 'targets': {'wk inc rsv hosp': 2, 'wk inc rsv prop ed visits': 5}},
}


def slug(target):
    return target.replace('wk inc ', '').replace(' ', '_')


def location_codes(values):
    return values.astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(2)


def export_b0(run):
    """Exact hub mapping: reference=context_end+7d; hub horizon=internal lead-1."""
    postal_to_fips = {v: k for k, v in STATE_FIPS.items()} | {'US': 'US'}
    frames = {}
    for held in SEASONS:
        with np.load(Path(run) / f'eval_{held}' / 'forecasts.npz', allow_pickle=False) as data:
            if not np.allclose(data['quantile_levels'], LEVELS):
                raise ValueError('Expected the full 23-level quantile grid')
            for spec in HUBS.values():
                for target, c in spec['targets'].items():
                    q = data['quantiles'][:, :, :, c, :]
                    n, h, l = q.shape[1:]
                    reference = [(date.fromisoformat(d) + timedelta(weeks=1)).isoformat() for d in data['context_end']]
                    frame = pd.DataFrame({
                        'reference_date': np.repeat(reference, h * l),
                        'target_end_date': np.repeat(data['target_dates'].reshape(-1), l),
                        'location': np.tile([postal_to_fips[v] for v in data['locations']], n * h),
                        'horizon': np.tile(np.repeat(np.arange(h), l), n),
                        'b0_original_truth': data['truth'][:, :, c, :].reshape(-1),
                        'b0_original_mask': data['mask'][:, :, c, :].reshape(-1),
                    })
                    frame[QCOLS] = q.reshape(23, -1).T
                    # Retain only held-out target dates, regardless of a revised truth's missingness.
                    keep = frame.target_end_date.map(lambda d: season(date.fromisoformat(d))) == held
                    frames[(held, target)] = frame[keep].copy()
    return frames


class GitHubSnapshot:
    def __init__(self, path):
        self.path = str(path)
        self.commit = self.git('rev-parse', 'HEAD').decode().strip()

    def git(self, *args):
        return subprocess.check_output(['git', f'--git-dir={self.path}', *args])

    def read(self, path):
        return self.git('show', f'{self.commit}:{path}')

    def blobs(self, files):
        # One process for all local blobs avoids thousands of Git invocations.
        with subprocess.Popen(['git', f'--git-dir={self.path}', 'cat-file', '--batch'],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
            for sha, path in files:
                process.stdin.write((sha + '\n').encode())
                process.stdin.flush()
                header = process.stdout.readline().split()
                if len(header) != 3:
                    raise ValueError(f'Cannot read blob {path}: {header}')
                size = int(header[2])
                content = process.stdout.read(size)
                process.stdout.read(1)
                yield path, content
            process.stdin.close()

    def model_files(self):
        entries = self.git('ls-tree', '-r', self.commit, 'model-output').decode().splitlines()
        groups = defaultdict(list)
        for entry in entries:
            meta, path = entry.split('\t', 1)
            parts = path.split('/')
            if len(parts) == 3 and path.endswith(('.csv', '.parquet')):
                groups[parts[1]].append((meta.split()[2], path))
        return groups


def frozen_truth(snapshot, spec):
    content = snapshot.read(spec['truth'])
    df = pd.read_csv(io.BytesIO(content), dtype={'location': str}) if spec['truth'].endswith('.csv') else pd.read_parquet(io.BytesIO(content))
    for column in ['as_of', 'target_end_date']:
        df[column] = df[column].astype(str)
    df['location'] = location_codes(df.location)
    result, vintages = [], {}
    for target in spec['targets']:
        part = df[df.target == target]
        vintage = part.as_of.max()
        # Full publisher release: no per-row fallback to an earlier release after omissions.
        selected = part[part.as_of == vintage][['target', 'target_end_date', 'location', 'observation']].drop_duplicates()
        if selected.duplicated(['target', 'target_end_date', 'location']).any():
            raise ValueError(f'Conflicting truth in {target} release {vintage}')
        result.append(selected)
        vintages[target] = vintage
    return pd.concat(result).rename(columns={'observation': 'observed'}), vintages


def wide_quantiles(df, target):
    """Reject whole malformed tasks, never silently repair crossing or missing quantiles."""
    df = df[df.target == target].copy()
    if df.empty:
        return pd.DataFrame(columns=KEY + QCOLS), {}
    df['quantile_level'] = pd.to_numeric(df.output_type_id, errors='coerce').round(6)
    df['value'] = pd.to_numeric(df.value, errors='coerce')
    df = df[df.quantile_level.isin(LEVELS)].copy()
    keys = KEY + ['quantile_level']
    conflict = df.groupby(keys, dropna=False).value.nunique(dropna=False).gt(1)
    bad_keys = conflict[conflict].reset_index()[KEY].drop_duplicates()
    if not bad_keys.empty:
        df = df.merge(bad_keys.assign(conflict=True), on=KEY, how='left')
        df = df[df.conflict.isna()].drop(columns='conflict')
    df = df.drop_duplicates(keys)
    table = df.pivot(index=KEY, columns='quantile_level', values='value').reindex(columns=LEVELS)
    table.columns = QCOLS
    values = table.to_numpy()
    valid = np.isfinite(values).all(1) & (values >= 0).all(1) & (np.diff(values, axis=1) >= -1e-10).all(1)
    if 'prop ed' in target:
        valid &= (values <= 1).all(1)
    audit = {'conflicting_tasks': len(bad_keys), 'invalid_or_incomplete_tasks': int((~valid).sum()), 'valid_tasks': int(valid.sum())}
    return table.loc[valid].reset_index(), audit


def extract_hub(hub, mirrors, cache, allowed_refs):
    spec = HUBS[hub]
    snapshot = GitHubSnapshot(Path(mirrors) / f'hub_{hub}_current.git')
    folder = Path(cache) / hub
    marker = folder / 'manifest.json'
    if marker.exists():
        metadata = json.loads(marker.read_text())
        if metadata['commit'] == snapshot.commit and metadata['reference_dates'] == sorted(allowed_refs):
            return metadata
        raise ValueError('Use a new cache when the hub commit or reference dates change')
    folder.mkdir(parents=True, exist_ok=True)
    truth, vintages = frozen_truth(snapshot, spec)
    truth.to_parquet(folder / 'truth.parquet', index=False)
    groups = snapshot.model_files()
    audits, files_read = [], 0
    for i, (model, files) in enumerate(sorted(groups.items())):
        parts = []
        # Hub file names begin with the reference Saturday.
        files = [(sha, path) for sha, path in files if Path(path).name[:10] in allowed_refs]
        for path, content in snapshot.blobs(files):
            if content.startswith(b'version https://git-lfs'):
                audits.append({'model': model, 'path': path, 'error': 'Git LFS pointer, not forecast payload'})
                continue
            df = pd.read_csv(io.BytesIO(content), dtype={'location': str, 'output_type_id': str}) if path.endswith('.csv') else pd.read_parquet(io.BytesIO(content))
            required = set(KEY) | {'target', 'output_type', 'output_type_id', 'value'}
            if not required.issubset(df.columns):
                audits.append({'model': model, 'path': path, 'error': 'Missing required forecast columns'})
                continue
            df['reference_date'] = df.reference_date.astype(str)
            df['target_end_date'] = df.target_end_date.astype(str)
            df['location'] = location_codes(df.location)
            df['horizon'] = pd.to_numeric(df.horizon, errors='coerce')
            df = df[(df.output_type == 'quantile') & df.target.isin(spec['targets']) &
                    df.horizon.isin([0, 1, 2, 3]) & df.reference_date.isin(allowed_refs)].copy()
            expected = pd.to_datetime(df.reference_date) + pd.to_timedelta(df.horizon * 7, unit='D')
            aligned = pd.to_datetime(df.target_end_date) == expected
            if (~aligned).any():
                audits.append({'model': model, 'path': path, 'misaligned_rows': int((~aligned).sum())})
            parts.append(df[aligned])
            files_read += 1
        if parts:
            all_rows = pd.concat(parts, ignore_index=True)
            for target in spec['targets']:
                wide, audit = wide_quantiles(all_rows, target)
                if len(wide):
                    destination = folder / slug(target)
                    destination.mkdir(exist_ok=True)
                    wide.to_parquet(destination / f'{model}.parquet', index=False)
                audits.append({'model': model, 'target': target, **audit})
        if (i + 1) % 10 == 0 or i + 1 == len(groups):
            print(json.dumps({'hub': hub, 'extracted_models': i + 1, 'total_models': len(groups)}), flush=True)
    metadata = {'hub': hub, 'commit': snapshot.commit, 'url': spec['url'],
                'reference_dates': sorted(allowed_refs), 'truth_vintages': vintages,
                'files_read': files_read, 'audit': audits}
    marker.write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata
