"""Run the complete EpiBench config pipeline on a frozen local Hubverse snapshot.

Assumption: retain the existing ensemble-supported task set and full-release truth.
The snapshot adapts those inputs to EpiBench; it does not replace its scorer.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

from .configurations import digest
from .hubs import KEY, QCOLS, location_codes
from .scoring import match_forecasts, matched_scores


def score_case(wide, units, case, folder, *, epibench=Path('../epibench'),
               mirrors=Path('data/mirrors'), commit='HEAD'):
    """Score all candidates AND the reference through `python -m epibench score`."""
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    source = Path(epibench).resolve() / 'src' / 'epibench'
    if not (source / 'score.py').exists():
        raise FileNotFoundError(f'EpiBench checkout missing at {source.parent.parent}')
    code = {p.name: digest(p.read_text()) for p in source.glob('*.py')}
    schemas = {}
    mirror = Path(mirrors) / f"hub_{case['hub']}_current.git"
    for name in ('admin.json', 'tasks.json'):
        schemas[name] = subprocess.check_output(
            ['git', f'--git-dir={mirror}', 'show', f'{commit}:hub-config/{name}']).decode()
    # Validate completeness before EpiBench's config route (which otherwise warns).
    models = {}
    for model, frame in wide.groupby('model', sort=True):
        models[model] = match_forecasts(frame, units, case['target']).assign(model=model)
    if case['ensemble'] not in models:
        raise ValueError('The official ensemble must be included for EpiBench relative WIS')
    payload = pd.concat(models.values(), ignore_index=True)
    versions = subprocess.check_output(['Rscript', '-e',
        'cat(paste("R", getRversion()), paste("scoringutils", packageVersion("scoringutils")), '
        'paste("purrr", packageVersion("purrr")), sep="\\n")'], text=True)
    fingerprint = digest(dict(payload=pd.util.hash_pandas_object(payload, index=False).astype(str).tolist(),
                              case=case, schemas=schemas, epibench=code, versions=versions,
                              adapter=Path(__file__).read_text()))
    stamp = folder / 'provenance.json'
    result_file = folder / 'output' / 'EpiBenchmark_scores.csv'
    if stamp.exists() and json.loads(stamp.read_text())['fingerprint'] == fingerprint and result_file.exists():
        scores = pd.read_csv(result_file, dtype={'location': str})
    else:
        if result_file.exists():
            raise ValueError(f'Inputs or scorer changed; use a new output directory: {folder}')
        hub = folder / 'hub'
        for name, content in schemas.items():
            path = hub / 'hub-config' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        truth = units[['target_end_date', 'location', 'observed']].drop_duplicates()
        if truth.duplicated(['target_end_date', 'location']).any():
            raise ValueError('Conflicting frozen truth')
        # Single release prevents EpiBench latest-per-row selection from changing truth.
        truth = truth.rename(columns={'observed': 'observation'}).assign(
            target=case['target'], as_of=case.get('truth_release', units.target_end_date.max()))
        (hub / 'target-data').mkdir(parents=True, exist_ok=True)
        # Parquet preserves numeric-looking location strings in hubdata truth loading.
        truth.to_parquet(hub / 'target-data/time-series.parquet', index=False)
        paths = {}
        for model, frame in models.items():
            long = frame[KEY + QCOLS].melt(id_vars=KEY, var_name='output_type_id', value_name='value')
            long['output_type_id'] = long.output_type_id.str.removeprefix('q')
            long = long.assign(output_type='quantile', target=case['target'])
            directory = hub / 'model-output' / model if model == case['ensemble'] else folder / 'models' / model
            directory.mkdir(parents=True, exist_ok=True)
            long.to_csv(directory / 'forecasts.csv', index=False)
            if model != case['ensemble']:
                paths[model] = str(directory)
        config = dict(hub_path=str(hub), target=case['target'], models=paths,
                      baseline_model=case['ensemble'], output_path=str(folder / 'output'),
                      evaluation_start_date=str(units.target_end_date.min()),
                      evaluation_end_date=str(units.target_end_date.max()))
        config_file = folder / 'score.yaml'
        config_file.write_text(yaml.safe_dump(config, sort_keys=False))
        command = [sys.executable, '-m', 'epibench', 'score', '--config-path', str(config_file)]
        env = dict(os.environ, PYTHONPATH=str(source.parent) + os.pathsep + os.environ.get('PYTHONPATH', ''))
        with (folder / 'epibench.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        if result.returncode:
            raise RuntimeError(f"EpiBench scoring failed; see {folder / 'epibench.log'}")
        scores = pd.read_csv(result_file, dtype={'location': str})
        stamp.write_text(json.dumps(dict(fingerprint=fingerprint, command=command, epibench_code_sha256=code,
            schemas_sha256={k:digest(v) for k,v in schemas.items()}, r_versions=versions,
            hub_commit=commit, quantile_levels=[float(q[1:]) for q in QCOLS], target=case['target'], n_units=len(units), models=list(models)), indent=2) + '\n')
        (folder / 'r_versions.txt').write_text(versions)
    # EpiBench's CSV bridge infers numeric locations when no US row is present.
    scores['location'] = location_codes(scores.location)
    scores['horizon'] = pd.to_numeric(scores.horizon)
    if set(scores.model) != set(models):
        raise ValueError('EpiBench changed the model set')
    checked = []
    for model, frame in scores.groupby('model'):
        if len(frame) != len(units):
            raise ValueError(f'EpiBench changed frozen support for {model}')
        frame = frame.merge(units[KEY + ['observed']], on=KEY, validate='one_to_one')
        checked.append(matched_scores(frame, units))
    scores = pd.concat(checked, ignore_index=True)
    # Verify EpiBench's relative score against the same reference, allowing zero WIS.
    ref = scores[scores.model == case['ensemble']][KEY + ['wis']].rename(columns={'wis':'ensemble_wis'})
    scores = scores.merge(ref, on=KEY, validate='many_to_one')
    expected = scores.wis.div(scores.ensemble_wis).where(scores.ensemble_wis.ne(0))
    np.testing.assert_allclose(scores.rwis, expected, equal_nan=True)
    return scores
