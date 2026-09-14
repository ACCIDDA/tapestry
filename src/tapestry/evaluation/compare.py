"""Score saved quantiles on ensemble-supported units through R scoringutils."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

from .hubs import B0_NAME, HUBS, KEY, QCOLS, SEASONS, export_b0, extract_hub, slug


def score_with_r(wide, folder, rscript='Rscript'):
    """Persist reusable long-form input and invoke actual scoringutils, not a Python substitute."""
    payload = wide.melt(id_vars=['model', *KEY, 'observed'], value_vars=QCOLS,
                        var_name='quantile_level', value_name='predicted')
    payload['quantile_level'] = payload.quantile_level.str.removeprefix('q').astype(float)
    payload.to_csv(folder / 'quantiles.csv.gz', index=False)
    del payload
    script = Path(__file__).with_name('score_quantiles.R')
    command = [rscript, str(script), str(folder / 'quantiles.csv.gz'),
               str(folder / 'scores.csv'), str(folder / 'r_versions.txt')]
    result = subprocess.run(command, capture_output=True, text=True)
    (folder / 'scoringutils.log').write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f'scoringutils failed; see {folder / "scoringutils.log"}')
    return pd.read_csv(folder / 'scores.csv', dtype={'location': str})


def compare_case(ours, hub, held_out, target, cache, output, model_name, rscript):
    spec = HUBS[hub]
    source = Path(cache) / hub / slug(target)
    ensemble_file = source / f'{spec["ensemble"]}.parquet'
    if not ensemble_file.exists():
        return {'hub': hub, 'season': held_out, 'target': target, 'status': 'no ensemble forecasts'}
    ensemble = pd.read_parquet(ensemble_file)
    truth = pd.read_parquet(Path(cache) / hub / 'truth.parquet')
    truth = truth[(truth.target == target) & np.isfinite(truth.observed)].drop(columns='target')
    # The ENSEMBLE defines the scoring units, restricted to our held-out forecasts and known truth.
    units = ours[KEY + ['b0_original_truth', 'b0_original_mask']].merge(ensemble[KEY], on=KEY, validate='one_to_one')
    units = units.merge(truth, on=['target_end_date', 'location'], validate='many_to_one')
    if units.empty:
        return {'hub': hub, 'season': held_out, 'target': target, 'status': 'no shared ensemble-scored units'}
    folder = Path(output) / f'{hub}_{slug(target)}_{held_out}'
    folder.mkdir(parents=True, exist_ok=True)
    base = units[KEY + ['observed']]
    units.to_parquet(folder / 'units.parquet', index=False)
    ours = ours[KEY + QCOLS].merge(base, on=KEY, validate='one_to_one').assign(model=model_name)
    frames = [ours]
    for file in sorted(source.glob('*.parquet')):
        candidate = pd.read_parquet(file).merge(base, on=KEY, validate='one_to_one')
        if len(candidate):
            frames.append(candidate.assign(model=file.stem))
    wide = pd.concat(frames, ignore_index=True)
    wide.to_parquet(folder / 'quantiles.parquet', index=False)
    print(json.dumps({'case': folder.name, 'units': len(base), 'models': wide.model.nunique(), 'quantile_rows': len(wide) * len(QCOLS)}), flush=True)
    scores = score_with_r(wide, folder, rscript)
    metrics = ['wis', 'ae_median', 'interval_coverage_50', 'interval_coverage_95',
               'overprediction', 'underprediction', 'dispersion', 'bias']
    leaderboard = scores.groupby('model')[metrics].mean()
    leaderboard['n'] = scores.groupby('model').size()
    leaderboard['coverage'] = leaderboard.n / len(base)
    leaderboard['eligible_best'] = ((leaderboard.n == len(base)) &
        ~leaderboard.index.isin([model_name, spec['ensemble']]) &
        ~leaderboard.index.str.lower().str.contains('baseline'))
    # Best means minimum mean WIS on the identical ensemble-defined support.
    # Partial submitters are scored/reported but cannot win by omitting hard tasks.
    eligible = leaderboard[leaderboard.eligible_best].sort_values('wis')
    best = str(eligible.index[0]) if len(eligible) else None
    leaderboard.sort_values(['eligible_best', 'wis'], ascending=[False, True]).to_csv(folder / 'leaderboard.csv')
    detail = scores.assign(geography=np.where(scores.location == 'US', 'US', 'states_dc'))
    detail.groupby(['model', 'geography', 'horizon'])[metrics].mean().to_csv(folder / 'scores_by_horizon.csv')
    detail.groupby(['model', 'location', 'horizon'])[metrics].mean().to_csv(folder / 'scores_by_location.csv')
    mismatch = units.b0_original_mask & ~np.isclose(units.b0_original_truth, units.observed, rtol=1e-5, atol=1e-7)
    info = {'hub': hub, 'season': held_out, 'target': target, 'status': 'scored', 'directory': folder.name,
            'model_name': model_name, 'ensemble': spec['ensemble'], 'best': best,
            'n_units': len(base), 'n_reference_dates': int(base.reference_date.nunique()),
            'reference_min': base.reference_date.min(), 'reference_max': base.reference_date.max(),
            'n_locations': int(base.location.nunique()), 'n_models': int(wide.model.nunique()),
            'eligible_best_models': list(eligible.index),
            'best_rule': 'lowest mean WIS among non-baseline, non-official-ensemble submitted models with 100% coverage of the same ensemble-scored units',
            'truth_changed_from_original_cv_cells': int(mismatch.sum()),
            'hub_truth_additional_cells': int((~units.b0_original_mask).sum()),
            'summary': leaderboard.loc[[m for m in [model_name, best, spec['ensemble']] if m]].reset_index().to_dict('records')}
    (folder / 'comparison.json').write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps({'case': folder.name, 'best': best, 'complete_competitors': len(eligible)}), flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='data/experiments/b0_season_cv_20260913')
    parser.add_argument('--mirrors', default='data/mirrors')
    parser.add_argument('--cache', default='data/evaluation/hub_cache')
    parser.add_argument('--output', default='data/evaluation/b0_hub_comparison')
    parser.add_argument('--model-name', default=B0_NAME)
    parser.add_argument('--rscript', default=shutil.which('Rscript') or 'Rscript')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        raise ValueError('Use a new output directory to preserve a completed comparison')
    ours = export_b0(args.run)
    refs = set().union(*(set(frame.reference_date) for frame in ours.values()))
    manifest = {'config': vars(args), 'support': 'only units with complete ensemble quantiles and frozen hub truth, overlapping held-out B0 forecasts',
                'quantiles': QCOLS, 'model_information': 'finalized-input retrospective CV, not real-time hub submissions',
                'best_rule': '100% identical ensemble support; lowest mean WIS including US; official ensemble and baseline excluded',
                'scoring': 'R scoringutils as_forecast_quantile/score; metrics mirror epibench scoring_bridge.py',
                'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*') if p.suffix in {'.py', '.R'}},
                'run_manifest_sha256': hashlib.sha256((Path(args.run) / 'manifest.json').read_bytes()).hexdigest(),
                'hubs': {}, 'cases': []}
    before = time.perf_counter()
    for hub, spec in HUBS.items():
        manifest['hubs'][hub] = extract_hub(hub, args.mirrors, args.cache, refs)
        for target in spec['targets']:
            for held_out in SEASONS:
                info = compare_case(ours[(held_out, target)], hub, held_out, target,
                                    args.cache, output, args.model_name, args.rscript)
                manifest['cases'].append(info)
                (output / 'progress.json').write_text(json.dumps(manifest, indent=2) + '\n')
    manifest['seconds'] = time.perf_counter() - before
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    summary = []
    for case in manifest['cases']:
        for row in case.get('summary', []):
            summary.append({'hub': case['hub'], 'season': case['season'], 'target': case['target'], **row})
    pd.DataFrame(summary).to_csv(output / 'comparison_summary.csv', index=False)
    from .summary import refresh
    refresh(output)
    print(json.dumps({'output': args.output, 'seconds': manifest['seconds']}), flush=True)


if __name__ == '__main__':
    main()
