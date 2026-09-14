"""Run the staged B0 1–3 sweep and score on the frozen original hub task sets."""
import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

from tapestry.evaluation.hubs import export_b0, KEY
from tapestry.evaluation.epibench import score_case
from tapestry.evaluation.scoring import match_forecasts, aggregate_scores, objective

ROOT = Path('data/experiments/b0_full_20260914')
FROZEN = Path('data/evaluation/b0_hub_comparison')
METRICS = ['wis', 'ae_median', 'interval_coverage_50', 'interval_coverage_95']


def save(name, obj):
    (ROOT / name).write_text(json.dumps(obj, indent=2) + '\n')


def score_run(folder):
    out = folder / 'hub_scores'
    out.mkdir(exist_ok=True)
    exported = export_b0(folder)
    manifest = json.loads((FROZEN / 'manifest.json').read_text())
    records, cases = [], []
    for case in manifest['cases']:
        if case['status'] != 'scored':
            continue
        source = FROZEN / case['directory']
        units = pd.read_parquet(source / 'units.parquet')[KEY + ['observed']]
        ours = match_forecasts(exported[(case['season'], case['target'])], units, case['target'])
        ensemble = pd.read_parquet(source / 'quantiles.parquet')
        ensemble = ensemble[ensemble.model == case['ensemble']]
        hub = manifest['hubs'][case['hub']]
        scoring_case = dict(case, truth_release=hub['truth_vintages'][case['target']])
        scored = score_case(pd.concat([ours.assign(model='candidate'), ensemble], ignore_index=True),
                            units, scoring_case, out / 'epibench' / case['directory'], commit=hub['commit'])
        candidate = scored[scored.model == 'candidate'].assign(target=case['target'], season=case['season'])
        reference_metrics = [m for m in METRICS if m != 'wis']
        reference = scored[scored.model == case['ensemble']][KEY + reference_metrics].rename(
            columns={m: 'ensemble_' + m for m in reference_metrics})
        candidate = candidate.merge(reference, on=KEY, validate='one_to_one')
        records.append(aggregate_scores(candidate, METRICS + ['ensemble_' + m for m in METRICS]))
        cases.append({k: case[k] for k in ('directory', 'season', 'target', 'ensemble', 'n_units')})
    summary = pd.concat(records, ignore_index=True).drop(columns='model')
    summary.to_csv(out / 'summary.csv', index=False)
    (out / 'support.json').write_text(json.dumps(cases, indent=2) + '\n')
    return summary


def execute(name, flags, seed=42):
    key = f'{name}_s{seed}'
    folder = ROOT / key
    command = [sys.executable, '-m', 'tapestry.models.season_cv', '--epochs', '50',
               '--eval-members', '2048', '--seed', str(seed), '--output', str(folder), *flags]
    if not (folder / 'scores.csv').exists():
        print(json.dumps({'started': key, 'command': command}), flush=True)
        with (ROOT / f'{key}.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    summary = score_run(folder)
    result = {'name': name, 'seed': seed, 'flags': flags, 'directory': str(folder),
              'flu_objective': objective(summary), 'admissions_objective': objective(summary, True)}
    results[key] = result
    save('results.json', results)
    print(json.dumps({'completed': result}), flush=True)
    return result


def best(candidates, multi=False):
    field = 'admissions_objective' if multi else 'flu_objective'
    return min(candidates, key=lambda r: r[field])


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    save('protocol.json', {
        'scope': 'Staged full 50-epoch, 2048-draw three-season runs for experiments 1–3; not an exhaustive Cartesian grid.',
        'seeds': [42, 43, 44],
        'selection': 'Seed 42 stages: scaling -> lookback -> dynamics -> task weights. Select minimum geometric mean flu WIS/ensemble, equally weighted by season and geography. Multi-admission finalist gives each admission target equal weight then each available season/geography equal weight.',
        'repeats': 'Baseline plus distinct best influenza and best multi-admission variants across all seed-42 runs receive seeds 43/44.',
        'inference': 'Exploratory selection on already examined folds, not untouched validation; no early stopping, calibration or test-label scaler fitting.',
        'population': 'Frozen fixed population table across seasons; native US remains independent.',
        'frozen_comparison_sha256': hashlib.sha256((FROZEN / 'manifest.json').read_bytes()).hexdigest(),
        'driver_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    base = execute('baseline', [])
    representations = [base, execute('sqrt_geo', ['--count-transform', 'sqrt', '--geography']),
                       execute('fourth_root_geo', ['--count-transform', 'fourth_root', '--geography'])]
    representation = best(representations)
    save('selection.json', {'representation': representation})
    histories = [representation]
    for weeks in (12, 26):
        histories.append(execute(representation['name'] + f'_h{weeks}', representation['flags'] + ['--lookback', str(weeks)]))
    history = best(histories)
    dynamics = execute(history['name'] + '_dynamics', history['flags'] + ['--dynamics'])
    features = best([history, dynamics])
    for weight in ('balanced_admissions', 'flu_only'):
        execute(features['name'] + '_' + weight, features['flags'] + ['--loss-weights', weight])
    candidates = [r for r in results.values() if r['seed'] == 42]
    flu, multi = best(candidates), best(candidates, True)
    save('selection.json', {'representation': representation, 'history': history, 'features': features,
                            'flu_finalist': flu, 'admissions_finalist': multi})
    finalists = {r['name']: r for r in [base, flu, multi]}
    for seed in (43, 44):
        for candidate in finalists.values():
            execute(candidate['name'], candidate['flags'], seed)
    combined = []
    for result in results.values():
        frame = pd.read_csv(Path(result['directory']) / 'hub_scores/summary.csv')
        combined.append(frame.assign(variant=result['name'], seed=result['seed']))
    pd.concat(combined, ignore_index=True).to_csv(ROOT / 'all_hub_scores.csv', index=False)
    save('complete.json', {'runs': len(results), 'fits': 3 * len(results), 'finalists': list(finalists)})
    print('ALL EXPERIMENTS COMPLETE', flush=True)


results = json.loads((ROOT / 'results.json').read_text()) if (ROOT / 'results.json').exists() else {}
if __name__ == '__main__':
    main()
