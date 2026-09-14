"""Run the staged B0 1–3 sweep and score on the frozen original hub task sets."""
import json
import hashlib
import subprocess
import sys
from pathlib import Path

import pandas as pd

from tapestry.evaluation.hubs import export_b0, KEY, QCOLS
from tapestry.evaluation.compare import score_with_r
from tapestry.evaluation.scoring import match_forecasts, matched_scores, aggregate_scores, objective

ROOT = Path('data/experiments/b0_full_20260914')
FROZEN = Path('data/evaluation/b0_hub_comparison')
METRICS = ['wis', 'ae_median', 'interval_coverage_50', 'interval_coverage_95']


def save(name, obj):
    (ROOT / name).write_text(json.dumps(obj, indent=2) + '\n')


def score_run(folder):
    out = folder / 'hub_scores'
    if (out / 'summary.csv').exists():
        return pd.read_csv(out / 'summary.csv')
    out.mkdir(exist_ok=True)
    exported = export_b0(folder)
    manifest = json.loads((FROZEN / 'manifest.json').read_text())
    frames, ensembles, cases = [], [], []
    for case in manifest['cases']:
        if case['status'] != 'scored':
            continue
        name = case['directory']
        source = FROZEN / name
        units = pd.read_parquet(source / 'units.parquet')[KEY + ['observed']]
        ours = exported[(case['season'], case['target'])][KEY + QCOLS]
        wide = match_forecasts(ours, units, case['target'])
        # Unique model key keeps simultaneous target/season tasks distinct in R.
        frames.append(wide.assign(model=name))
        ens = pd.read_csv(source / 'scores.csv', dtype={'location': str})
        ens = ens[ens.model == case['ensemble']].copy()
        ens = ens.merge(units[KEY + ['observed']], on=KEY, validate='one_to_one')
        matched = matched_scores(ens, units, METRICS)
        ensembles.append(matched.assign(model=name))
        cases.append({k: case[k] for k in ('directory', 'season', 'target', 'ensemble', 'n_units')})
    scores = score_with_r(pd.concat(frames, ignore_index=True), out)
    ens = pd.concat(ensembles, ignore_index=True)
    records = []
    for case in cases:
        units = pd.read_parquet(FROZEN / case['directory'] / 'units.parquet')
        s = scores[scores.model == case['directory']].merge(
            units[KEY + ['observed']], on=KEY, validate='one_to_one')
        s = matched_scores(s, units, METRICS)
        e = ens[ens.model == case['directory']][KEY + METRICS].rename(
            columns={metric: 'ensemble_' + metric for metric in METRICS})
        paired = s.merge(e, on=KEY, validate='one_to_one').assign(
            target=case['target'], season=case['season'])
        records.append(aggregate_scores(paired, METRICS + ['ensemble_' + m for m in METRICS]))
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
