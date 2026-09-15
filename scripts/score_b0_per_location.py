"""Per-location WIS for every crosses run, using the same scoring path as totals.py.

`manager rank` only keeps three geography aggregates (all, states_dc, US), which
weight a location by its WIS and so let the US carry ~45% of admissions. This
writes one row per run/target/season/location so a score can instead weight all
52 locations equally. Verified against season_scores.csv: the states_dc/US sums
reproduce to 1e-14.

    .venv/bin/python scripts/score_b0_per_location.py -o per_location.parquet
"""
import argparse
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd

from tapestry.evaluation.hubs import KEY, QCOLS, export_b0
from tapestry.evaluation.totals import quantile_scores, frozen_cases
from tapestry.evaluation.scoring import match_forecasts

FROZEN = Path('data/evaluation/b0_hub_comparison_q23')
CASES = frozen_cases(FROZEN)

def score_one(run):
    frames = export_b0(run['path'])
    out = []
    for case in CASES:
        folder = FROZEN / case['directory']
        units = pd.read_parquet(folder / 'units.parquet')
        quantiles = pd.read_parquet(folder / 'quantiles.parquet')
        model = match_forecasts(frames[(case['season'], case['target'])], units, case['target'])
        ens = match_forecasts(quantiles[quantiles.model.eq(case['ensemble'])], units, case['target'])
        model = model.sort_values(KEY).reset_index(drop=True)
        ens = ens.sort_values(KEY).reset_index(drop=True)
        y = model.observed.to_numpy()
        t = pd.DataFrame({
            'location': model.location.to_numpy(),
            'model_wis': quantile_scores(model[QCOLS].to_numpy(), y).wis.to_numpy(),
            'ensemble_wis': quantile_scores(ens[QCOLS].to_numpy(), y).wis.to_numpy()})
        g = t.groupby('location').sum()
        g['n'] = t.groupby('location').size()
        out.append(g.reset_index().assign(target=case['target'], season=case['season'],
                                          name=run['name'], seed=run['seed']))
    return pd.concat(out, ignore_index=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-e', '--experiment', default='b0-us-cross-4')
    parser.add_argument('-r', '--ranking', required=True, help='ranking-<hash> folder name')
    parser.add_argument('--root', type=Path, default=Path('data/experiments'))
    parser.add_argument('-o', '--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    manifest = json.loads((args.root / args.experiment / args.ranking / 'manifest.json').read_text())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        parts = list(pool.map(score_one, manifest['runs'], chunksize=1))
    frame = pd.concat(parts, ignore_index=True)
    frame.to_parquet(args.output)
    print(f'{len(frame)} rows, {frame.groupby(["name", "seed"]).ngroups} runs, {frame.location.nunique()} locations')


if __name__ == '__main__':
    main()
