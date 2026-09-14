"""Summarize the staged B0 sweep without modifying or regenerating forecasts."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('data/experiments/b0_full_20260914')


def table(frame, cols):
    rows = ['| ' + ' | '.join(cols) + ' |', '| ' + ' | '.join(['---'] * len(cols)) + ' |']
    rows += ['| ' + ' | '.join(str(row[c]) for c in cols) + ' |' for _, row in frame.iterrows()]
    return '\n'.join(rows)


def main():
    results = json.loads((ROOT / 'results.json').read_text())
    selection = json.loads((ROOT / 'selection.json').read_text())
    completion = json.loads((ROOT / 'complete.json').read_text())
    scores = pd.read_csv(ROOT / 'all_hub_scores.csv')
    pooled = scores[scores.horizon.astype(str).eq('all')]
    flu = pooled[pooled.target.eq('wk inc flu hosp')]
    rows = []
    for r in results.values():
        if r['seed'] != 42:
            continue
        row = {'Variant': r['name'], 'Flu ratio*': f"{r['flu_objective']:.3f}", 'Admissions ratio*': f"{r['admissions_objective']:.3f}"}
        for season in ('2023-2024', '2024-2025', '2025-2026'):
            for geo in ('states_dc', 'US'):
                value = flu[(flu.variant == r['name']) & (flu.seed == 42) & (flu.season == season) & (flu.geography == geo)].iloc[0]
                row[f'{season} {geo}'] = f"{value.wis_ratio:.3f}"
        rows.append(row)
    sweep = pd.DataFrame(rows)
    repeats = []
    for name in completion['finalists']:
        r = [x for x in results.values() if x['name'] == name]
        row = {'Variant': name}
        for key, label in [('flu_objective', 'Flu'), ('admissions_objective', 'Admissions')]:
            values = [x[key] for x in r]
            row[label + ' ratio mean'] = f'{np.mean(values):.3f}'
            row[label + ' seed range'] = f'{min(values):.3f}–{max(values):.3f}'
        repeats.append(row)
    repeated = pd.DataFrame(repeats)
    detail = pooled[pooled.variant.isin(completion['finalists'])].groupby(['variant', 'target', 'season', 'geography']).agg(
        seeds=('seed', 'nunique'), wis=('wis', 'mean'), ensemble_wis=('ensemble_wis', 'first'),
        wis_ratio=('wis_ratio', 'mean'), ratio_min=('wis_ratio', 'min'), ratio_max=('wis_ratio', 'max'),
        coverage50=('interval_coverage_50', 'mean'), coverage95=('interval_coverage_95', 'mean'),
        ensemble_coverage50=('ensemble_interval_coverage_50', 'first'), ensemble_coverage95=('ensemble_interval_coverage_95', 'first'),
        n=('n', 'first')).reset_index()
    detail.to_csv(ROOT / 'finalist_summary.csv', index=False)
    flu_detail = detail[detail.target.eq('wk inc flu hosp')].copy()
    for c in ['wis', 'ensemble_wis']:
        flu_detail[c] = flu_detail[c].map(lambda x: f'{x:.2f}')
    for c in ['wis_ratio', 'ratio_min', 'ratio_max']:
        flu_detail[c] = flu_detail[c].map(lambda x: f'{x:.3f}')
    for c in ['coverage50', 'coverage95']:
        flu_detail[c] = flu_detail[c].map(lambda x: f'{100*x:.1f}%')
    duration = sum(json.loads((Path(r['directory']) / 'manifest.json').read_text())['elapsed_seconds'] for r in results.values())
    text = f'''# B0 experiments 1–3: full staged comparison

Completed {completion['runs']} full runs / {completion['fits']} season fits. Each fit uses 50 epochs,
width 64, latent dimension 16, eight independent training draws, and 2,048 evaluation
draws per origin. Sum of fit/evaluation run times: {duration/60:.1f} minutes, excluding
hub rescoring. All 23 quantiles, fixed truth, and original ensemble-supported tasks
are retained. R scoringutils scores each new forecast; existing ensemble scores
are reused on exactly matching keys. Admission quantiles retain half-up rounding.

## Findings

The most promising repeated candidate is fourth-root population scaling, geographic
metadata, twelve weeks, and the dynamics bundle, retaining the original
`[1,.1,.1,.1,.1,.1]` loss weights. Its mean aggregate influenza ratio across seeds
is 0.937 (range 0.916–0.950), versus baseline 1.106 (1.073–1.150).
Its three-admission ratio is 0.917 (0.883–0.947), versus baseline 1.206.
These are balanced geographic/season aggregates, not a pooled count-space score.

The benefit is not uniform. Three-seed mean influenza WIS/ensemble ratios for this
candidate are 1.050 / 0.875 / 0.936 for states/DC and 1.028 / 0.843 / 0.940 for US
across 2023–24 / 2024–25 / 2025–26. Baseline states/DC ratios are
0.851 / 0.772 / 0.940: most of the aggregate improvement comes from US, while the
first two seasons lose state performance. State 95% interval coverage declines
to approximately 79% / 77% / 81%; US coverage improves to 90% / 87% / 94%.
This is not an unconditional replacement for the state-focused baseline.

Twelve weeks without dynamics looked best for influenza at seed 42, but its
three-seed mean ratio is 1.028, showing substantial initialization sensitivity.
Twenty-six weeks hurt at seed 42. Balanced admission weights slightly helped the
multi-admission objective but hurt flu; flu-only supervision did not help flu.
Those latter variants were screened at one seed, so their ranking is preliminary.
A balanced-weights plus dynamics combination was not tested in this staged sweep.

The final audit passed all 42 fitted-fold artifact checks, with unchanged native
loss scales across variants, finite parameters/forecasts, ordered quantiles,
valid output units, and correct season masks. Replacing all held-out observations
with extreme values left fitting inputs, labels, and both scaler types unchanged
for each of the three seasons and all 8/12/26-week lookbacks. See
`holdout_audit.json` and `artifact_audit.json`. Evaluation observations are excluded
from fitting; past evaluation-season observations are available as forecast context.

## Seed-42 staged sweep

Ratios below one beat the ensemble. Each geographic column is mean native-unit
WIS divided by ensemble WIS on exactly the same units. The star marks an aggregate
geometric mean: equal seasons and geography groups within each target, then equal
target weight for the three-admission objective. Only available ensemble task sets
are included; COVID and RSV do not have three complete seasons of hub comparisons.

{table(sweep, list(sweep.columns))}

The selected scaling branch was `{selection['representation']['name']}`; the history
branch was `{selection['history']['name']}`; adding dynamics selected
`{selection['features']['name']}`. Loss weights were compared on that feature branch.
The full candidate sweep selected `{selection['flu_finalist']['name']}` for influenza
and `{selection['admissions_finalist']['name']}` for the admissions objective.
This is a staged sweep, not a Cartesian search of every interaction.

## Three-seed repeats

Baseline and distinct finalists were run with seeds 42, 43, and 44. These are means
and ranges of independently trained models' score ratios, not a pooled predictive
mixture and not confidence intervals. Finalist identities were selected at seed 42;
additional seeds assess initialization sensitivity.

{table(repeated, list(repeated.columns))}

## Influenza by geography and season, averaged over three seeds

{table(flu_detail, ['variant', 'season', 'geography', 'wis', 'ensemble_wis', 'wis_ratio', 'ratio_min', 'ratio_max', 'coverage50', 'coverage95'])}

Full six-channel, geography, and horizon results are in `all_hub_scores.csv`;
`finalist_summary.csv` includes all scored targets and mean coverage for finalists.
Flu-only supervision still generates all six channels, but its auxiliary forecasts
are unsupervised outputs, not evidence of a trained multi-pathogen forecast.

## Interpretation limits and reproducibility

All three folds have been examined before this sweep, and branch selection uses
their exploratory scores. Only the 2025–26 fold trains exclusively on earlier
seasons. Even that fold uses finalized surveillance inputs, not operational vintages.
These results cannot establish prospective performance. A fixed frozen population
table supplies every retrospective season; US predictions remain native forecasts.
No additional calibration or early stopping was fitted.

`protocol.json` records the selection rule, `selection.json` records branch choices,
`results.json` records switches and objectives, and each run has its code/data hashes,
training histories, model checkpoints, forecasts, native-data scores, and hub scores.
Hub scorer inputs, R versions, task support, and logs are retained in `hub_scores/`.
The source comparison is `data/evaluation/b0_hub_comparison`; no new Hub data was fetched.

Run: `.venv/bin/python experiments/b0/run_b0_experiments.py`.
Report: `.venv/bin/python experiments/b0/report_b0_experiments.py`.
'''
    (ROOT / 'report.md').write_text(text)
    Path('docs/results/b0-full-experiments.md').write_text(text)
    print(ROOT / 'report.md')


if __name__ == '__main__':
    main()
