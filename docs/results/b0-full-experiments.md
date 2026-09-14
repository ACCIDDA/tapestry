# B0 experiments 1–3: full staged comparison

The [configuration evaluation report](b0-configuration-comparison.md) includes
the original B0 alongside this sweep, with stable identifiers, downloadable
rankings, and inline projection fans and scoring diagnostics for every target/season.

Completed 14 full runs / 42 season fits. Each fit uses 50 epochs,
width 64, latent dimension 16, eight independent training draws, and 2,048 evaluation
draws per origin. Sum of fit/evaluation run times: 31.0 minutes, excluding
hub rescoring. The historical staged-search results below used 23 quantiles.
The linked current report now scores and exports only five quantiles on the same
frozen truth and original ensemble-supported tasks; consult it for current rankings. The current linked report freshly rescores all candidates and ensembles through
the full EpiBench command on exactly matching keys. The original staged search
used the earlier direct R bridge; its selection history is retained here. Admission quantiles retain half-up rounding.

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

| Variant | Flu ratio* | Admissions ratio* | 2023-2024 states_dc | 2023-2024 US | 2024-2025 states_dc | 2024-2025 US | 2025-2026 states_dc | 2025-2026 US |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 1.073 | 1.176 | 0.859 | 1.808 | 0.750 | 1.188 | 0.832 | 1.327 |
| sqrt_geo | 0.944 | 0.932 | 0.938 | 0.912 | 0.973 | 0.987 | 0.928 | 0.928 |
| fourth_root_geo | 0.921 | 0.923 | 0.939 | 0.909 | 0.873 | 0.833 | 0.981 | 1.003 |
| fourth_root_geo_h12 | 0.913 | 0.907 | 0.946 | 0.847 | 0.819 | 0.786 | 1.049 | 1.068 |
| fourth_root_geo_h26 | 1.073 | 1.083 | 1.019 | 0.940 | 1.162 | 1.229 | 1.060 | 1.050 |
| fourth_root_geo_h12_dynamics | 0.950 | 0.883 | 1.110 | 1.068 | 0.856 | 0.805 | 0.949 | 0.950 |
| fourth_root_geo_h12_balanced_admissions | 0.938 | 0.902 | 0.932 | 0.828 | 0.880 | 0.864 | 1.065 | 1.091 |
| fourth_root_geo_h12_flu_only | 0.960 | 1.487 | 0.983 | 0.941 | 0.809 | 0.764 | 1.159 | 1.182 |

The selected scaling branch was `fourth_root_geo`; the history
branch was `fourth_root_geo_h12`; adding dynamics selected
`fourth_root_geo_h12`. Loss weights were compared on that feature branch.
The full candidate sweep selected `fourth_root_geo_h12` for influenza
and `fourth_root_geo_h12_dynamics` for the admissions objective.
This is a staged sweep, not a Cartesian search of every interaction.

## Three-seed repeats

Baseline and distinct finalists were run with seeds 42, 43, and 44. These are means
and ranges of independently trained models' score ratios, not a pooled predictive
mixture and not confidence intervals. Finalist identities were selected at seed 42;
additional seeds assess initialization sensitivity.

| Variant | Flu ratio mean | Flu seed range | Admissions ratio mean | Admissions seed range |
| --- | --- | --- | --- | --- |
| baseline | 1.106 | 1.073–1.150 | 1.206 | 1.176–1.224 |
| fourth_root_geo_h12 | 1.028 | 0.913–1.091 | 0.975 | 0.907–1.027 |
| fourth_root_geo_h12_dynamics | 0.937 | 0.916–0.950 | 0.917 | 0.883–0.947 |

## Influenza by geography and season, averaged over three seeds

| variant | season | geography | wis | ensemble_wis | wis_ratio | ratio_min | ratio_max | coverage50 | coverage95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 2023-2024 | US | 2427.68 | 1468.11 | 1.654 | 1.436 | 1.808 | 11.1% | 45.3% |
| baseline | 2023-2024 | states_dc | 30.21 | 35.48 | 0.851 | 0.845 | 0.859 | 46.0% | 87.9% |
| baseline | 2024-2025 | US | 5294.86 | 4182.15 | 1.266 | 1.182 | 1.429 | 18.2% | 49.7% |
| baseline | 2024-2025 | states_dc | 76.45 | 99.04 | 0.772 | 0.750 | 0.800 | 48.5% | 91.0% |
| baseline | 2025-2026 | US | 4005.47 | 2788.00 | 1.437 | 1.327 | 1.649 | 21.7% | 56.5% |
| baseline | 2025-2026 | states_dc | 60.87 | 64.73 | 0.940 | 0.832 | 1.058 | 43.9% | 87.7% |
| fourth_root_geo_h12 | 2023-2024 | US | 1738.43 | 1468.11 | 1.184 | 0.847 | 1.447 | 40.8% | 86.7% |
| fourth_root_geo_h12 | 2023-2024 | states_dc | 40.57 | 35.48 | 1.143 | 0.946 | 1.317 | 37.0% | 79.9% |
| fourth_root_geo_h12 | 2024-2025 | US | 3657.69 | 4182.15 | 0.875 | 0.753 | 1.085 | 38.3% | 90.1% |
| fourth_root_geo_h12 | 2024-2025 | states_dc | 88.94 | 99.04 | 0.898 | 0.818 | 1.056 | 36.8% | 78.6% |
| fourth_root_geo_h12 | 2025-2026 | US | 3047.75 | 2788.00 | 1.093 | 0.973 | 1.238 | 56.5% | 90.8% |
| fourth_root_geo_h12 | 2025-2026 | states_dc | 68.61 | 64.73 | 1.060 | 0.974 | 1.157 | 36.1% | 78.5% |
| fourth_root_geo_h12_dynamics | 2023-2024 | US | 1508.65 | 1468.11 | 1.028 | 0.779 | 1.235 | 43.1% | 89.7% |
| fourth_root_geo_h12_dynamics | 2023-2024 | states_dc | 37.27 | 35.48 | 1.050 | 0.904 | 1.137 | 37.4% | 79.4% |
| fourth_root_geo_h12_dynamics | 2024-2025 | US | 3525.63 | 4182.15 | 0.843 | 0.736 | 0.988 | 37.3% | 86.7% |
| fourth_root_geo_h12_dynamics | 2024-2025 | states_dc | 86.68 | 99.04 | 0.875 | 0.795 | 0.975 | 33.9% | 76.8% |
| fourth_root_geo_h12_dynamics | 2025-2026 | US | 2620.06 | 2788.00 | 0.940 | 0.928 | 0.950 | 61.3% | 93.8% |
| fourth_root_geo_h12_dynamics | 2025-2026 | states_dc | 60.61 | 64.73 | 0.936 | 0.923 | 0.949 | 38.3% | 81.2% |

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
