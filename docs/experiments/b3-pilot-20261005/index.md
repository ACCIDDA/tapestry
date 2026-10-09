# B3 exploratory pilot: reporting treatments, new forecasters and historical ILI

5 October 2026. The user authorized a few hours of exploratory jobs before a
larger random search. This supersedes the earlier flu-only and forward-primary
proposal. **All six admission/ED targets remain in fitting and evaluation.**
The 1,000-configuration search is not launched.

<!-- model-choices:start -->
## Model choices

What this report's models were trained on, how errors and corrections were made, what they learned to predict, which seasons they were trained and evaluated on, and what they were scored on, for every configuration (generated 9 October 2026 from the saved scenario strings).

Each heading links to its explanation in [Model choices A–F](../../reference/model-choices.md). One row per group of configurations with identical choices.

| Configurations | [Training histories (A)](../../reference/model-choices.md#a-training-histories) | [Error source (B)](../../reference/model-choices.md#b-error-source) | [Error signals (C)](../../reference/model-choices.md#c-error-signals) | [Correction model (D)](../../reference/model-choices.md#d-correction-model) | [Evaluation inputs (E)](../../reference/model-choices.md#e-evaluation-inputs) | [Forecast view (F)](../../reference/model-choices.md#f-input-view) | [Prediction labels](../../reference/model-choices.md#labels-and-folds) | [Evaluated season ← training seasons](../../reference/model-choices.md#labels-and-folds) |
|---|---|---|---|---|---|---|---|---|
| 13 configurations: pathogen_distance_quantile_head, pathogen_distance_finalized, target_neighbors_kinsa_quantile_head, … | final values | each fold's latest training season | admissions | tree on synthetic examples; newest 8 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 5 configurations: pathogen_distance_reported, target_neighbors_kinsa_reported, series_mlp_reported, … | actual archived reports | each fold's latest training season | admissions | tree on synthetic examples; newest 8 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 7 configurations: pathogen_distance_admission_errors, pathogen_distance_early_errors_recent_reports, pathogen_distance_synchronous_half_episodes, … | final values with artificial reporting errors | each fold's latest training season | admissions | tree on synthetic examples; newest 8 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| pathogen_distance_joint_two_weeks, target_neighbors_kinsa_joint_two_weeks, random_recipe_3 | final values with artificial reporting errors; plus reconstruction labels | each fold's latest training season | admissions | tree on synthetic examples; newest 8 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 2 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| pathogen_distance_joint_four_weeks, target_neighbors_kinsa_joint_four_weeks | final values with artificial reporting errors; plus reconstruction labels | each fold's latest training season | admissions | tree on synthetic examples; newest 8 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| pathogen_distance_synthetic_tree_pipeline, target_neighbors_kinsa_synthetic_tree_pipeline | final values with artificial reporting errors; corrected by the cross-fitted correction model | each fold's latest training season | admissions | tree on synthetic examples; newest 8 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| pathogen_distance_real_tree_pipeline, target_neighbors_kinsa_real_tree_pipeline | final values with artificial reporting errors; corrected by the cross-fitted correction model | each fold's latest training season | admissions | tree on real examples; newest 8 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| fixed_forecaster_real_tree | final values | none | admissions | tree on real examples; newest 8 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| fixed_forecaster_real_mlp | final values | none | admissions | mlp on real examples; newest 8 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| fixed_forecaster_pretrained_mlp | final values | none | admissions | mlp on synthetic then real examples; newest 8 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
<!-- model-choices:end -->
## Evaluation and scores

Both folds receive equal weight. The first trains on finalized 2022–23,
2023–24 and 2024–25 future outcomes and evaluates 2025–26. The second trains on
2022–23, 2023–24 and 2025–26 outcomes and evaluates 2024–25; it is retrospective.
Training-input changes below never change these finalized future labels. Joint
models additionally learn finalized recent-history labels. Both seasons have
already influenced development, so results are exploratory selection evidence.

Every model receives the new standard per-Hub-deadline reported evaluation
inputs, with scheduled finalized fills where archives are absent. ED availability
is the chosen T-0 regime. Missing final truth remains unavailable. This is not a
strict historical publication reconstruction; report fill shares. Correction is
computed on each Hub's own input episode, including holiday deadline differences.

Keep the existing native-unit six-target ensemble-relative WIS ranking: admission
weight 1, ED weight 0.5, states/DC 80%, US 20%, equal seasons. Also report flu
admission **native and log(1 + count)** ensemble-relative WIS using the same
location and season aggregation. No flu-only restriction and no new composite
mixing these metrics. Lower is better; 1 equals the matched Hub ensemble. The
retrospective frozen support contains only flu/COVID admissions; raw full-window
scores still cover all six targets, as in the standard evaluator.

Every fitted model is evaluated raw, fully admission-corrected, and with an equal
mixture of forecasts from raw and admission-corrected histories. ED remains raw.
Sampled models use 512 members, including 256 from each history in the mixture.
Direct-quantile models export exact submitted quantiles. Their mixture uses
2,048 deterministic probability points per distribution with linear interpolation
between submitted quantiles and constant tails; this numerical approximation is
explicit and is not a fitted dependence model. No joint trajectories are claimed
for direct quantiles. The same forecast random stream is used for paired sampled
comparisons. Alternative correction distributions reuse one forecaster checkpoint.

## Pilot design

[Exact scenarios and labels](design.json) and [scenario strings](scenarios.txt)
are generated by `scripts/plan_b3_pilot.py`. There are **37 configurations, seeds
42 and 43, two seasonal folds: 74 manager runs and 148 fold evaluations** before
counting the three evaluation-history views. Inner selection and refitting add
training work. The validation batch uses seven representative configurations
with two epochs and no early stopping; these are execution checks, not evidence
of forecast quality.

The research pilot caps training at 120 epochs with patience 15, batch size 8,
128 training samples for stochastic models and 256 validation samples. This is
an explicitly shorter screening recipe than the old 300-epoch/patience-30 fits.
The new shared quantile models do not sample their training loss. All forecasts
retain four future horizons, no finality input channel and no artificial
missingness regularizer. The four random recipes use fixed draw seed 20261005.

- Eighteen configurations cross two known MLP arrangements with nine treatments.
  The arrangements are independent pathogen MLPs with distance exchange and no
  covariates, and independent target MLPs with neighbor exchange and summarized
  Kinsa. Each compares unchanged finalized histories; reported histories with
  finalized fills; half-strength admission errors; half-strength early-season
  admission errors with actual/proxy recent-season reports; synchronized
  admission errors in half the episodes; joint reconstruction of two weeks at
  weight 0.05 or four weeks at weight 0.25; cross-fitted synthetic-tree-corrected
  training; and cross-fitted real-pair-tree-corrected training.
- Two matched quantile-head configurations retain each existing MLP arrangement
  and change the stochastic output/loss to 23 ordered quantiles and weighted
  pinball loss. This isolates the output change from the new shared encoders.
- Three additional fixed pathogen-MLP configurations compare real-pair tree,
  real-pair neural, and synthetic-pretrained then real-pair neural correctors.
  They retrain the same deterministic forecaster recipe with the same seed;
  the forecaster receives unchanged finalized training inputs. Only correction
  replays differ, and raw forecasts should coincide.
- Ten configurations compare shared per-series quantile MLP and residual time
  mixer, each from scratch, with historical ILI pretraining, with joint ILI
  forecasting, with reported training inputs, or with a damped growth anchor.
  These use width 32, a shared fit across six labels, rank-4 source adjustments,
  and a small gated all-target context contribution. They do not use spatial
  messages or auxiliary covariates. Historical ILI uses source identity 6;
  modern targets use identities 0–5. Shared weights, not fictitious historical
  admission labels, transfer the information.
- Four fixed-seed random recipes explore finalized, reported, empirical-error
  and joint training with selected histories 8/10/12, widths 32/64, learning
  rates 0.0005/0.001, decay 0/0.0001 and spatial none/distance/neighbors.

The growth anchor adds damped transformed recent growth using observations two
weeks apart only when both exist. Missing endpoints imply zero added trend.
The native-Q95 scientific loss weights remain unchanged in this pilot. Direct
quantile models use the corresponding 23-level WIS-equivalent pinball loss.

## Reporting development and assumptions

`input_mode=reported` is now an allowed training scenario. It invokes the same
archive-then-finalized-fill episode rule as evaluation. The pilot excludes
provisional reported context values from known-final observations used in loss
scales. Targets and masks used as prediction labels remain unchanged.

Empirical-error treatments retain the training-only donor season: 2024–25 for
forward evaluation, 2025–26 for the retrospective fold. They copy signed
age-aligned reporting errors, never prediction labels, and support admissions-only
perturbations or all signals. The early-season treatment perturbs 2022–23 and
2023–24 and uses actual/proxy inputs in the latest permitted training season.
The synchronized treatment shares donor dates across locations. Conditional
matching still uses the existing training-season phase features. Missing donor
pairs imply no transported error, not evidence of zero reporting error. Adding
2023–24 as a donor is not yet in this pilot.

The ported tree corrector uses 100 histogram-boosting iterations, seven leaves,
minimum leaf 100 and penalty 10. It reads reported paths, availability, calendar,
location and optional covariates. Only admissions are fitted and corrected.
The neural comparison uses the same feature matrix and a 32-wide residual MLP,
200 minibatch updates. Pretraining supplies 200 additional updates on synthetic
reported/finalized histories before 200 updates on the identical real pairs.
This is **synthetic-revision pretraining**, not masked reconstruction and not
historical ILI forecasting pretraining. Those are distinct methods.

Real supervision requires an actually archived input cell and an archived
12-week-age reference value published by the fitting cutoff. Finalized fills
never supervise reporting errors. Age 12 is an approximate maturity endpoint,
not guaranteed final truth. Labels from inner validation dates are removed.
Cross-fitting removes each corrected origin block's entire context-date union
from corrector labels, including its occurrence in other windows. This is
four-way temporal-block cross-fitting, not leave-one-season-out; there is only
one recent donor season for real reporting supervision. Final correctors use
all permitted fitting pairs, with no online held-out-season updates.

The two corrected-training pilot arms share a corrected-history handoff; one
uses synthetic-pair trees and the other real-pair trees. They are not presented
as separate mechanisms merely because of the `corrected`/`two_stage` names.
All three evaluation views remain available for both. Nowcast diagnostics compare
raw/corrected newest-week MAE, four-week mean-level MAE and two-week log-growth
MAE on identical genuine pairs. Log growth is log(1 + mean of the last two weeks)
minus log(1 + mean of the preceding two weeks). Tail-of-
season cells lacking a mature reference within the season are excluded. This
initial tree/MLP corrector does not produce predictive intervals; coverage is
reported for the downstream forecasts instead.

## Historical ILI, included now

`data/processed/historical_ili.npz` freezes **32,575 observed cells across 55
source geographies**, reference dates October 9, 2010–July 23, 2022. All selected
reports were published before August 1, 2022. Percent values are converted to
proportions; gaps remain missing. The source and output hashes are recorded in
`historical_ili.json`, and the experiment manager pins the auxiliary dataset hash.
This separates old ILI supervision from all modern held-out seasons.

The model learns the next four ILI weeks from each location's own history,
with at least half the input window observed and all four labels observed.
The data yield 31,771 twelve-week examples; those examples are correlated and
are not independent epidemic seasons. Pretraining uses 200 updates of 64
historical examples. Joint training adds one 64-example historical batch per
modern optimizer step, with loss weight 0.25 after within-example scale
normalization. These are starting settings, not tuned weights. Modern training
exposure is held fixed. ILI is a syndromic source whose definition and reporting
may change; transfer assumes useful shared dynamics, not identical measurement.

## Launch and follow-up

The cluster source lives in
`/proj/jlessler/projects/tapestry-all/tapestry-b3-pilot-20261005`, separate from
the shared checkout. Source is copied into each experiment's manager snapshot.
Use only L40 GPUs for this pilot, four processes per GPU and four GPUs. The
research allocation stops after four hours; incomplete runs are not silently
extended. Notifications use the standard launcher's completion summary.

```bash
cd /proj/jlessler/projects/tapestry-all/tapestry-b3-pilot-20261005
export PYTHONPATH=src
# Helper calls the manager plan once with every exact Scenario string and seeds 42 43:
.venv/bin/python scripts/plan_b3_pilot.py --plan
# Equivalent manager plan after generating scenarios.txt:
.venv/bin/python -m chromantis.experiment.planner plan -e b3-pilot-20261005 \
  -s $(cat docs/experiments/b3-pilot-20261005/scenarios.txt) --seeds 42 43 --device cuda
LANES=4 GPUS=4 sbatch --job-name=b3-pilot-20261005 --array=0-3 \
  --time=04:00:00 scripts/jlessler.sbatch b3-pilot-20261005
.venv/bin/python -m chromantis.experiment.planner status -e b3-pilot-20261005
.venv/bin/python -m chromantis.experiment.planner rank -e b3-pilot-20261005 --no-plots
```

Plan with either the helper or the equivalent command, not both. Do not replan
an active experiment. `status` prints resubmission commands; a resubmission is
additional compute beyond this initial four-hour allocation. For a stopped
partial pilot, `rank --allow-incomplete --no-plots` is explicitly exploratory:
`pilot-rankings.csv` includes a seed-count column, so single-seed candidates
must not be compared as though replicated. The standard native ranking remains
unchanged. `pilot-target-season-scores.csv` and `pilot-seed-scores.csv` preserve
all three history treatments and both flu scales.

After observing runtime and paired results, refine the large random design with
the user's proposed treatment shares: 20% finalized, 25% artificial errors,
20% actual/proxy archived training, 15% joint, 10% corrected synthetic training,
10% real-corrector/forecaster pipelines. Architecture and historical-transfer
choices cross those treatments where compatible. Do not allocate 1,000 expensive
runs before learning whether the new methods are stable and correctly wired.

### Validation launches

The first check is job 3938849, seven configurations × two seeds, two refit
epochs per component. A second check (job 3939192) exercises inner validation, the real-pair
pipeline, the early-error/actual-recent treatment, the old-encoder quantile head
and ILI pretraining, with three epochs and patience one. These short fits are
excluded from research rankings. Commands from the cluster directory above:

```bash
.venv/bin/python scripts/plan_b3_pilot.py --experiment b3-pilot-check-20261005 --smoke --plan
LANES=4 GPUS=2 sbatch --job-name=b3-pilot-check-20261005 --array=0-1 \
  --time=00:30:00 scripts/jlessler.sbatch b3-pilot-check-20261005
.venv/bin/python -m chromantis.experiment.planner status -e b3-pilot-check-20261005
.venv/bin/python -m chromantis.experiment.planner rank -e b3-pilot-check-20261005 --no-plots

.venv/bin/python scripts/plan_b3_pilot.py --experiment b3-pilot-inner-check-20261005 --smoke-inner --plan
LANES=4 GPUS=2 sbatch --job-name=b3-pilot-inner-check-20261005 --array=0-1 \
  --time=00:30:00 scripts/jlessler.sbatch b3-pilot-inner-check-20261005
.venv/bin/python -m chromantis.experiment.planner status -e b3-pilot-inner-check-20261005
.venv/bin/python -m chromantis.experiment.planner rank -e b3-pilot-inner-check-20261005 --no-plots
```

The partial validation ranking is checked with CPU Slurm job 3939274 using
`planner rank -e b3-pilot-check-20261005 --allow-incomplete --no-plots`.

### Research submission

Research array **3939394** was submitted on 5 October 2026: four L40 workers,
four lanes each, four-hour time limit per worker. It depends on successful
completion of validation arrays 3938849 and 3939192 and ranking check 3939274:

```bash
LANES=4 GPUS=4 sbatch --job-name=b3-pilot-20261005 --array=0-3 \
  --time=04:00:00 --dependency=afterok:3938849:3939192:3939274 \
  --kill-on-invalid-dep=yes scripts/jlessler.sbatch b3-pilot-20261005
```

The native six-target ranking remains the primary manager ranking. The separate
`pilot-rankings.csv` retains raw/half/full correction views, the same six-target
native score, and influenza admission native/log scores. Scores from the short
validation arrays must not be used to select the research winner.

Validation outcome: both validation arrays completed with exit code 0 (14 and
8 configuration/seed runs respectively), and ranking job 3939274 completed with
exit code 0. The focused scientific checks passed for WIS/quantile-loss equality,
ordered bounded predictions, masked-input exclusion, mixture semantics,
unchanged prediction labels, admissions-only perturbations and donor/validation
isolation. On 12 completed validation seed runs, raw-view six-target scores
matched the existing manager score with maximum absolute difference 6.7e-16.
All three evaluation-history views and both influenza score scales were present.
The validation scores are deliberately not interpreted as research findings.

## Completed pilot analysis

[Results, paired comparisons, graphs and scoring commands](analysis/README.md). All 74 pilot runs completed. The analysis retains the existing rank and separate influenza admission native/log scores; no 1,000-configuration search was submitted.
