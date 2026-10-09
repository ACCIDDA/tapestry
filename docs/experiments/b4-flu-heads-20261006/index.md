# Matched flu output-head comparison

The user explicitly requested this additional comparison after the overnight refinement cutoff. It is a new authorized experiment, not another autonomous overnight refinement. The feature encoder is held fixed (existing MLP); “four encoders” is interpreted as the four requested output-distribution variants. There was no exact configuration previously tested with both sampled and direct-quantile heads, so the native-score and log-admissions leaders are both anchors. Eight configurations × seeds 42/43 = 16 runs. Every variant is independently retrained; this is not an input replay.

<!-- model-choices:start -->
## Model choices

What this report's models were trained on, how errors and corrections were made, what they learned to predict, which seasons they were trained and evaluated on, and what they were scored on, for every configuration (generated 9 October 2026 from the saved scenario strings).

Each heading links to its explanation in [Model choices A–F](../../reference/model-choices.md). One column per group of configurations with identical choices.

| Choice | #1, #2, #3, #4 | #5, #6, #7, #8 |
|---|---|---|
| [Training histories (A)](../../reference/model-choices.md#a-training-histories) | final values with artificial reporting errors; corrected by the cross-fitted correction model | final values with artificial reporting errors; plus reconstruction labels |
| [Error source (B)](../../reference/model-choices.md#b-error-source) | each fold's latest training season | each fold's latest training season |
| [Error signals (C)](../../reference/model-choices.md#c-error-signals) | admissions | admissions |
| [Correction model (D)](../../reference/model-choices.md#d-correction-model) | tree on real examples; newest 2 week(s) of admissions | tree on synthetic examples; newest 2 week(s) of admissions |
| [Evaluation inputs (E)](../../reference/model-choices.md#e-evaluation-inputs) | real archived Wednesday reports | real archived Wednesday reports |
| [Forecast view (F)](../../reference/model-choices.md#f-input-view) | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used |
| [Prediction labels](../../reference/model-choices.md#labels-and-folds) | latest panel values, next 4 weeks | latest panel values, next 4 weeks + last 4 context weeks |
| [Evaluated season ← training seasons](../../reference/model-choices.md#labels-and-folds) | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
<!-- model-choices:end -->
## Models and objective

1. Existing sampled output head, marginal fair CRPS training.
2. The same sampled head with marginal fair CRPS plus **0.25 × four-week flu admission total WIS**. Quantiles are taken after summing each intact trajectory across the four positive forecast horizons. Total loss is divided by four times the training-only admission Q95 scale. It uses equal seasons, states/DC 80% and US 20%, and only completely observed windows wholly within a season. Recent-history reconstruction horizons never enter this sum. Coverage is diagnostic only, per the user's explicit answer. Early stopping remains based on marginal future-forecast loss, identical to its sampled control. The 0.25 coefficient is an exploratory fixed assumption, not tuned here.
3. Existing direct 23-quantile head: median plus 22 ordered gaps; marginal pinball/WIS training.
4. Compact direct-quantile head: three forecast-specific outputs (median, lower spread, upper spread), with two shared learned monotone 11-point shapes. Positive gaps are cumulatively summed and each side normalized to its outermost quantile; shapes initialize equally spaced. Shape parameters are shared across examples, locations and horizons within a head; spread and median vary by forecast. The compact shape is parameterized in model residual space, then transformed into native admissions/ED units by the unchanged model. It is not a claim of fixed native-unit skewness. This variant has the same marginal pinball/WIS loss as variant 3.

For the native-score anchor, the width-96 MLP jointly forecasts flu admissions and ED from flu histories plus Kinsa. Training histories receive synthetic reporting errors and cross-fitted corrections by trees trained on real report-to-mature pairs. Future labels are finalized flu outcomes. For the log-score anchor, the width-96 MLP jointly forecasts flu admissions and ED from flu histories, using artificial reporting errors and recent-history finalized reconstruction alongside finalized future labels. Its existing spatial-neighbor settings remain unchanged. All other hyperparameters, correction settings, data, and seeds are fixed within each anchor. Different parameter counts mean a shared seed cannot guarantee identical initialization or every random-number draw across head types.

## Seasons, evaluation and interpretation

Forward training seasons: 2022–23, 2023–24, 2024–25; evaluation: 2025–26. Retrospective training: 2022–23, 2023–24, 2025–26; evaluation: 2024–25. These are development folds, not an untouched test. Future finalized labels cover four weeks. Wednesday/holiday reported evaluation inputs fill missing archived cells with finalized values. Each fitted model is evaluated on reported histories, histories with the newest two flu-admission weeks tree-corrected (ED stays reported), and a 50/50 mixture of the two forecast distributions.

The common manager retains native/log flu admissions scores, equal-season weighting, and ED scores. The frozen relative score has no retrospective ED support, so raw ED WIS in both seasons remains necessary. New `distribution-scores.csv` records per-season, per-location, per-view weekly WIS and 50/80/90/95% coverage, simultaneous coverage of all four marginal weekly intervals, and sampled four-week admission-total WIS/coverage. All-four-week interval coverage is an empirical rectangular diagnostic, not a nominally calibrated joint region. It must not be compared to marginal nominal coverage as if they were the same event. Each diagnostic uses completely observed four-week windows, so its weekly WIS support differs from the standard frozen Hub tasks. Aggregate location scores with states/DC 80%, US 20% and then equal seasons; do not pool raw admission counts across geography or mix admission counts with ED proportions.

Cumulative distribution scores are available only for the sampled variants. Direct marginal quantiles do not specify dependence; sums of matching quantiles are deliberately not used. ED proportions are not summed. For a 50/50 evaluation-input mixture, whole sample trajectories are selected from each component before summing. No ensemble-relative cumulative WIS baseline is claimed.

## Execution

Remote checkout: `/proj/jlessler/projects/tapestry-all/tapestry-b4-flu-heads-20261006`. It shares only processed data, metadata, frozen support and environment with the prior pilot. Source is copied separately and pinned by each manager plan; running overnight refinement code is unchanged.

Validation experiment `b4-flu-heads-check-20261006`: all eight variants, seed 42, two epochs, one-epoch patience. Slurm **4006033**, one H100 with four workers. These scores are excluded from research comparisons. Scientific checks cover monotone compact quantiles, sum WIS mathematics and intact member alignment, exclusion of reconstruction horizons, missing-window handling, equal seasons/geography weighting, and saved diagnostic array alignment.

Manager commands (run in the remote checkout):

```bash
export PYTHONPATH=src
# Validation plan and launch, already submitted:
.venv/bin/python scripts/plan_b4_heads.py --smoke --plan
LANES=4 GPUS=1 sbatch --job-name=b4-flu-heads-check-20261006 --array=0-0 \
  --nodelist=g1803jles02 --cpus-per-task=8 --mem=95G --time=01:00:00 \
  scripts/jlessler.sbatch b4-flu-heads-check-20261006
.venv/bin/python -m tapestry.experiment.planner status -e b4-flu-heads-check-20261006
.venv/bin/python -m tapestry.experiment.planner rank -e b4-flu-heads-check-20261006 --no-plots
# Full comparison; helper expands the common planner command below:
.venv/bin/python scripts/plan_b4_heads.py --plan
# Equivalent plan (use only one, never replan a running experiment):
.venv/bin/python -m tapestry.experiment.planner plan -e b4-flu-heads-20261006 \
  -s $(cat docs/experiments/b4-flu-heads-20261006/scenarios.txt) --seeds 42 43 --device cuda
LANES=4 GPUS=2 sbatch --job-name=b4-flu-heads-20261006 --array=0-1 \
  --dependency=afterok:4006033 --kill-on-invalid-dep=yes \
  --nodelist=g1803jles02 --cpus-per-task=8 --mem=95G --time=04:00:00 \
  scripts/jlessler.sbatch b4-flu-heads-20261006
.venv/bin/python -m tapestry.experiment.planner status -e b4-flu-heads-20261006
.venv/bin/python -m tapestry.experiment.planner rank -e b4-flu-heads-20261006 --no-plots
```

Exact scenario strings and variant mappings are in design.json and scenarios.txt. Job state and main submission IDs are recorded below once verified. Monitor both this experiment and the earlier ILINet refinement; do not launch duplicates.

## Submission record

Main array **4006081** accepted, two H100 allocations with four workers each, dependent on successful validation **4006033**. The main plan has exactly eight configurations and 16 seed runs. Four new scientific tests passed; the three previously existing pilot-science checks also passed when run with the first three new checks. The additional saved-diagnostics test passed in the subsequent four-test run. No research score from validation will be included. Four-week total quantiles are computed from continuous native-unit draws; weekly admission quantiles retain the pre-existing integer rounding convention.

## Training completed

All eight short validation runs passed; validation allocation ended successfully at 09:28:24 EDT. Both main allocations started at 09:28:47 and finished at 09:43:34–09:43:35 EDT. All 16/16 research runs completed successfully. Nothing remains queued or training in this comparison. Final common-manager ranking, including distribution diagnostics, started during the user's status check; interpretation remains pending.

Final ranking completed: `ranking-63f2eff0ea53`. [Completed analysis and plots](analysis/README.md), full per-seed/per-season tables and distribution diagnostics are saved. Samples plus cumulative WIS is promising specifically on the reconstruction recipe; it is harmful on the real-tree-corrected training recipe. All work for this experiment is complete.
