# B6: search through 5 PM, two models, and today's forecasts

Latest update: the user authorized additional wide-source, season-count and ED-correction fits through approximately 17:00 EDT. This supersedes the original 16:46:43 cutoff. The submission deadline remains 23:00 EDT, target files 22:30.

Plan written 7 October 2026 and revised after the user authorized additions.
The user clarified: search for five hours, not twelve. Production and submission
preparation follow within the twelve-hour deadline. GPU validation was submitted
from the separate B6 checkout; its status and job ID belong in the execution log. Longleaf reported both patron nodes
idle when checked: four L40 GPUs on g1803jles01 and two H100 GPUs on g1803jles02.
This is a snapshot, not a reservation.

<!-- model-choices:start -->
## Model choices

What this report's models were trained on, how errors and corrections were made, what they learned to predict, which seasons they were trained and evaluated on, and what they were scored on, for every configuration (generated 9 October 2026 from the saved scenario strings).

### B6 search

Each heading links to its explanation in [Model choices A–F](../reference/model-choices.md). One row per group of configurations with identical choices.

| Configurations | [Training histories (A)](../reference/model-choices.md#a-training-histories) | [Error source (B)](../reference/model-choices.md#b-error-source) | [Error signals (C)](../reference/model-choices.md#c-error-signals) | [Correction model (D)](../reference/model-choices.md#d-correction-model) | [Evaluation inputs (E)](../reference/model-choices.md#e-evaluation-inputs) | [Forecast view (F)](../reference/model-choices.md#f-input-view) | [Prediction labels](../reference/model-choices.md#labels-and-folds) | [Evaluated season ← training seasons](../reference/model-choices.md#labels-and-folds) |
|---|---|---|---|---|---|---|---|---|
| 18 configurations: A, A_width192, A_width192_half_lr, … | final values with artificial reporting errors; corrected by the cross-fitted correction model | each fold's latest training season | admissions | tree on real examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 20 configurations: B, B_width192, B_width192_half_lr, … | final values with artificial reporting errors; plus reconstruction labels | each fold's latest training season | admissions | tree on synthetic examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| B0_separate_target_MLP | final values | each fold's latest training season | admissions | tree on synthetic examples; newest 8 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| B3_target_neighbors_Kinsa_real_tree | final values with artificial reporting errors; corrected by the cross-fitted correction model | each fold's latest training season | admissions | tree on real examples; newest 8 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| B4_additional_log_leader | final values with artificial reporting errors; corrected by the cross-fitted correction model | each fold's latest training season | admissions | tree on synthetic examples; newest 4 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| B5_confirmed_candidate_1 | final values with artificial reporting errors; corrected by the cross-fitted correction model | each fold's latest training season | admissions | tree on real examples; newest 1 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |

2 further configuration(s) used training routes deleted on 8 October 2026 and are not described by A–F.

### B6 extension

Each heading links to its explanation in [Model choices A–F](../reference/model-choices.md). One row per group of configurations with identical choices.

| Configurations | [Training histories (A)](../reference/model-choices.md#a-training-histories) | [Error source (B)](../reference/model-choices.md#b-error-source) | [Error signals (C)](../reference/model-choices.md#c-error-signals) | [Correction model (D)](../reference/model-choices.md#d-correction-model) | [Evaluation inputs (E)](../reference/model-choices.md#e-evaluation-inputs) | [Forecast view (F)](../reference/model-choices.md#f-input-view) | [Prediction labels](../reference/model-choices.md#labels-and-folds) | [Evaluated season ← training seasons](../reference/model-choices.md#labels-and-folds) |
|---|---|---|---|---|---|---|---|---|
| 8 configurations: X_A_width192_half_lr_ww_flu, X_A_width192_half_lr_clinical_lab, X_A_width192_half_lr_flusurv, … | final values with artificial reporting errors; corrected by the cross-fitted correction model | each fold's latest training season | admissions | tree on real examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| 8 configurations: X_B_width192_half_lr_ww_flu, X_B_width192_half_lr_clinical_lab, X_B_width192_half_lr_flusurv, … | final values with artificial reporting errors; plus reconstruction labels | each fold's latest training season | admissions | tree on synthetic examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| X_A_two_seasons, X_A_width192_half_lr_two_seasons, X_A_width256_half_lr_two_seasons | final values with artificial reporting errors; corrected by the cross-fitted correction model | each fold's latest training season | admissions | tree on real examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2023-24, 2025-26; 2025-26 ← 2023-24, 2024-25 |
| X_B_two_seasons, X_B_width256_half_lr_two_seasons, X_B5_confirmed_candidate_2_two_seasons | final values with artificial reporting errors; plus reconstruction labels | each fold's latest training season | admissions | tree on synthetic examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2023-24, 2025-26; 2025-26 ← 2023-24, 2024-25 |
| X_A_ED_errors_only, X_A_width192_half_lr_ED_errors_only, X_A_width256_half_lr_ED_errors_only, X_A_ed_inputs_flu_ED_errors_only | final values with artificial reporting errors; corrected by the cross-fitted correction model | each fold's latest training season | admissions and ED | tree on real examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| X_A_ED_corrected, X_A_width192_half_lr_ED_corrected, X_A_width256_half_lr_ED_corrected, X_A_ed_inputs_flu_ED_corrected | final values with artificial reporting errors; corrected by the cross-fitted correction model | each fold's latest training season | admissions and ED | tree on real examples; newest 2 week(s) of admissions and ED | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| X_B_ED_errors_only, X_B_width256_half_lr_ED_errors_only, X_B_actual_all_ED_errors_only, X_B_ed_inputs_flu_ED_errors_only | final values with artificial reporting errors; plus reconstruction labels | each fold's latest training season | admissions and ED | tree on real examples; newest 2 week(s) of admissions | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
| X_B_ED_corrected, X_B_width256_half_lr_ED_corrected, X_B_actual_all_ED_corrected, X_B_ed_inputs_flu_ED_corrected | final values with artificial reporting errors; plus reconstruction labels | each fold's latest training season | admissions and ED | tree on real examples; newest 2 week(s) of admissions and ED | real archived Wednesday reports | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | latest panel values, next 4 weeks + last 4 context weeks | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |

### B6 matched comparison

Each heading links to its explanation in [Model choices A–F](../reference/model-choices.md). One column per group of configurations with identical choices.

| Choice | 14 configurations: A, A_width192, A_width192_half_lr, … | 14 configurations: B, B_width192, B_width192_half_lr, … |
|---|---|---|
| [Training histories (A)](../reference/model-choices.md#a-training-histories) | final values with artificial reporting errors; corrected by the cross-fitted correction model | final values with artificial reporting errors; plus reconstruction labels |
| [Error source (B)](../reference/model-choices.md#b-error-source) | each fold's latest training season | each fold's latest training season |
| [Error signals (C)](../reference/model-choices.md#c-error-signals) | admissions | admissions |
| [Correction model (D)](../reference/model-choices.md#d-correction-model) | tree on real examples; newest 2 week(s) of admissions | tree on synthetic examples; newest 2 week(s) of admissions |
| [Evaluation inputs (E)](../reference/model-choices.md#e-evaluation-inputs) | real archived Wednesday reports | real archived Wednesday reports |
| [Forecast view (F)](../reference/model-choices.md#f-input-view) | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used |
| [Prediction labels](../reference/model-choices.md#labels-and-folds) | latest panel values, next 4 weeks | latest panel values, next 4 weeks + last 4 context weeks |
| [Evaluated season ← training seasons](../reference/model-choices.md#labels-and-folds) | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |

### B6 admissions-only and ED-only specialists

Each heading links to its explanation in [Model choices A–F](../reference/model-choices.md). One column per group of configurations with identical choices.

| Choice | A_hosp_inputs_flu_hosp, A_hosp_inputs_flu, A_ed_inputs_flu_ed, A_ed_inputs_flu | B_hosp_inputs_flu_hosp, B_hosp_inputs_flu, B_ed_inputs_flu_ed, B_ed_inputs_flu |
|---|---|---|
| [Training histories (A)](../reference/model-choices.md#a-training-histories) | final values with artificial reporting errors; corrected by the cross-fitted correction model | final values with artificial reporting errors; plus reconstruction labels |
| [Error source (B)](../reference/model-choices.md#b-error-source) | each fold's latest training season | each fold's latest training season |
| [Error signals (C)](../reference/model-choices.md#c-error-signals) | admissions | admissions |
| [Correction model (D)](../reference/model-choices.md#d-correction-model) | tree on real examples; newest 2 week(s) of admissions | tree on synthetic examples; newest 2 week(s) of admissions |
| [Evaluation inputs (E)](../reference/model-choices.md#e-evaluation-inputs) | real archived Wednesday reports | real archived Wednesday reports |
| [Forecast view (F)](../reference/model-choices.md#f-input-view) | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used | raw, half, corrected (and calibrated with early stopping) scored; forecast_view was not yet a recipe field (added 9 October 2026): see this page for the view each comparison used |
| [Prediction labels](../reference/model-choices.md#labels-and-folds) | latest panel values, next 4 weeks | latest panel values, next 4 weeks + last 4 context weeks |
| [Evaluated season ← training seasons](../reference/model-choices.md#labels-and-folds) | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 | 2024-25 ← 2022-23, 2023-24, 2025-26; 2025-26 ← 2022-23, 2023-24, 2024-25 |
<!-- model-choices:end -->
## Objective and assumptions

Select two defensible flu forecasting systems, evaluate their forecasts on common
historical tasks, fit selected recipes on all four completed seasons, and produce
forecasts using the reports available today. The five-hour search clock starts when the main GPU campaign starts, after code
preparation and data checks. Queue delay is separate. Freeze exploration at that
deadline, then finish production and forecast files. The checked 2026–27 Hub README sets the submission deadline at 23:00 EDT
on 7 October, with reference date 10 October. Aim to deliver files by 22:30 EDT.
This replaces the initial approximate 23:25 assumption based on the user's
twelve-hour estimate.

No artificial missing-report or Kinsa-outage experiments in this campaign. Natural
missing observations and normal publication lags remain. Outpatient flu means
outpatient claims for influenza, not ILINet ILI. Wastewater flu initially means the
existing flu WVAL-like index; the percentile index is an optional exploratory arm.
Two permitted FluSight submissions are taken as the user's stated constraint;
prepare files but do not publish submissions as part of this plan.

Five seeds means 44–48 for each matched configuration. Reuse existing results only
if model settings, code behavior, inputs, prediction labels, splits, and scoring
tasks are compatible. The same seed across different architectures is a matched
repeat identifier, not an identical optimization trajectory. These seeds and
evaluation seasons have already influenced selection; they are not an untouched
test. No claim of definitive generalization to 2026–27 follows from this study.

## Models and common evaluation

A is the width-96 MLP with Kinsa, sampled output, and training histories made by
degrading finalized reports then correcting them with cross-fitted trees learned
from real report-to-mature pairs. B is the width-96 MLP with neighbor exchange,
direct ordered quantiles, artificially degraded training histories, and an
auxiliary reconstruction task for four recent finalized weeks. Both learn
finalized next-four-week flu admissions and ED proportions.

Every new comparison uses two folds: train 2022–23, 2023–24, 2024–25 and evaluate
2025–26; train 2022–23, 2023–24, 2025–26 and evaluate 2024–25. The latter uses later
data and is a retrospective development comparison. Report the forward fold
separately and give it priority in deployment decisions. Preserve internal
training-only epoch selection and the recipe's training budget across matched
arms; flag cap hits rather than calling a capped large model inferior in general.

Evaluate on each historical issuance's available reports, with the established
two-week flu-admission tree correction and reported ED for the main comparison.
Retain the reported-input view for comparability with older forecasters. Record
archive-to-final fallback fractions for every covariate, season and location.
Never compare an older finalized-input score directly with a reported-input score.

Report admissions native and log WIS, ED WIS, interval coverage, and breakdowns by
season, horizon, state/DC versus US. Relative WIS divides by the same frozen Hub
ensemble on identical tasks; lower is better. State/DC weight is 80%, US 20%.
The existing combined score weights admissions 1 and ED 0.5 within season; its
2024–25 component has admissions only. ED decisions therefore also use separately
reported raw ED WIS on common tasks in both seasons. No arbitrary mixing of raw
admission counts and ED proportions. Check the pinned current FluSight scoring
specification before labeling any result an official-score comparison.

## Matched experiment: 140 two-fold runs

All settings other than the stated intervention stay fixed; each row uses five
seeds. One run comprises training and evaluation in both folds, not one fit.

| Comparison | Configurations | Runs |
|---|---:|---:|
| Original A and B | 2 | 10 |
| Width 192 and 256, original output head, separately on A and B | 4 | 20 |
| Residual decoder trunk with 2, 3 or 4 blocks at width 96, separately on A and B | 6 | 30 |
| Add one source: flu wastewater, clinical-lab flu positivity, FluSurv flu, outpatient flu, separately on A and B | 8 | 40 |
| A using actual reports in half its training episodes; B using actual reports in all episodes | 2 | 10 |
| Width 192/256 at half the recipe learning rate, on A and B | 4 | 20 |
| A retrained without Kinsa; B retrained with Kinsa | 2 | 10 |
| Total | 28 | 140 |

The original A/B heads are not both two-residual-block heads. Preserve their
original heads as references. Within each new residual-trunk family, only depth
changes between 2/3/4 blocks. A keeps sampled output and its loss; B keeps ordered
quantiles and its loss. Original-head versus residual-trunk differences change
architecture as well as depth and will be described accordingly. This avoids
silently changing B into a sampled model. Count parameters and record runtime.

Covariates are added to A's existing Kinsa and to B's existing no-extra-source
inputs. Add flu-only source selectors: existing wastewater groups also include
COVID/RSV, and outpatient also includes COVID. Keep masks and normalization aligned
with the selected channels. Use the same summary encoding for all four sources.
FluSurv's archive has a substantial 2020–2025 gap and limited catchment geography;
audit coverage before running, preserve absent data, and distinguish a fallback-heavy
retrospective comparison from evidence that the source helps in real time. Do not
silently omit the arm or invent state-level coverage.

First rescore the saved five-seed A+B and two-seed archived-report variants. Compare
equal-recipe quantile averages for original A+B, A+B trained on actual reports,
A trained on half actual reports+B, and both changed. This is four ensemble
comparisons from fitted models, not four additional training recipes. Complete the
missing seeds before promotion. Future labels stay finalized in every variant.

## Admissions-only and ED-only models: 40 runs

For each A/B training recipe, fit four configurations with five seeds:

1. Predict admissions only, using flu admissions history.
2. Predict admissions only, adding flu ED history as an input.
3. Predict ED only, using flu ED history.
4. Predict ED only, adding flu admissions history as an input.

A retains Kinsa; B retains no external source. Remove the excluded signal from
both the forecaster and its correction features, including its availability mask.
The other signal in configurations 2/4 is a covariate, not a prediction label.
B's auxiliary reconstruction applies only to the selected output signal, so the
other signal does not re-enter as hidden supervision. Admissions correction is
used only when admissions is an input; do not introduce an ED corrector here.
Compare specialists with the corresponding jointly supervised A/B on each target.
An admissions specialist and ED specialist can supply different targets in one
forecasting system. ED-only forecasts are not a second FluSight admissions entry.

## Recover B0–B5 candidates and test ensemble contribution

Recover exact scenarios, training seasons, training-history treatments, prediction
labels, code revisions, checkpoints and saved forecasts. Start with the best flu
admissions native/log and ED candidates within each generation, not its old
six-target combined-score winner. Rescore saved forecasts on identical support,
then replay compatible checkpoints on common reported inputs. Replaying changes
evaluation inputs; it does not retrain a model. Retrain an explicitly documented
recipe when compatible checkpoints are absent. A port to current code is a new
fit and must not be described as the original historical model.

Initial shortlist:

- B0: inspect B0.0 and B0.1, including the width-64 separate-target MLP that leads
  the saved B0.1 aggregate, then choose on flu-only common scores.
- B1: compare its supplied-final-flags, longer-training and gated revision
  candidates; do not select from the initial screen alone.
- B2: include the distance-sharing pathogen MLP without extra covariates and the
  target MLP with neighbors/Kinsa; inspect the later reporting-error variants.
- B3: include the target MLP trained behind real-pair correction and its
  synthetic-pair counterpart; inspect the best flu-specific scores.
- Additional B4: sampled B with cumulative-admissions loss and the best genuinely
  different native/log or ED candidate.
- Additional B5: two candidates selected using the confirmation results and
  complementary forecast errors, not only the original best-of-400 ranking.

Screen saved forecasts broadly, then budget at most eight additional recipes
(one from each B0–B3 generation, two each from B4/B5), up to 40 five-seed runs if
full retraining is necessary. Evaluate each alone and as an equal-weight addition
or replacement in A+B. For a replacement of A, keep B fixed, and vice versa.
Use the same seeds/tasks on both sides. Do not search arbitrary ensemble weights.
Historical models lacking operationally comparable forecasts remain contextual
until replayed or refitted; their old scores cannot qualify them for deployment.

## Limited exploration and choosing two systems

If early complete comparisons justify it, reserve up to two five-seed interaction
recipes: the best width/depth combination and a target specialist with its most
promising external or other-pathogen input. This is explicitly exploratory, not
a full width × depth × source factorial.

Evaluate candidates alone, as five-seed ensembles, and inside fixed A+B. Consider
paired seed differences, season/horizon tradeoffs and coverage alongside means.
Prefer a repeatable improvement of roughly 1% or more, including forward-season
benefit, with no material degradation of the other admissions objective. This is
a practical selection rule, not a significance threshold. Report all exceptions.

System 1 is the strongest stable admissions ensemble, starting from A+B and
replacing B with actual-report-trained B if its five-seed ensemble improvement
holds. System 2 is the best substantively different admissions ensemble, possibly
an admissions specialist or log-admissions-oriented model. It must have reasonable
standalone skill; diversity alone is not enough. Prefer different forecast errors
over a second cosmetic version of the same ensemble. Choose ED outputs separately
using ED scores; do not worsen a FluSight admissions model to improve a combined
admissions/ED rank. If no second system improves on existing alternatives, report
that honestly and retain the strongest distinct baseline.

## Five-hour search budget, followed by production

Preparation before the clock: implement configurable decoder depth, independent
input/output scope and flu-only covariate selection; recover historical artifacts;
audit source coverage and today's latest available weeks; pin inputs and code;
run brief smoke checks of the largest sampled model and both target specialists.
Scientific checks cover loss/label scope, input exclusion, date alignment and
training-only transformations. No broad test suite.

Use both H100s and all four L40s in partition jlessler, one GPU per allocation.
Initial main concurrency: four workers per H100, two per L40 (16 total), adjusted from
observed memory and runtime. Send large sampled width/depth models to H100s and
smaller/direct-quantile models to L40s. Do not double-submit the same configuration
or let workers compete for the same run. All six research allocations have a five-hour
walltime; a campaign deadline handles staggered starts.

| Elapsed search time | Priority |
|---|---|
| 0–30 minutes | Seed-count and actual-report ensemble rescoring; main core jobs and timing |
| 30–210 minutes | Complete matched experiments and specialists; rescore/replay historical candidates; admit justified historical refits and limited interactions |
| 210–270 minutes | Finish five-seed blocks, evaluate broad ensembles and quantile averages versus mixtures |
| 270–300 minutes | Freeze selections, report completed and incomplete contrasts; no new exploration |
| After search | Production fits, today's inference, two model files and report before submission deadline |

There are 180 core runs (140 matched + 40 specialists), up to 40 historical
refits, and at most 10 interaction runs before reuse. The combined core manifest
is generated by `scripts/plan_b6.py --with-specialists --with-historical` and uses the shared manager
experiment `b6-search-full-20261007`. New default-zero decoder depth preserves original
heads; specialist scope is expressed by flu-only loss weights and explicit
admissions/ED input selection. Each research run includes both seasonal folds.

Use actual observed completion rates to admit additional work. Five hours is a
hard search limit, not a claim that every optional arm will fit. Drop optional
interactions before essential matched contrasts. Unfinished five-seed comparisons
are reported as incomplete and cannot qualify a production change. Saved original
A/B production fits remain available while challengers train.

Compare original A and B with seeds 44–48 versus 42–48 on identical saved tasks,
including the A+B ensemble. Also compare multiple five-seed subsets: seeds 42/43
were used in original selection, so an improvement from adding them is not clean
proof of a seed-count effect. If the seven-seed ensemble improves the relevant
scores consistently, fit ten seeds per selected production recipe (42–51), reusing
only compatible existing production checkpoints. Moving the original two recipes
from five to ten seeds costs ten additional fits; changing B's training treatment
requires its own complete set and may cost more. More seeds are not a guaranteed
improvement, and weighting remains equal per recipe regardless of seed counts.

System 1 is explicitly the primary FluSight model: conservative A+B, substituting
B trained on actual archived reports only if the ensemble result holds up. System
2 starts as an equal-recipe ensemble of the additional members that improve the
A+B reference. Test the assembled ensemble and leave-one-member-out contributions:
individual benefits do not guarantee the full collection helps. Retain a distinct
specialist-based second system if it demonstrates a clearer complementary strength.
An ED specialist contributes ED outputs, never fictitious admissions predictions.

Compare quantile averaging with predictive-distribution mixtures for both final
systems. Old artifacts save 23 quantiles, so saved-forecast mixtures reconstruct
piecewise-linear inverse CDFs with constant endpoint tails; they are not recovery
of original sampled trajectories. Re-evaluate sampled finalists if full-draw
mixtures are necessary to resolve the decision. Direct-quantile models inherently
require an explicit interpolation/tail convention. Prefer the mixture's better
coverage when WIS is effectively tied; report the tradeoff rather than asserting
one rule is universally best.

Kinsa is assumed to be available at issuance. Compare A retrained with versus
without Kinsa under normal evaluation inputs, and their contribution inside the
ensemble. Keep Kinsa in production where it helps; no artificial outage evidence
is needed for that decision. Until a matched result supports removal, original
A retains Kinsa. The same principle applies to adding Kinsa to B.

The half-learning-rate extension uses four arms: widths 192 and 256 on each of
A and B. This resolves the user's arithmetic (four arms, twenty runs) while
ensuring the explicitly requested width-256 comparison exists on both recipes.

Production fits train selected recipes on 2022–23 through 2025–26, with finalized
future labels and internal epoch selection, then forecast the actual latest
available reports as of 7 October 2026. Today's 2026–27 outcomes are not selection
labels. Keep the frozen historical comparison panel separate from the refreshed
operational input snapshot; rebuilding historical data changes the experiment.
Determine the current issuance/reference date from the pinned Hub task metadata.

## Deliverables and manager commands

Deliver a manifest with completed/deferred/failed runs, plain-language model
descriptions, five-seed matched effects, each candidate's contribution to A+B,
native/log admission and ED tables, coverage and season/horizon graphs, two model
definitions, production checkpoint locations and today's forecast files. Provide
graphs without visual inspection, as requested by repository instructions. No
guarantee of notifications substitutes for saved artifacts. Keep ntfy completion
notifications enabled, with afterany summary jobs and explicit status commands.

Experiment names: b6-check-20261007, b6-search-full-20261007,
b6-historical-20261007, b6-interactions-20261007, b6-production-20261007.
The core and validation scenario manifests are generated by scripts/plan_b6.py. Execution
will provide the exact planner plan, sbatch launch, status and rank commands for
every experiment, and the scored replay/ensemble commands. Use the existing
planner/scorer and dispatcher; only add a small manifest generator and scheduling
logic required to reserve scoring/production time. Do not launch these commands
from this planning document.

Source records: [B4 study](b4-flu-study-20261006/index.md),
[B4.polish/B5](b4-polish-b5-20261007/index.md),
[B0.1](archive/b0-1-crosses/index.md), [B1](archive/b1-overnight/index.md),
[B2](b-2-t0/index.md), [B3](b3-pilot-20261005/analysis/README.md),
[data availability](../data/index.md), [workflow](../workflow.md).

## Preparation findings

The frozen comparison panel ends on 19 September 2026 (last stored issuance
16 September). Today's forecasts require a separate refreshed operational
snapshot; do not overwrite the pinned comparison panel. Local scientific checks
cover specialist value/mask exclusion in the forecaster and correction features,
flu-only covariate aliases, quantile ordering and checkpoint reload at added depth.

## Execution update: 7 October, 11:55 EDT

See [execution record](b6-search-full-20261007/execution.json). All fifteen short
GPU validation runs passed. The active research experiment is
`b6-search-full-20261007`: 44 configurations × five seeds = 220 two-fold runs,
including eight documented historical recipe refits. H100 array 4129248 and L40
array 4129249 started 11:52:29 EDT with 4h40 allocations, ending around 16:32.
The original research clock began 11:46:43; exploration freezes no later than
16:46:43 EDT. The earlier arrays were stopped to expand the queue; the manager
required a new experiment because an additional B5 candidate needs the historical
ILI input, changing the pinned input set. No scientific guard was bypassed.

The saved-forecast comparison (CPU job 4128575) completed. Its first results are
in [the B6 saved-ensemble report](b6-saved-ensembles-20261007/index.md).
A 15-minute heartbeat continues monitoring and authorized analysis/production
work in this chat, quiet when there is no meaningful change.

Active manager commands (run in `/proj/jlessler/projects/tapestry-all/tapestry-b6-20261007`):

```bash
export PYTHONPATH=src
.venv/bin/python scripts/plan_b6.py --with-specialists --with-historical --experiment b6-search-full-20261007 --plan
LANES=4 GPUS=6 sbatch --job-name=b6-full-h100 --array=0-1 --nodelist=g1803jles02 --cpus-per-task=4 --mem=160G --time=04:40:00 scripts/jlessler.sbatch b6-search-full-20261007
LANES=2 GPUS=6 sbatch --job-name=b6-full-l40 --array=0-3 --nodelist=g1803jles01 --cpus-per-task=2 --mem=100G --time=04:40:00 scripts/jlessler.sbatch b6-search-full-20261007
.venv/bin/python -m tapestry.experiment.planner status -e b6-search-full-20261007
.venv/bin/python -m tapestry.experiment.planner rank -e b6-search-full-20261007 --allow-incomplete --no-plots
```

These launches are already submitted: do not run them again just to inspect.
Resume only unfinished work using status, with time capped by the search deadline.

The current production Hub clone confirms an 11 PM Eastern deadline, rather than
the initial approximate 23:25 deadline. Aim for completed files by 22:30 EDT.
`production/README.md`, `.gitignore`, `pyproject.toml`, and Delphi source-client
updates appeared concurrently in the shared workspace; preserve those external
changes. The production README documents Hub clones, ACCIDDA as team abbreviation,
and unresolved final model names. The NHSN archive refresh succeeded on retry;
other operational sources continue in the background.


## First complete-block scoring batch

CPU job 4133702 scores complete five-seed recipes alone, added to A+B, and as
replacements for A or B. This is forecast combination, not retraining. Initially
seven new blocks were complete: width-256 A at both learning rates, width-192/256
B at both learning rates, and the current-code width-64 B0 recipe refit. Every
recipe uses the two three-season training folds, finalized future labels, and
2024–25/2025–26 evaluation described above. The B0 refit is evaluated on reported
histories; A/B-derived recipes use the two-week admission correction. These input
views are saved per recipe in groups.json.

Until newly fitted references finish, saved B4.polish A/B seeds 44–48 provide the
reference. All scenario fields match except stress_views (the saved fit requests
additional post-fit evaluation views). No training setting is ignored. The builder
also requires matching historical panel hashes, quantile levels, labels, locations
and shared issuance dates. The job inventories complete blocks again at startup.

Commands in the remote B6 checkout (job already submitted):

```bash
export PYTHONPATH=src
.venv/bin/python scripts/ensemble_b6_candidates.py --out data/experiments/b6-candidate-ensembles-1240 --saved-reference --plan
sbatch scripts/b6_candidates.sbatch --out data/experiments/b6-candidate-ensembles-1240 --saved-reference
squeue -j 4133702
.venv/bin/python -m tapestry.experiment.planner status -e b6-search-full-20261007
.venv/bin/python -m tapestry.experiment.planner rank -e b6-search-full-20261007 --allow-incomplete --no-plots
.venv/bin/python scripts/ensemble_b6_candidates.py --out data/experiments/b6-candidate-ensembles-1240 --rank-only
```

Run the ensemble rank command only after the job completes. Outputs use a fresh
folder for each batch to prevent stale score caches after membership changes.


## Numerical ensemble audit and replacement scoring batch

The first candidate scoring job finished. Comparing its reference arrays with the
saved A5+B5 reference found identical dates, truth and masks, but some admissions
quantiles differed by one count at half-integer rounding boundaries. Averaging
within each recipe first introduced extra floating-point rounding; the output
also inherited the first member's storage precision. The builder now accumulates
in float64, averages equal-size recipe groups in one pass, and rounds admissions
once. This changes no model training. Job 4136266 supersedes job 4133702 for
selection; confirm its reference against the earlier saved-ensemble reference.
The first batch suggested a benefit from width-256 A at half learning rate, but
exact gains will be reported from the replacement scores.

Replacement commands, already launched in the remote B6 checkout:

```bash
export PYTHONPATH=src
.venv/bin/python scripts/ensemble_b6_candidates.py --out data/experiments/b6-candidate-ensembles-1255 --saved-reference --plan
sbatch scripts/b6_candidates.sbatch --out data/experiments/b6-candidate-ensembles-1255 --saved-reference
squeue -j 4136266
.venv/bin/python -m tapestry.experiment.planner status -e b6-search-full-20261007
.venv/bin/python -m tapestry.experiment.planner rank -e b6-search-full-20261007 --allow-incomplete --no-plots
.venv/bin/python scripts/ensemble_b6_candidates.py --out data/experiments/b6-candidate-ensembles-1255 --rank-only
```

Operational inference and CSV export scripts are prepared, but have not yet run
on today's completed panel. The episode builder now optionally retains issuances
with no future labels; default research behavior is unchanged. Its date and label
mask check passes. The local machine has 32 GiB RAM: use two panel-build workers,
not the default source-wide concurrency, and an explicit operational output path.


## Expanded scoring batch, 13:10 EDT

Job 4138206 includes the newly complete width-192 A arms and historical B3/B5
recipe refits. The planner found 16 complete five-seed blocks and 59 ensemble
groups. Exact membership is saved at job startup. The first validated width
findings and season graph are in the width report (`b6-candidate-ensembles-1255`, an interim page removed on 9 October 2026).

Commands in the remote B6 checkout; already submitted:

```bash
export PYTHONPATH=src
.venv/bin/python scripts/ensemble_b6_candidates.py --out data/experiments/b6-candidate-ensembles-1310 --saved-reference --plan
sbatch scripts/b6_candidates.sbatch --out data/experiments/b6-candidate-ensembles-1310 --saved-reference
squeue -j 4138206
.venv/bin/python -m tapestry.experiment.planner status -e b6-search-full-20261007
.venv/bin/python -m tapestry.experiment.planner rank -e b6-search-full-20261007 --allow-incomplete --no-plots
.venv/bin/python scripts/ensemble_b6_candidates.py --out data/experiments/b6-candidate-ensembles-1310 --rank-only
```


## Broad ensemble and first specialist checks, 13:25 EDT

Job 4140207 adds the joint five-recipe ensemble A+B+A-width192-half-rate+
A-width256-half-rate+the sampled B5 actual-25%-report recipe. Each recipe has
five seeds. Leave-one-member-out comparisons check whether individually useful
additions remain useful together. The batch also includes newly complete depth,
FluSurv-on-B, and admissions-only specialist blocks. Specialists contribute only
the target they learned. Nothing is selected for production yet.

Commands in the remote B6 checkout; already submitted:

```bash
export PYTHONPATH=src
.venv/bin/python scripts/ensemble_b6_candidates.py --out data/experiments/b6-candidate-ensembles-1325 --saved-reference --broad A_width192_half_lr A_width256_half_lr B5_confirmed_candidate_2 --plan
sbatch scripts/b6_candidates.sbatch --out data/experiments/b6-candidate-ensembles-1325 --saved-reference --broad A_width192_half_lr A_width256_half_lr B5_confirmed_candidate_2
squeue -j 4140207
.venv/bin/python -m tapestry.experiment.planner status -e b6-search-full-20261007
.venv/bin/python -m tapestry.experiment.planner rank -e b6-search-full-20261007 --allow-incomplete --no-plots
.venv/bin/python scripts/ensemble_b6_candidates.py --out data/experiments/b6-candidate-ensembles-1325 --rank-only
```


## Broader six-recipe comparison and Kinsa, 13:40 EDT

Job 4142372 scores 28 complete five-seed blocks and 108 groups at plan time.
It adds the three-residual-block B recipe to the earlier five-recipe ensemble;
removing that member recovers the previous ensemble definition. The newly
complete clinical-lab addition to A, A retrained without Kinsa, B1 and B4 refits
are also included. Actual-report B and several covariate/specialist arms remain
incomplete and cannot yet be promoted.

Commands, already submitted in the remote B6 checkout:

```bash
export PYTHONPATH=src
.venv/bin/python scripts/ensemble_b6_candidates.py --out data/experiments/b6-candidate-ensembles-1340 --saved-reference --broad A_width192_half_lr A_width256_half_lr B5_confirmed_candidate_2 B_blocks3 --plan
sbatch scripts/b6_candidates.sbatch --out data/experiments/b6-candidate-ensembles-1340 --saved-reference --broad A_width192_half_lr A_width256_half_lr B5_confirmed_candidate_2 B_blocks3
squeue -j 4142372
.venv/bin/python -m tapestry.experiment.planner status -e b6-search-full-20261007
.venv/bin/python -m tapestry.experiment.planner rank -e b6-search-full-20261007 --allow-incomplete --no-plots
.venv/bin/python scripts/ensemble_b6_candidates.py --out data/experiments/b6-candidate-ensembles-1340 --rank-only
```


## User-requested extension through 17:00 EDT

The original 220-run campaign completed without failures. The user then requested
new fits to address wide-model source interactions, training-season count and ED
nowcasting, targeting a finish around 5 PM. A separate pinned experiment contains
38 configurations × five seeds = 190 two-fold runs:

- Add each of wastewater flu, clinical-lab flu positivity, FluSurv and outpatient
  flu separately to A/B at widths 192 and 256, using half their original rates:
  16 configurations, 80 runs. A retains Kinsa; B retains its original source setup.
- Retrain A96, A192-half-rate, A256-half-rate, B96, B256-half-rate and the additional
  sampled B5 recipe on the latest two permitted training seasons: 6 configurations,
  30 runs. This removes only 2022–23 from each original three-season fold. Forward
  evaluation 2025–26 trains on 2023–24/2024–25; reverse evaluation 2024–25 trains on
  2023–24/2025–26 and remains retrospective. This tests the effect of adding the
  oldest third season; it does not prove what adding a fourth season will do.
- Paired ED-error-only versus ED-error-plus-correction arms for A96, A192-half,
  A256-half, B96, B256-half, actual-report B, and A/B ED specialists that use both
  flu histories: 16 configurations, 80 runs. Synthetic augmentation now includes
  ED while preserving covariates. Both paired arms use real-report-to-mature trees;
  only the corrected arm adds ED channels to the two-week correction. A corrects
  cross-fitted training histories and evaluation inputs. B keeps its reconstruction
  training on uncorrected histories and applies correction at evaluation. The B
  paired controls also change admissions from synthetic-tree to real-tree correction,
  so comparison with original B is a combined pipeline change. Paired arms isolate
  the additional ED correction. Actual-report B already carries reported ED.

Every fit still learns finalized future labels. Correction trees use only allowed
training-season report-to-mature pairs, with cross-fitting where applied to A's
training inputs. Proportions stay bounded in [0,1]; excluded channels stay excluded.
Eight scientific checks pass. Six short two-fold fits are running as job 4144886;
GPU arrays 4145165/4145166 require those checks to finish successfully and the
original arrays to release their GPUs. Workers stay at four per H100 and two per
L40. The launcher uses an absolute 17:00 deadline, including queue delay. Monitor
actual throughput; do not promise that every arm completes before measuring it.

Validation commands, already submitted:

```bash
export PYTHONPATH=src
.venv/bin/python scripts/plan_b6_extension.py --smoke --experiment b6-extension-check-20261007 --plan
sbatch scripts/b6_extension_check.sbatch
.venv/bin/python -m tapestry.experiment.planner status -e b6-extension-check-20261007
.venv/bin/python -m tapestry.experiment.planner rank -e b6-extension-check-20261007 --allow-incomplete --no-plots
```

Full extension commands, already queued:

```bash
export PYTHONPATH=src
.venv/bin/python scripts/plan_b6_extension.py --plan
LANES=4 GPUS=6 sbatch --array=0-1 --nodelist=g1803jles02 --cpus-per-task=4 --mem=160G --dependency=afterok:4144886,afterany:4129248 --kill-on-invalid-dep=yes scripts/b6_extension.sbatch b6-extension-20261007
LANES=2 GPUS=6 sbatch --array=0-3 --nodelist=g1803jles01 --cpus-per-task=2 --mem=100G --dependency=afterok:4144886,afterany:4129249 --kill-on-invalid-dep=yes scripts/b6_extension.sbatch b6-extension-20261007
.venv/bin/python -m tapestry.experiment.planner status -e b6-extension-20261007
.venv/bin/python -m tapestry.experiment.planner rank -e b6-extension-20261007 --allow-incomplete --no-plots
```

The latest-report audit and its units, coverage and maturity limitations are in
[the reporting-regime report](b6-reporting-regime-20261007/index.md). The current
pattern is heterogeneous upward backfill in both ED and admissions. This does not
establish a publisher schema break or justify a fixed correction multiplier.

## 14:04 EDT: complete core scoring and GPU validation

All 220 core runs completed without failures. Job 4146026 scores all 44 five-seed blocks in 162 groups, including the six-recipe broad ensemble and leave-one-out comparisons. Exact commands are recorded in execution.json candidate_scoring_jobs.

CPU validation 4144886 trained its first two models but full forecast evaluation was too slow for the 20-minute allocation. It was cancelled after changing both extension dependencies to successful GPU validation 4146093. The GPU validation waits for CPU termination, so attempts cannot overlap. This is a compute-placement change, not a scientific design change.

```bash
export PYTHONPATH=src
.venv/bin/python scripts/plan_b6_extension.py --smoke --experiment b6-extension-check-20261007 --plan
LANES=4 GPUS=1 sbatch --job-name=b6-extension-check-gpu --nodelist=g1803jles02 --cpus-per-task=4 --mem=80G --time=00:30:00 --dependency=afterany:4144886 scripts/jlessler.sbatch b6-extension-check-20261007
.venv/bin/python -m tapestry.experiment.planner status -e b6-extension-check-20261007
.venv/bin/python -m tapestry.experiment.planner rank -e b6-extension-check-20261007 --allow-incomplete --no-plots
```

## 14:18 EDT: extension running and complete core report

All six GPU validation runs passed; extension arrays4145165/4145166 now run all six GPUs. Seven extension fits were complete at the first snapshot, with no failure reported. Sampled memory about28GB/H100 and29GB/L40; retain concurrency. The full core report is b6-core-complete-1400/index.md, with explicit fold/input definitions, ED raw WIS in both seasons, coverage CSV and graph. Actual-report B does not yet warrant replacing B because its native admissions gain accompanies worse log and recent ED scores.

Job4147912 evaluates reference, the eight-recipe broad ensemble and eight leave-one-out groups under averaged quantiles and distribution mixtures. This adds three-block A and the ILI-pretrained B5 recipe to the earlier six members. Manager commands:

```bash
export PYTHONPATH=src
.venv/bin/python scripts/ensemble_b6_candidates.py --out data/experiments/b6-broad-eight-1420 --broad A_width192_half_lr A_width256_half_lr B5_confirmed_candidate_2 B_blocks3 A_blocks3 B5_confirmed_candidate_1 --broad-only --rules vincent mixture --plan
sbatch scripts/b6_candidates.sbatch --out data/experiments/b6-broad-eight-1420 --broad A_width192_half_lr A_width256_half_lr B5_confirmed_candidate_2 B_blocks3 A_blocks3 B5_confirmed_candidate_1 --broad-only --rules vincent mixture
squeue -j 4147912
.venv/bin/python scripts/ensemble_b6_candidates.py --out data/experiments/b6-broad-eight-1420 --rank-only
```

## User clarification: 5 PM is a planning target

The user explicitly removed the mandatory freeze: plan to finish around17:00 EDT, but do not stop useful work solely at that time. This supersedes all earlier hard-stop language. Production files remain targeted for22:30 before23:00 submission. Future extension launches no longer contain the absolute17:00 timeout. Already-running arrays retain the old process timer and must be checked near target if unfinished; a script edit cannot retroactively remove it. No additional fits launched by this clarification.

## 14:30 EDT: first extension scoring and eight-member results

43 of190 extension runs completed. Job4152428 scores only complete five-seed blocks (47total at plan,44core+3extension). Exact commands:

```bash
export PYTHONPATH=src
.venv/bin/python scripts/ensemble_b6_candidates.py --additional-experiment b6-extension-20261007 --out data/experiments/b6-extension-score-1430 --plan
sbatch scripts/b6_candidates.sbatch --additional-experiment b6-extension-20261007 --out data/experiments/b6-extension-score-1430
squeue -j4152428
.venv/bin/python -m tapestry.experiment.planner status -e b6-extension-20261007
.venv/bin/python scripts/ensemble_b6_candidates.py --out data/experiments/b6-extension-score-1430 --rank-only
```

Eight-recipe analysis, coverage by season and generated graph: b6-broad-eight-1420/index.md. Mixing improves native admissions/recent ED and coverage, while quantile averaging has better log admissions. Original A removal improves all three aggregate scores; do not infer benefit from simultaneous untested removals. Operational outpatient archive refresh continues; no production fit launched yet.

## 14:45 EDT: first paired ED result

85/190 extension runs complete. The first matched B ED-specialist pair modestly favors evaluation ED correction on2025–26, but not retrospective2024–25 for the specialist alone, and does not improve A+B ED. Explicit training/input/label distinctions and numerical scores are in b6-extension-score-1430/index.md. New scoring job4156204 includes49complete blocks at plan (five extension),171groups.

```bash
export PYTHONPATH=src
.venv/bin/python scripts/ensemble_b6_candidates.py --additional-experiment b6-extension-20261007 --out data/experiments/b6-extension-score-1445 --plan
sbatch scripts/b6_candidates.sbatch --additional-experiment b6-extension-20261007 --out data/experiments/b6-extension-score-1445
squeue -j4156204
.venv/bin/python -m tapestry.experiment.planner status -e b6-extension-20261007
.venv/bin/python scripts/ensemble_b6_candidates.py --out data/experiments/b6-extension-score-1445 --rank-only
```

## 15:00 EDT:123 extension runs complete

Early joint-B pipeline results and forward/reverse season caveat are in b6-extension-score-1445/index.md. Recent-season log admissions worsens despite aggregate gain; no promotion yet. Job4160803 scores51complete five-seed blocks at plan,179groups. Operational outpatient claims pull remains active; no new production fits.

```bash
export PYTHONPATH=src
.venv/bin/python scripts/ensemble_b6_candidates.py --additional-experiment b6-extension-20261007 --out data/experiments/b6-extension-score-1500 --plan
sbatch scripts/b6_candidates.sbatch --additional-experiment b6-extension-20261007 --out data/experiments/b6-extension-score-1500
squeue -j4160803
.venv/bin/python -m tapestry.experiment.planner status -e b6-extension-20261007
.venv/bin/python scripts/ensemble_b6_candidates.py --out data/experiments/b6-extension-score-1500 --rank-only
```

## 15:15 EDT:156 extension runs complete

B strongly benefits from the third training season in both folds. Within revised B, evaluation ED correction gives a small additional gain. Full treatment definitions, limitations and graph are in b6-extension-score-1500/index.md. Job4163123 scores71complete five-seed blocks at plan (27extension),259groups.

```bash
export PYTHONPATH=src
.venv/bin/python scripts/ensemble_b6_candidates.py --additional-experiment b6-extension-20261007 --out data/experiments/b6-extension-score-1515 --plan
sbatch scripts/b6_candidates.sbatch --additional-experiment b6-extension-20261007 --out data/experiments/b6-extension-score-1515
squeue -j4163123
.venv/bin/python -m tapestry.experiment.planner status -e b6-extension-20261007
.venv/bin/python scripts/ensemble_b6_candidates.py --out data/experiments/b6-extension-score-1515 --rank-only
```

## 15:30 EDT: all extension fits complete

All190 extension runs completed;82complete recipes across both campaigns. No need to handle the legacy17:00timer because arrays exited. Interim findings in b6-extension-score-1515/index.md; final full scorer4164639 uses299groups. Production audit found five exact-scenario A/B checkpoints each trained on all four seasons; verified run mapping A=91f17ac66760,B=69f21e6aec42. Dataset/code compatibility still must be checked before reuse.

```bash
export PYTHONPATH=src
.venv/bin/python scripts/ensemble_b6_candidates.py --additional-experiment b6-extension-20261007 --out data/experiments/b6-extension-complete-1530 --plan
sbatch scripts/b6_candidates.sbatch --additional-experiment b6-extension-20261007 --out data/experiments/b6-extension-complete-1530
squeue -j4164639
.venv/bin/python -m tapestry.experiment.planner status -e b6-extension-20261007
.venv/bin/python scripts/ensemble_b6_candidates.py --out data/experiments/b6-extension-complete-1530 --rank-only
```

## 15:50 EDT: conservative production and joint extension ensemble

All82 recipes fully scored; complete report b6-extension-complete-1530/index.md. All six nested season arms favor three seasons. Original A+B production retraining on four seasons is launched with10seeds each; same memory-conscious GPU concurrency. New fits use campaign code/data rather than assuming older checkpoint compatibility. System2 joint12-member comparison checks four extension additions and each member removal under both averaging rules. Exact manager commands:

```bash
export PYTHONPATH=src
.venv/bin/python scripts/plan_b6_production.py --recipes A B --experiment b6-production-primary-20261007 --plan
LANES=4 GPUS=6 sbatch --job-name=b6-production-primary --array=0-1 --nodelist=g1803jles02 --cpus-per-task=4 --mem=160G --time=02:30:00 scripts/jlessler.sbatch b6-production-primary-20261007
LANES=2 GPUS=6 sbatch --job-name=b6-production-primary --array=0-3 --nodelist=g1803jles01 --cpus-per-task=2 --mem=100G --time=02:30:00 scripts/jlessler.sbatch b6-production-primary-20261007
.venv/bin/python -m tapestry.experiment.planner status -e b6-production-primary-20261007
.venv/bin/python -m tapestry.experiment.planner rank -e b6-production-primary-20261007 --allow-incomplete --no-plots
.venv/bin/python scripts/ensemble_b6_candidates.py --additional-experiment b6-extension-20261007 --out data/experiments/b6-broad-twelve-1550 --broad A_width192_half_lr A_width256_half_lr B5_confirmed_candidate_2 B_blocks3 A_blocks3 B5_confirmed_candidate_1 X_A_ED_errors_only X_B_ED_corrected X_A_width256_half_lr_ED_corrected X_B_width256_half_lr_flusurv --broad-only --rules vincent mixture --plan
sbatch scripts/b6_candidates.sbatch --additional-experiment b6-extension-20261007 --out data/experiments/b6-broad-twelve-1550 --broad A_width192_half_lr A_width256_half_lr B5_confirmed_candidate_2 B_blocks3 A_blocks3 B5_confirmed_candidate_1 X_A_ED_errors_only X_B_ED_corrected X_A_width256_half_lr_ED_corrected X_B_width256_half_lr_flusurv --broad-only --rules vincent mixture
squeue -j4165916
.venv/bin/python scripts/ensemble_b6_candidates.py --out data/experiments/b6-broad-twelve-1550 --rank-only
```

Operational outpatient archive pull still active; remaining refresh and panel build must finish before export. No submission published.

## 16:05 EDT: primary fits complete, operational refresh unblocked

All20primary fits completed. Outpatient archive download remains active after nearly3hours; original coordinator PID45025 is paused withSIGSTOP while child64512continues. New independent remaining-source loop tmp/b6_refresh_remaining.py/session5541 writes operational-remaining.log/status.json. Do not resume coordinator blindly: audit outpatient committed snapshot after it ends, record result, then retire coordinator to avoid duplicate acquisition. Frozen historical panel stays unchanged.

Twelve-member ensemble shows tradeoffs; nine-member assembled comparison removes originalA,originalB,correctedA256 jointly. New reference means A_blocks3+X_B_ED_corrected; originalA+B comparisons must use prior matching outputs.

```bash
export PYTHONPATH=src
.venv/bin/python scripts/ensemble_b6_candidates.py --additional-experiment b6-extension-20261007 --out data/experiments/b6-system2-nine-1605 --base-a A_blocks3 --base-b X_B_ED_corrected --broad A_width192_half_lr A_width256_half_lr B5_confirmed_candidate_2 B_blocks3 B5_confirmed_candidate_1 X_A_ED_errors_only X_B_width256_half_lr_flusurv --broad-only --rules vincent mixture --plan
sbatch scripts/b6_candidates.sbatch --additional-experiment b6-extension-20261007 --out data/experiments/b6-system2-nine-1605 --base-a A_blocks3 --base-b X_B_ED_corrected --broad A_width192_half_lr A_width256_half_lr B5_confirmed_candidate_2 B_blocks3 B5_confirmed_candidate_1 X_A_ED_errors_only X_B_width256_half_lr_flusurv --broad-only --rules vincent mixture
squeue -j4167254
.venv/bin/python scripts/ensemble_b6_candidates.py --out data/experiments/b6-system2-nine-1605 --rank-only
```

## 16:20 EDT: System2 selected and production launched

Nine-recipe mixture improves native/log admissions in both developmentfolds and recentED compared with originalA+B. Older fold remains retrospective. Full selection report, coverage and graph: b6-system2-nine-1605/index.md. Retain the directly evaluatednine members for balanced native/log/ED performance. System1 remains originalA+B quantile average; its20productionfits complete. System2 uses90fresh four-season fits, finalized predictionlabels andtenseeds perrecipe. Modelnames asked asynchronously; no publishing.

```bash
export PYTHONPATH=src
.venv/bin/python scripts/plan_b6_production.py --experiment b6-production-system2-20261007 --recipes A_blocks3 X_B_ED_corrected A_width192_half_lr A_width256_half_lr B5_confirmed_candidate_2 B_blocks3 B5_confirmed_candidate_1 X_A_ED_errors_only X_B_width256_half_lr_flusurv --plan
LANES=4 GPUS=6 sbatch --job-name=b6-production-system2 --array=0-1 --nodelist=g1803jles02 --cpus-per-task=4 --mem=160G --time=03:00:00 scripts/jlessler.sbatch b6-production-system2-20261007
LANES=2 GPUS=6 sbatch --job-name=b6-production-system2 --array=0-3 --nodelist=g1803jles01 --cpus-per-task=2 --mem=100G --time=03:00:00 scripts/jlessler.sbatch b6-production-system2-20261007
.venv/bin/python -m tapestry.experiment.planner status -e b6-production-system2-20261007
.venv/bin/python -m tapestry.experiment.planner rank -e b6-production-system2-20261007 --allow-incomplete --no-plots
```

Operational NWSS refreshed; NWSSaux nowactive in separate lane. Outpatientarchive stilldownloading, coordinator45025remainspaused. Audit completionbefore retiringcoordinator.

## 16:30 EDT: production and horizon checks

System2production48/90complete. NWSSaux download stillactive, so its coordinator91003ispaused while child91256continues. Prioritylane tmp/b6_refresh_priority.py/session97281refreshessevenremaining sources, logs operational-priority.log/status.json; Kinsa complete, ILINetactive. Original coordinator45025also remainspaused with outpatientchild64512active. Audit committed snapshots before retiring bothcoordinators; do not blindlyresume duplicatequeues.

Horizonjob4170931uses the sharedrawWIS scorer with an optionalhorizon filter and verifies task/truth support hashes acrosssystems. Production pinnedcode unchanged.

```bash
export PYTHONPATH=src
.venv/bin/python scripts/score_b6_horizons.py --plan
sbatch scripts/b6_horizons.sbatch
squeue -j4170931
.venv/bin/python scripts/score_b6_horizons.py --rank
```

## 16:45 EDT: all production fits complete; operational build

All110production fits completed. Horizonreport b6-selected-horizons/index.md supports nativeadmissions/ED gains at everyhorizonbothseasons; recentlog near-tie.

All requiredselectedsources refreshed. NWSSaux committed26,991,849rows; its pausedcoordinator91003retired. Outpatient archivechild64512continues with coordinator45025paused. No selectedrecipe usesoutpatient, so separatebuildroot data/operational-b6-pinned-20261007 locks27snapshotrefs, explicitly retainingunusedSep16outpatient. Acquisitionrootrawsymlink restored/preserved forrunningdownloads; pinnedrootseparate. First oldrootindexbuildcancelledandrestartedonpinnedroot. Buildsession19024 logs output/b6/operational-pinned-build.log, runs NWSSindices thenpanelbuild with2workers. Frozenhistoricalpanelunchanged.

```bash
.venv/bin/python -m tapestry.dataset.build nwss-indices --data-root data/operational-b6-pinned-20261007
.venv/bin/python -m tapestry.dataset.build build --data-root data/operational-b6-pinned-20261007 --truth-day 2026-10-07 --workers 2 --output data/operational-b6-pinned-20261007/processed/panel.npz
```

## 17:00 EDT: operational build limited to required inputs

General build spent14minutes rebuildingunused wastewaterindices. Cancelledthatstage and replacedwith samepanelbuilder extracting onlyflu admissions,flu ED,Kinsa,FluSurv. Fullcalendar/schema retained; othercolumns explicitlyNaN, recordedasomittedsources. This is an operationalinputpanel, neverforretraining. Forecastreplay checksrequiredinputscope and rejects omittedsources. No selectedmodelusesothercovariates. Buildsession50899, logoutput/b6/operational-selected-build.log,2workers.

```bash
.venv/bin/python -m tapestry.dataset.build build --data-root data/operational-b6-pinned-20261007 --truth-day 2026-10-07 --workers 2 --sources nhsn_flu_admissions nssp_flu_proportion kinsa_ili flusurv_flu_rate --output data/operational-b6-pinned-20261007/processed/panel.npz
```

Prepared scripts/replay_b6.py and scripts/b6_replay.sbatch onLongleaf; notlaunched. Next: auditbuiltpanel, copyremote, smokeonecheckpoint, then110replays. Managerplan/status inexecution.json.

## 17:15 EDT: operational replay launched

Selected-sourcepanel completed41.8seconds,0.9MB. Audit data/operational-b6-pinned-20261007/input-audit.json: Oct7issuance/Oct3context,fluadmissions52/52finite,ED51/52(IAmissing),US3246admissions/.0058ED. No observedfuturelabels. MissingIAremainsmissing. Frozenhistoricalpanelunchanged. Copiedpaneltoidenticalremotepath. CPUAseed42smokereplay succeeded:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TAPESTRY_TORCH_THREADS=1 .venv/bin/python scripts/forecast_b6.py --checkpoint data/experiments/b6-production-primary-20261007/mlp-pathogen-scheduled_final-91f17ac66760/s42/attempt-001/eval_2026-2027 --dataset data/operational-b6-pinned-20261007/processed/panel.npz --issuance 2026-10-07 --output data/experiments/b6-operational-smoke-20261007 --device cpu
export PYTHONPATH=src
.venv/bin/python scripts/replay_b6.py --plan
sbatch scripts/b6_replay.sbatch
squeue -j4184437
.venv/bin/python scripts/replay_b6.py --status
```

110taskreplayjob4184437uses4workersoneH100,64GB; no fittingandnoaccuracyrankingforunobservedfuture. Finalmodelnamesstillpending.

## 17:30 EDT: both forecast files generated

All 110 operational replays completed. Both forecast files are in output/b6/submission-20261007 with provisional System1/System2 names. Each has 9,568 rows and passes local Python task, date, unit, quantile and Hub-bound checks. One System2 Hawaii ED upper-tail value was capped to the Hub maximum of 0.25; exact original value is recorded in provenance. No median changed. Official R validation was not run. The delivery README records methods, missing Iowa ED input, source scope and remaining naming/metadata work. Nothing submitted.

## 17:45 EDT: metadata and national comparison ready

Both provisional metadata drafts pass the Hub JSON schema including format checks. Validation used an isolated uv environment, without changing project dependencies. The R hubValidations package is absent; documented Python CSV checks passed. National medians and 95% interval graph were generated without visual inspection. Both selected systems forecast growth, with slightly lower System2 medians. Only final names remain for the submission package. Unused outpatient archive continues independently; coordinator45025 remains paused pending completion audit. Nothing published.

## 18:16 EDT: unused archive cleanup complete

Outpatient download child 64512 has exited without a new committed snapshot. The latest snapshot remains September 16, and its log contains neither a completion record nor an error; the exit status is unavailable. This refresh is not counted as successful. After verifying the paused coordinator 45025 identity and absence of children, it was terminated and its exit verified, without resuming the duplicate source queue. No selected recipe uses outpatient data, so the pinned operational panel and completed forecasts remain unchanged. No retry is needed for this delivery. Final model names remain pending.

## 22:45 EDT: FluSurv member removed from System2

At the user's request, `X_B_width256_half_lr_flusurv` was dropped from System2. The remaining eight recipes' existing operational forecasts (ten seeds each, trained on all four seasons) were recombined into a marginal mixture with weight 1/8 per recipe. No model was refit. The output is in output/b6/submission-20261007-noflusurv, and it passes the Python Hub checks. Five members use Kinsa as their only external covariate, and three use none. This eight-recipe combination was not evaluated historically. Nothing was submitted.
