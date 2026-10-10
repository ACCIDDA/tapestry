# Model choices A–F

Every Chromantis forecaster since the B3 pilot (5 October 2026) is defined by six separate
choices (A has three parts), plus its prediction labels and its folds. Several words were used for more than
one of these choices, so this page fixes the vocabulary. Every experiment report links
its own table of choices to the sections below. Field names are the scenario fields
([field key](scenario.md)); the names in parentheses are the names used before 8–9 October
2026. They are no longer read: the B7 and System2 records were rewritten to the current names
on 9 October 2026, and older experiments are kept as reports only.

## Words with several meanings {#terms}

| Word | Meanings in this project |
|---|---|
| **nowcaster** | (a) The **correction model**: small gradient-boosted trees (or an MLP) that adjust the newest reported weeks of admissions, and of ED if requested, towards their likely final values ([choice D](#d-correction-model)). This is the only nowcaster in the code. (b) A separate neural nowcast network (`task=nowcast`, `task=pipeline`), deleted 8 October 2026. (c) Per-signal finalizers (`task=finalize`), deleted 8 October 2026. |
| **reconstruction** | A `joint` network also predicts the final values of its last context weeks, as extra training labels ([choice A](#a-training-histories)). A nowcast inside the forecaster, separate from the correction model. |
| **vintaged / vintage** | (a) **Real vintages**: values as archived on a Wednesday. (b) **Artificial vintages**: final values made to look preliminary with sampled reporting errors ([B](#b-error-source), [C](#c-error-signals)). (c) The deleted `input_mode=vintaged`. This page says *archived reports* and *artificial errors* instead. |
| **corrected** | (a) A **training treatment**: artificial histories corrected by the correction model before the network learns from them ([A](#a-training-histories)). (b) An **input view** at forecast time: inputs corrected by the correction model ([F](#f-input-view)). |
| **pilot** | The B3 pilot experiment that introduced this training route; not a model property. |

## A. Training histories: what the network learns from {#a-training-histories}

Three separate fields (until 9 October 2026 bundled into one, `pilot_method`, briefly
`training_histories`; the translation of old values is in the second table):

| Field | Values | Meaning |
|---|---|---|
| `history_source` (A1) | `finalized`, `reported`, `artificial` | Training inputs: final reference histories with the documented source lags; the actual archived Wednesday reports (finalized values where nothing was archived); or final histories made preliminary with artificial reporting errors ([B](#b-error-source), [C](#c-error-signals)) |
| `history_correction` (A2) | `0`, `1` | Correct the artificial training histories with the correction model ([D](#d-correction-model)) before the network learns from them. Cross-fitted: four blocks of origins, each corrected by a model that never saw it. Only with `artificial`. |
| `reconstruction_labels` (A3) | `0`, `1` | Also learn the final values of the last `reconstruction_weeks` context weeks (weight `joint_weight`). Labels, not inputs. Implemented with uncorrected artificial histories only. |

| Old value (`pilot_method`) | Equals |
|---|---|
| `finalized` | `history_source=finalized` |
| `reported` | `history_source=reported` |
| `errors` | `history_source=artificial` |
| `corrected`, `two_stage` | `history_source=artificial`, `history_correction=1` |
| `joint` | `history_source=artificial`, `reconstruction_labels=1` |

Whatever A is, every fold also fits and saves a correction model ([D](#d-correction-model)),
used at forecast time when the forecast view is `corrected` or `half` ([F](#f-input-view)).
Code: `experiment/fit.py` `fit`.

## B. Error source: where the artificial reporting errors come from {#b-error-source}

An empirical bootstrap of real reporting errors: for each training history, an error
window from a real archived release with a similar epidemic phase is transported
(age, signal and location aligned; labels never change). Code: `dataset/reporting_error.py`.

| Setting | Donor releases |
|---|---|
| default (`error_seasons=latest`, no `error_reference`) | the latest permitted training season of each fold. When 2024–25 is held out, that is 2025–26, a later season. |
| `error_seasons=all` (was `vintage_seasons`) | every permitted season, weights 1, ½, ¼ from the latest |
| `error_reference=2025-2026` (was `revision_reference`) | the prescribed 2025–26 reporting process in every fold, including the fold that evaluates 2025–26 |

It matters for `errors`, `corrected` and `joint` training histories, for correction models
trained on synthetic examples ([D](#d-correction-model)), and for prescribed evaluation
inputs ([E](#e-evaluation-inputs)). Details of the matching: `reporting_*` fields.

## C. Error signals: which signals receive artificial errors {#c-error-signals}

`error_signals` (was `revision_signals`): `admissions` (counts only; ED and covariates
keep their final values), `targets` (admissions and ED), or `all` (also covariates).
`error_scope` and `actual_share` can replace some artificial draws by actual reports.

## D. Correction model: what the nowcaster learns from {#d-correction-model}

`corrector_examples` + `corrector_model` (both were `pilot_nowcaster`):

| Setting | The correction model learns from |
|---|---|
| `corrector_examples=synthetic` (was `synthetic_*`) | artificial preliminary histories ([B](#b-error-source), [C](#c-error-signals)) paired with their final values |
| `corrector_examples=real` (was `real_*`) | real archived reports paired with their values twelve weeks later, from the donor season(s) |
| `corrector_examples=synthetic_then_real` (was `pretrained_mlp`) | synthetic pretraining, then real pairs (MLP only) |
| `corrector_model=tree` / `mlp` | gradient-boosted trees (default) / a residual MLP |

It corrects the newest `correction_weeks` weeks of admissions, and of ED when
`correction_ed=1`. `correction_noise` adds sampled correction errors (the `sampled` view).
Code: `model/revision_tree.py`; saved per fold as `nowcaster.pkl`.

## E. Evaluation inputs: what the fitted model is scored on {#e-evaluation-inputs}

The problem's `evaluation.inputs` value (formerly the recipe field
`evaluation_inputs`, and before that `evaluation_vintaging`):

| Value | Inputs of every held-out forecast |
|---|---|
| `reported` | **Real archived Wednesday reports** at each Hub's deadline. Where the documented schedule says a value is available but nothing was archived, the finalized value is used and counted (★ footnote). For FluSight October–May issuances this happens for the newest ED week in every 2023–24 and 2024–25 issuance (ED assumed at T-0), 15–23% of the newest admission weeks, and nearly all Kinsa (its archive starts April 2026). |
| `prescribed` | **Artificial histories**: final values made preliminary with the recipe's `error_reference` process, every season, even where real archives exist; the problem's `evaluation.draws` sets the number of independent draws shared by every configuration and seed. |

`planner replay -e NAME --inputs reported` scores already-fitted folds on real reports
without refitting. Production always uses the real operational reports of the issuance.

## F. Forecast view: how the fitted pair is used at forecast time {#f-input-view}

`forecast_view` (a recipe field since 9 October 2026; default `corrected`, what both
submitted models use):

| View | Inputs given to the network |
|---|---|
| `raw` | the evaluation (or operational) inputs as they are |
| `corrected` | the newest weeks corrected by the fold's correction model ([D](#d-correction-model)) |
| `half` | not an input: the 50/50 mixture of the raw and corrected forecast distributions (research only, not in production) |

A configuration is **ranked and deployed in its own `forecast_view`**. Every fit is also
scored in the other views and in the optional `calibrated` (corrected, spread recalibrated
on validation weeks), `sampled` (corrected with sampled correction errors), `delayed`
(newest admission week withheld) and `nokinsa` (covariates withheld) views, as
diagnostics without a rank. Until 9 October 2026 reports showed each configuration at its
best view after scoring (B7: corrected for all), an extra selection on the evaluation
data; a release must now use the view of its recipes (`chromantis.production` checks it).

## Labels and folds {#labels-and-folds}

**Labels and forecast horizons** are the targets and horizons in the required problem file;
`reconstruction_labels=1` adds the recent reconstruction weeks as an auxiliary objective.
**Folds** and their permitted training seasons also come from that problem. The current
respiratory research problems use leave-one-season-out folds for 2023–24 through 2025–26,
with training seasons among 2022–23 through 2025–26 (`training_window` can shorten this).
The production problem fits all four completed seasons and emits the 2026–27 fold.

## Pathways in the code {#pathways}

| Use | Choices | Command |
|---|---|---|
| Fit and evaluate | A–F; ranked in the recipe's F view, other views as diagnostics | `planner plan/run`, `scripts/jlessler.sbatch` |
| Re-evaluate fitted folds on other inputs, no refit | E switched to `reported` (or `prescribed`) | `planner replay` |
| Combine saved forecasts, no refit | the members' A–F | `python -m chromantis.evaluation.ensembles` |
| Production | release recipes; real operational reports; the recipes' forecast view | `python -m chromantis.production run` |

## What each experiment used {#experiments}

From each experiment's saved scenario strings. Before the B3 pilot, routes now deleted were
used (B0 finalized inputs; B1–B2 dated inputs with finality flags; B2 reporting augmentation
with a linear correction); those are not described by A–F.

| Experiment | A | B | C | D | E | Folds |
|---|---|---|---|---|---|---|
| B3 pilot | all five compared | fold's latest | admissions | mostly synthetic trees | reported | 2024–25, 2025–26 |
| B4 (600, heads, refine, refineTop2, polish) | mostly `corrected` (sampled "A") and `joint` (quantile "B"); `errors`/`reported`/`finalized` as controls | fold's latest | admissions | synthetic and real trees both compared for `corrected`; synthetic for `joint` | reported | same |
| B5 (explore, covariates, confirm) | `corrected`, `joint`; some `actual_share` | latest or all | admissions | real for `corrected`, mostly synthetic for `joint` | reported | same |
| B6 (search, extension, matched, specialists) | `corrected`, `joint` | fold's latest | admissions; `targets` for the `X_*` and extension variants | real for `corrected`, mixed for `joint` | reported | same |
| B7 fast | `corrected` (A_blocks3, B5 ILI), `joint` (B_width256) | prescribed 2025–26 | targets | synthetic trees (one MLP variant), admissions and ED | prescribed, 3 draws | 2023–24, 2024–25, 2025–26 |
| B7 vs System2 | saved folds of both, no refit | each system's own | own | own | reported (replayed) | 2024–25, 2025–26 |
| **System2** (submitted on time) | 5 `corrected`, 3 `joint` | latest (2025–26 in production) | admissions; targets for 2 recipes | real trees for `corrected`, mostly synthetic for `joint` | production: real reports, `corrected` view | trained on all four seasons |
| **B7** (merged submission) | 2 `corrected`, 1 `joint` | prescribed 2025–26 | targets | synthetic trees, admissions and ED | production: real reports, `corrected` view | trained on all four seasons |

Two consequences:
- B_width256 (`joint`) is trained on uncorrected histories but forecast with corrected inputs; it scored best that way in B7.
- System2 and B7 differ in B, C and D at once, not only in their recipes.

History: written 8 October 2026 at the user's request ("these terms mean many things"), with
the renames listed in `model/scenario.py` (`RENAMED`). On 9 October 2026, after review, the
training treatment was split into A1–A3 (`TREATMENTS`), the forecast view became a recipe field,
and every report's protocol table describes each configuration, with its training seasons per
fold, evaluated seasons and labels.
