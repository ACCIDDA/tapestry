# B3 plan: 1,000-configuration sweep for forecasting from Wednesday reports

Drafted 5 October 2026. Nothing has been launched. This page records the plan
and the decisions behind it; results will get their own page.

## Goal and protocol

Find the most accurate and most stable forecaster of the six targets (flu, COVID,
RSV hospital admissions and ED proportions, next four weeks), when the model is
given Wednesday reports at each Hub's deadline, as in the
[standard evaluation](../workflow.md#standard-evaluation).

Every run is fitted twice, once per evaluation fold, with seeds 42 and 43:

| Fold | Training seasons (finalized future labels) | Evaluation season (Wednesday-report inputs) |
|---|---|---|
| A, forward | 2022–23, 2023–24, 2024–25 | 2025–26 |
| B, retrospective | 2022–23, 2023–24, 2025–26 | 2024–25 |

**Both folds count equally** (user decision, 5 October 2026: the goal is a stable
best model, not a best forward score). The ranking score is the equal-season mean
of the Hub-relative WIS from `planner rank`: lower is better, and 1 means as good
as the Hub ensemble. Finalists must also be in the top 10% of each fold
separately. The two folds do not score the same targets. Fold A scores all six
targets. Fold B (2024–25) has frozen Hub tasks only for flu and COVID
admissions, because there was no RSV Hub and no ED Hub that season. The six
targets are also scored as raw WIS in both seasons, outside the Hub comparison.

Evaluation uses 512 forecast samples. Where the documented schedule says a value
was available but no report was archived by the deadline, the model receives the
finalized value, and these fills are starred.

## Decisions taken

- **Kinsa has no revisions** (user, 5 October 2026). A finalized Kinsa value
  therefore equals what was published. Assumption: it was published by the
  Wednesday deadline, as the T-0 schedule says.
- **Vintaged data is not settled; it is a main axis of the sweep.** The finding
  below is the reason.
- **Fewer learning-rate and weight-decay levels.** The user will search further
  if needed.
- **Dropped (B2 evidence):** convolutional encoders, attention, pooled and
  national-broadcast spatial sharing, raw covariate histories, and "all
  covariates".

### Why vintaged training is still open

The [overnight comparisons](vintage-overnight-20261005/principal-three-seed-comparisons.md)
were summarized as "no treatment beats finalized training". That was a judgment
on fold A (forward) only. Fold B points the other way. Example: C1 is the
pathogen MLP with distance sharing and no covariates. C1 trained on finalized
histories scored 0.940 (fold A) and 0.935 (fold B). The same C1 retrained with
half-strength artificial early-report errors in all three training seasons scored
0.956 (fold A, 1.7% worse) and 0.850 (fold B, 9.0% better). That gives an
equal-season mean of 0.903 against 0.937. Training on actual archived reports in
the most recent training season helped C3 in both folds: the target MLP with
neighbor sharing and Kinsa improved by 3.2% (fold A) and 12.0% (fold B). These
scores came from the old evaluation path. Fold B scores only flu and COVID
admissions, and the 2025–26 donor errors are better archived than 2024–25's.
With equal fold weights, several of these treatments win on the mean.

## Part A: the sweep

There are 1,000 configurations drawn at random with a fixed seed by
`scripts/plan_b3_sweep.py`. C1, C2, C3 and B0 are added as fixed anchors. Each
axis's effect is estimated by regression over all runs, per fold and in
interaction with the training treatment.

### Training treatments: what changes in the training inputs

The prediction labels are always finalized future values. The table below
describes the code as it is today, followed by the best version we will run.

| Treatment | Share | What the code does today | What it is missing / best version |
|---|---|---|---|
| T0 Finalized histories | 20% | Finalized T-0 histories; optional artificial masking and known-final flag | Reference; nothing to improve |
| T1 Artificial early-report errors (`reporting_augmentation=vintage`) | 25% | Reporting errors (report vs final, log ratio by report age) come from the latest training season only. Errors are matched per location on recent level/growth and calendar, using 8 nearest donors, and applied to every signal in all three training seasons. | Levels: seasons changed {all three, early two only with the recent season's actual reports}; signals {all, admissions only}; strength {0.5, 1}; share of examples changed {0.5, 1}; method {local, one donor date for all locations (`synchronous_log`)}. Port `vintage_scope` / `vintage_signals` from the private overnight code ([patch](vintage-overnight-20261005/implementation.patch)). Check: add 2023–24 flu-admission archives as donors (32 of 44 Wednesdays archived). |
| T2 Training inputs corrected by the nowcaster (`reporting_augmentation=nowcast`) | 10% | Same donor errors, but measured after the seasonal chain-ladder nowcaster (`adaptive_chain`) corrects the reports. At evaluation, the newest 8 target weeks are *fully replaced* by its nowcast, ED included. | Use the overnight-selected tree nowcaster (not in main; [patch](nowcast-overnight-20261005/nowcast-mechanisms-complete.patch)). Correct admissions only and leave ED raw, as in the winning mixture. Strength {0.5, 1}. Overnight result: retraining behind a correction was worse than giving the same correction to an unchanged model, so T2 must beat T0 combined with the mixture evaluation (Part C), not just T0. |
| T3 Joint reconstruct-and-forecast (`weekend_family=joint`) | 15% | Training inputs get T1 errors (missingness transfer off). The network predicts the recent weeks (finalized) and the next four weeks. The two losses are normalized separately and weighted by `joint_weight`, a wiring that was fixed overnight. | Levels: `joint_weight` {0.5, 1, 2}; reconstructed weeks {2, 4}; T1 strength {0.5, 1}. Check that the overnight wiring fix is the version in main (`training.objective_weights`). |
| T4 Corrector, then forecaster (`weekend_family=two_stage`) | 10% | A ridge revision regression, cross-fitted by season on synthetic reports, corrects the inputs. The forecaster trains on corrected synthetic reports and is evaluated on corrected Wednesday reports. | Levels: `correction_strength` {0.5, 1}; `correction_features` {phase, phase_local}. Check: replace the ridge corrector with the selected tree nowcaster trained on real archive pairs. `task=pipeline` (neural nowcaster) is not scored by the standard evaluation and stays out. |
| T5 Actual archived reports as training inputs | 20% | `input_mode=vintaged` takes the newest `asof_weeks` weeks from the archive. **Where nothing was archived, the cell is marked unavailable**, which covers all of 2022–23 and most of 2023–24. Evaluation fills those cells with finalized values instead, so training and evaluation disagree. | Add `input_mode=reported` for training, using the evaluation's own rule: the archived report where one exists, otherwise the finalized value. This is the treatment that helped C3 in both folds. Levels: input rule {reported, vintaged with `asof_weeks` 2, vintaged with `asof_weeks` 12} × early seasons {unchanged, T1 errors at strength 0.5}. |

Constraints the sampler enforces: T1–T4 require `mask_rate=0` and
`supplied_final=0`, and T3/T4 also require `reporting_missingness=0`.

### Pilot before the full launch (method tuning)

The aim is to run each treatment at its best before it enters the sweep. Each
treatment's best-known settings are run on C1 and C3, seeds 42 and 43, both
folds, under the new evaluation (about 50 runs). This pilot:

1. Reproduces the overnight direction for T1/T5 under the new evaluation path.
2. Measures run time per treatment, to size the sweep.
3. Drops any level that fails in both folds and both architectures before the
   sweep spends budget on it.

### Evaluation-input axis: same trained model, different evaluation inputs (no retraining)

Every finished run is scored three ways:

- reported histories only;
- a 50/50 mixture of forecast samples from reported and nowcast-corrected
  admission histories, with ED left raw;
- forecasts from corrected admission histories only.

This needs the selected nowcaster and the mixture ported into the standard
evaluation path (`evaluate_hubs`). The overnight version used `replay_from`,
which the standard evaluation does not score. Each Hub's deadline must be used
separately when computing these corrections.

### Architecture and training-recipe axes

| Axis | Levels |
|---|---|
| `fit_partition` × `head_sharing` | {pathogen, target, all} × {shared, pathogen, target} |
| `lookback` | 8, 10, 12, 16 |
| `width` | 32, 64, 128 |
| `latent`, `noise`, `us_error` | {8, 16, 32}, {global, local}, {none, shared_factor} |
| `heads`, `decoder` | {shared, state_us}, {legacy, residual2} |
| `count_transform`, `ed_transform` | {fourth_root, sqrt, log1p}, {logit, fourth_root} |
| `spatial` | none, distance, neighbors |
| `covariate_set` (summary encoder) | none, kinsa, ilinet |
| `signal_features`, `dynamics`, `location_embedding` | {none, multiscale}, {on, off}, {0, 4} |
| `mask_rate`, `supplied_final` | {0, 0.2}, {0, 1} (T0 only) |
| `lr`, `weight_decay`, `batch_size` | {5e-4, 1e-3}, {0, 1e-4}, {8, 16} |

Fixed: MLP encoder, `loss_weights=objective`, `input_normalization=b0`, 300 epochs
with patience 30, and the existing validation calendar.

## Part B: new architectures (second experiment, same folds and seeds)

Each is run on the 3–5 best sweep backbones, each with its best training treatment.

1. `inputs=own|pathogen|all`: limits which target histories the network reads.
   `own` with `fit_partition=target` is a true FluSight-only model: flu admissions
   in, flu admissions out. The current C1 flu network reads all six histories.
2. A direct 23-quantile head with no quantile crossing, trained with pinball loss.
   Each quantile function is expanded to 512 evenly spaced samples, so the scorer
   is unchanged.
3. A damped-trend anchor: predict departures from a smoothed, flattening recent trend.
4. Non-neural comparators: a Flusion-style pooled gradient-boosted quantile model
   and pooled quantile regression.

## Part C: post-processing, no retraining

Pool the two seeds into one forecast. Apply the reported/corrected mixture.
Calibrate bias and spread using out-of-fold training-season predictions only.

## Selection and its limits

Rank by the equal-fold mean, then require the top-10%-per-fold rule. Confirm the
top 3–5 in a new experiment with seeds 42–46. Two seeds do not separate
configurations within about 0.01. Both seasons have now shaped many decisions,
so the winner's margin is expected to shrink on 2026–27.

## Order of work

1. Port the T1 options and the tree nowcaster from the private overnight code.
   Add `input_mode=reported` for training and put the mixture evaluation into
   `evaluate_hubs`. Check that every treatment goes through the standard
   evaluation.
2. Pilot (about 50 runs), then size the sweep.
3. `scripts/plan_b3_sweep.py`, then launch Part A. Code Part B while it runs.
