# Expanded B2 comparison: reporting revisions, separate nowcasting, and joint learning

[Read the completed weekend analysis and graphs (5 October)](results-20261005.md).

> **5 October audit:** The initial training loop did not call the separately
> normalized joint-loss helper. For the weekend runs, both `joint_weight=0.25`
> and `joint_weight=1` actually used the same pooled loss across all eight
> horizons. The advertised weight comparison is invalid. Forecast scores from
> those models can still describe joint learning with that pooled loss, but
> duplicate weight settings must not be treated as distinct interventions.
> The overnight nowcasting agent is repairing this in new, separately named runs.
>
> **Weekend status:** 1,730 of 4,320 seed runs completed before the remaining
> GPU allocations were stopped on 5 October to free both patron nodes for the
> user's requested focused overnight experiments. Each completed seed run has
> both evaluation seasons. Partial results require explicit seed completeness.

## Question and size

Can a B2-like forecaster recover forecasting accuracy when its evaluation inputs
contain reporting revisions? This experiment retrains models; it does not merely
supply different inputs to existing B2 checkpoints.

The design has **1,440 configurations × 3 seeds (42, 43, 44) × 2 evaluation seasons
= 8,640 outer model fits and evaluations**. This is 4.66 times B2's 309
configurations. There are additional inner fits for early stopping. The separate
nowcasters are inexpensive regression fits, not additional B2 neural networks.

We retain 48 original B2 architectures after removing the finality indicator and
random input masking. The first 16 are the highest-ranked distinct architectures;
the other 32 greedily cover differences in encoder, spatial sharing, independently
fitted target groups, covariates, covariate encoding, signal features, coordinates,
lookback, width, and decoder. Original B2 rank breaks ties. Every architecture is
crossed with the same 30 treatments. This is an architecture-by-treatment design,
not a fresh exhaustive search of all original B2 settings.

## Exact seasonal splits

| Evaluation inputs and season | Finalized trajectories used to train the forecaster and any nowcaster | Season supplying empirical reporting errors |
|---|---|---|
| Wednesday reports in **2024–25** on the assumed B2 availability schedule | **2022–23, 2023–24, 2025–26** | **2025–26** |
| Wednesday reports in **2025–26** on the assumed B2 availability schedule | **2022–23, 2023–24, 2024–25** | **2024–25** |

The 2024–25 evaluation is retrospective. Both seasons have already been used for
model development, and architecture selection uses the previous B2 ranking.
Neither is an untouched prospective test. Reporting errors from the evaluation
season are excluded from training. Inner early-stopping weeks are also excluded
from the inner error library and preprocessing fit.

## Availability and reporting delays

“Full availability” means the B2 documented source schedule: six targets through
the latest event week; source-specific one-week delays for ILINet, clinical labs,
and FluSurv; the existing Vermont inpatient exception; genuine missing final
historical values remain unavailable. No additional reporting-gap masks are
copied and no random masking regularizer is applied. The finality indicator is
omitted from every network.

At evaluation, use actual values archived by the Wednesday issuance, including
revisions for the entire context history, for targets **and covariates**. Preserve
the B2 availability mask and source delays. Where an archived value is absent but
the B2 schedule assumes it exists, use its frozen finalized value as an explicit
numerical proxy. This isolates revisions under an assumed availability schedule;
it is **not** an operational evaluation of unpublished observations. Evaluation
proxy counts are saved in each fold manifest. No proxy enters the empirical
training-error library. Rows with missing finalized prediction labels remain
unscored; “full availability” does not invent historical observations or labels.

## What changes in training

All forecasters learn finalized values at **t+1, t+2, t+3, t+4**. Joint models also
learn finalized values at **t−3, t−2, t−1, t**. Neither reporting errors nor nowcast
corrections are applied to these prediction labels.

Empirical errors compare real archived reports with finalized values from the
permitted donor season. Error windows preserve report age and signal alignment.
The default draws an entire signal-by-age window from the same location, matched
to local level, growth, and seasonal position. It transports signed log ratios
with a floor of 5% of the training-season location/signal peak (at least one count
or 0.0001 proportion). Donor missing cells contribute zero perturbation. National
covariates stay shared across locations. A same-origin donor is excluded.

### 12 direct-forecast training treatments (576 configurations)

All 12 are evaluated on the same raw Wednesday report values and assumed B2
availability schedule. They differ only in training-input errors:

1. Finalized historical training inputs without added error.
2. Local seasonal log errors at one-quarter strength.
3. Local seasonal log errors at one-half strength.
4. Local seasonal log errors at full strength.
5. Half the episodes unchanged, half with half-strength local errors.
6. Half the episodes unchanged, half with full-strength local errors.
7. Calendar-matched errors at half strength.
8. Calendar-matched errors at full strength.
9. Synchronized national donor windows at half strength.
10. Synchronized national donor windows at full strength.
11. Local phase matching without calendar matching, full strength.
12. Local errors with a freshly drawn episode-level strength uniform from 0 to 1.

Half strength multiplies the **log error** by 0.5. It does not halve the model's
training duration or change evaluation reports. Synchronized windows choose one
donor date for all locations, preserving concurrent cross-location reporting
patterns; local methods can choose different donor dates by location.

### 10 separately trained nowcaster–forecaster treatments (480 configurations)

First fit a ridge regression to reconstruct the latest four finalized weeks from
synthetic provisional histories. Each of the six targets has its own regression.
Predictors include report age and, for phase models, recent observed level, recent
growth, acceleration, coverage, annual timing, and proximity to Christmas. Optional
location indicators shrink toward the shared correction. Prediction features use
only the supplied history, never future outcomes or the finalized epidemic peak.
The fit uses two independent artificial revisions per permitted training episode.
It balances trajectory seasons within each target. Corrections are capped at a
factor of four on the stabilized scale and ED values remain between zero and one.

The ten settings are: phase features with ridge penalties 1, 10, or 100; phase
features with penalty 10 and correction strengths 0.25 or 0.5; age-only with
penalty 10; phase plus location features with penalties 1, 10, or 100; and phase
features with penalty 10, half correction, and synchronized donor errors.

The forecaster is retrained on freshly simulated histories corrected by a
regressor trained on the **other trajectory seasons**. Its prediction labels
remain finalized future values. The empirical donor-error library is still shared
within the permitted fold: this cross-fitting excludes the trajectory season
from regression labels, not from the training fold's augmentation distribution.
It therefore does not establish an independent estimate of nowcaster residuals.
For early stopping, both the library and all regression fitting exclude hidden
validation weeks. For held-out evaluation, refit the nowcaster on all three
permitted training seasons and apply it to actual Wednesday reports before
forecasting. It does not recalibrate using held-out-season outcomes.

This extends the previous correction work by explicitly learning acceleration,
comparing local shrinkage and correction strength, and retraining the forecaster
on the resulting corrected input distribution. It does not reuse the old online
seasonal nowcaster, its cached predictions, or its residual library. Nowcast
uncertainty is not separately sampled in this family; future uncertainty comes
from the B2 forecaster.

### 8 joint reconstruction-and-forecast treatments (384 configurations)

Train one B2 neural model with eight output horizons: four reconstructed recent
weeks and four future weeks. Shared representations learn both tasks together;
there is no separately fitted nowcaster or explicit correction-then-forecast
stage. For architectures with independently fitted pathogen or target components,
each component jointly learns its own reconstruction and forecast outputs.

Cross local full-strength, local half-strength, synchronized full-strength, and
uniform-random-strength errors with reconstruction loss weights 0.25 and 1.
Normalize forecast and reconstruction weights separately: the forecast loss has
weight one, and reconstruction adds the specified weight. This preserves the
forecast target/season/geography objective rather than silently diluting it.
That was the intended objective; the 5 October audit above found that the original
fitter instead used one pooled normalization across all eight horizons. The two
nominal reconstruction-weight settings were therefore not distinct treatments.
Forecast ranking uses only t+1 to
t+4. Recent-week outputs and reconstruction scores are stored separately.

## Training, scoring, and interpretation

Keep B2's 300-epoch cap and patience 30, selection followed by a full refit,
training members and optimizer settings. Use 256 forecast members for evaluation.
All three seeds use the same data and held-out support. The panel, population
file, frozen comparison support, and source snapshot are pinned by the common
planner. No earlier experiment's code snapshot is changed.

Use the existing Hub-relative weighted interval score (WIS): lower is better,
1 equals the matched Hub ensemble. State/DC and US weights are 80% and 20%; target
weights are 1 for admissions and 0.5 for ED proportions; the overall comparison
gives each season equal weight. The 2024–25 frozen Hub support covers fewer targets
than 2025–26. Report seasons, targets, and geographies separately, as well as
paired seed differences from the same architecture trained on unchanged finalized
inputs and evaluated on raw reports. A two-stage comparison changes training and
evaluation processing together; it is not solely a training augmentation effect.

Scores selected from 1,440 configurations are subject to selection optimism. Three
seeds measure training variation, not uncertainty about future seasons. Error
transport assumes local standardized revisions from the donor season remain
informative in another season. Sparse or absent donor revisions stay unperturbed,
which can make artificial reports easier than real reports. Proxy-filled
held-out inputs can also make the evaluation optimistic.

## Launch status

Launched 2 October 2026. All six allocations were confirmed running: L40 array
**3477740**, H100 array **3477741**. Automatic report job **3477798** waits for
both arrays. Nine short seed runs (18 scored folds) completed before launch.
Nine focused scientific checks passed. Full-budget research results are pending.

## Running and resuming on Longleaf

Working directory: `/proj/jlessler/projects/tapestry-all/tapestry-weekend-20261002`.
The planner writes the exact 1,440-row design to `data/experiments/b2-weekend-20261002/design.csv`.

```bash
cd /proj/jlessler/projects/tapestry-all/tapestry-weekend-20261002
export PYTHONPATH="$PWD/src"

# Plan once; do not rerun against an existing experiment.
.venv/bin/python scripts/plan_b2_weekend.py -e b2-weekend-20261002

# Four L40s, four concurrent fits per GPU; two H100s, six per GPU.
LANES=4 GPUS=6 sbatch --job-name=b2-weekend-20261002 --array=0-3 --nodelist=g1803jles01 --cpus-per-task=8 --mem=100G scripts/jlessler.sbatch b2-weekend-20261002 --retry-failed
LANES=6 GPUS=6 sbatch --job-name=b2-weekend-20261002 --array=0-1 --nodelist=g1803jles02 --cpus-per-task=12 --mem=160G scripts/jlessler.sbatch b2-weekend-20261002 --retry-failed

.venv/bin/python -m chromantis.experiment.planner status -e b2-weekend-20261002
.venv/bin/python -m chromantis.experiment.planner rank -e b2-weekend-20261002 --allow-incomplete --no-plots
.venv/bin/python scripts/report_b2_weekend.py -e b2-weekend-20261002
```

`status` prints the resubmission command for unfinished work. Both arrays draw
from one shared queue. Notifications are enabled by the standard launcher.
Allocation time limits are 48 hours. The report job ranks the completed subset
and produces season-specific plots after the arrays finish, including on timeout.
Its tables retain architecture, training treatment, evaluation season, and seed.
The report was queued with:

```bash
sbatch --dependency=afterany:3477740:3477741 scripts/b2_weekend_report.sbatch b2-weekend-20261002
```

## Research log — 2 October 2026

The previous input-error runs mixed training interventions with availability
mismatches; removing copied gaps alone did not reliably improve forward forecasts.
This design fixes availability across all treatments, removes random masking and
the finality indicator, broadens the donor dependence/strength assumptions, and
compares direct, separate, and joint learning on the same architectures. The
separate regression deliberately uses simulated histories from the three permitted
seasons, allowing the requested donor-season split without relying on the older
online nowcaster's masked archive.

Scientific checks cover unchanged prediction labels and availability, exclusion
of held-out error seasons, separate joint-loss normalization, and prediction
features independent of finalized future labels. A short GPU run exercises all
three model families before the full launch. Short-run scores are implementation
checks, not research results. Cluster job IDs and observed launch state are saved
in `launch.json` beside this protocol.

The initial 56-fit concurrency exceeded GPU memory during full-size validation.
Those arrays were stopped and the same pinned design restarted with 28 concurrent
fits (four per L40, six per H100), retrying failures. All 43 recorded failures in
that first allocation were CUDA memory errors. No model settings or score rules
were changed to address the resource issue.
