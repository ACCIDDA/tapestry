# Proposed 1,000-configuration vintage-input forecasting experiment

> Superseded by the [B3 exploratory pilot](../b3-pilot-20261005/index.md): all six targets retained, both seasons equally weighted, native six-target ranking plus flu native/log reports, historical ILI and nowcaster development included. The fixed 1,000-row proposal below is not the launch design.

5 October 2026. **Design only; no experiment planned and no jobs launched.**
The user selected FluSight admission log-scale WIS as the primary objective,
two seeds per configuration, and preparation of the design before implementation.

The proposal is a small forecasting network trained on finalized histories,
with direct ordered quantiles and an optional separate admission nowcaster.
This is a hypothesis to test, not an established winner. The sweep retains
the existing sampled-output MLPs as substantial, matched competition.

## What the evidence supports

The original B2 comparison evaluated finalized inputs; its selected scores are
not benchmarks for the new reported-input evaluation. In the subsequent work,
the pathogen-specific MLP with distance sharing and no covariates, trained on
unchanged finalized 2022–23/2023–24/2024–25 histories and finalized next-four-week
labels, remained a strong model when evaluated on 2025–26 Wednesday reports
with finalized fills. Giving that saved model an equal mixture of raw admission
histories and nonlinear-nowcaster histories reduced six-target native-scale
ensemble-relative WIS from 0.928529 to 0.918262 in matched H100/2,048-draw
evaluation. Lower is better. This changes evaluation inputs, not forecaster
training. It does not establish a gain on flu-only log WIS or the new evaluation.

Broad artificial revisions and retraining behind a nowcaster did not consistently
improve that model. MLPs, short histories, compact source summaries and limited
spatial sharing deserve most of the budget. New architectures and reporting
treatments still need direct matched comparisons. Sources:
[completed post-B2 comparisons](../overnight-b2-technical-results-20261005.md),
[original B2](../b-2-t0/index.md), and [current evaluation](../../workflow.md#standard-evaluation).

## Is the existing model flu-only?

No. The pathogen-partition bundle fits three independent networks. Its flu
network learns flu admission and flu ED labels, while receiving all six target
histories. The target-partition bundle fits six networks; its flu-admission
network learns only admission labels but still receives all six histories.
Neither makes a clean flu-input-only comparison.

The existing `loss_weights=flu_only` sets positive weight only on flu admissions;
it does not remove other inputs or provide a complete flu-only execution path
through every partition and scorer. The proposal explicitly separates output
labels, input channels and parameter sharing. Flu-only models must not be ranked
using their untrained COVID/RSV outputs.

## Fixed evaluation and fitting protocol

| Role | Forecaster training seasons | Prediction labels | Evaluation inputs and season |
|---|---|---|---|
| Primary forward development | 2022–23, 2023–24, 2024–25 | Unchanged finalized t+1 through t+4 values for the specified target set | 2025–26 reports at each forecast pathogen's own Hub deadline |
| Secondary retrospective | 2022–23, 2023–24, 2025–26 | Same label definition | 2024–25 reports at each forecast pathogen's own Hub deadline |

Except for the explicitly named reporting treatments, training inputs are
unchanged finalized histories under the source schedule. Every row uses seeds
42 and 43. Refitting on the full permitted training seasons follows blocked
training-season epoch selection. Validation outcomes must be excluded from
fitted scales, auxiliary training and nowcaster fitting as well as from the
forecaster's labels. All transformed losses are computed from unchanged
finalized labels; the transformation changes the objective, not truth.

Use the rebuilt, per-Hub-deadline panel and the new shared evaluator. All past
weeks in each candidate's 8-, 10- or 12-week context use reported evaluation
inputs. Targets are available through T-0 under the chosen schedule; ILINet,
clinical laboratories and FluSurv through T-1. Missing numerical archives receive
finalized fills where the standard protocol permits them. Naturally absent
truth stays missing. Report fill fractions for every candidate's actual input
scope. This is the chosen retrospective availability regime, not a faithful
reconstruction of all historically unpublished observations.

**Primary ranking assumption:** average WIS after applying `log(1 + count)` to
the 23 quantiles and truth, on identical frozen FluSight admission tasks in
2025–26, states/DC only. Give each task equal weight; average the two seed
scores. Use frozen Hub truth on this support. Lower is better. This is a stable
common-task objective; it does not change with the set of sweep candidates.
The log offset, equal-task aggregation, state/DC focus and choice of the forward
season are explicit proposed choices following the user's log-WIS priority.

Also report the existing log ensemble-relative and official-style pairwise
FluSight tables, full-window raw log WIS against panel truth, native-unit flu
WIS, coverage, bias, horizon results and national forecasts. Preserve the six-target
reports for models trained on all six labels. The current manager still sorts
by native six-target WIS; a primary-score selector is required before this launch.
Do not average the retrospective season into the primary selection score.

Sampled models use the standard 512 evaluation draws. Direct quantile models
provide their exact 23 quantiles, without sampling them and estimating them again.
They do not claim coherent trajectory samples. Their direct-quantile export
requires integration with the shared scorer; it is not implemented by this design.
Both seasons have already informed model development; neither is an untouched test.

## Exactly 1,000 proposed configurations

| Group | Count | Fixed construction |
|---|---:|---|
| Existing sampled MLPs | 360 | 3 histories × 3 spatial choices × 2 fitting partitions × 2 masking rates × 10 optimizer/width recipes |
| Direct-quantile architectures | 240 | 5 architectures × 2 histories × 2 widths × 2 spatial choices × 2 learning rates × 3 covariate choices |
| Flu-only labels and input sharing | 160 | 20 fixed architecture contexts × 2 label sets × 4 input sets |
| Reporting histories and nowcasters | 120 | The same 20 contexts × 6 reporting treatments |
| Training-loss alignment | 80 | The same 20 contexts × 4 alternative loss definitions |
| Pooled boosted quantile trees | 40 | 2 histories × 2 input sets × 2 depths × 5 minimum-leaf sizes |
| **Total** | **1,000** | **2,000 configuration/seed evaluations; 4,000 seasonal evaluations** |

These are forecast-pipeline configurations, not 1,000 unrelated neural training
jobs. Eighty reporting configurations reuse their matched forecaster checkpoints
and fit or reuse a separate corrector. Do not refit identical forecasters merely
to inflate the run count. Correction-only comparisons use the same checkpoints
and forecast randomness. Report both pipeline count and unique fitted models.

The [complete configuration list](configurations.csv) fixes every row in advance.
[Design metadata](design.json) records counts and folds.
[Generator](build_design.py) regenerates these proposal artifacts and checks
uniqueness. Its fields describe proposed behavior: **they are not accepted
`Scenario` strings**, and generating this file does not call the planner.

### Existing MLPs: 360

Cross histories 8/10/12; spatial none/distance/neighbors; pathogen/target fits;
artificial masking probability 0/0.2; and these ten `(width, learning rate,
weight decay)` recipes:

```text
(32, 0.0003, 0)    (32, 0.001, 0)
(64, 0.0003, 0)    (64, 0.001, 0)
(128, 0.001, 0)    (64, 0.003, 0)
(64, 0.001, 0.0001) (64, 0.001, 0.001)
(32, 0.001, 0.001) (128, 0.001, 0.001)
```

Use no extra covariates, all six inputs and outputs, fourth-root admission
inputs, logit ED inputs, the recent-value anchor and native-Q95-normalized CRPS.
Hold the remaining sampled-model settings to the post-B2 reference: width as
specified, latent 16, global noise, legacy decoder, shared state/US output heads,
calendar/dynamics/population features on, no location embedding or finality
channel, batch size 8, 128 training draws, 300-epoch cap, patience 30 and the
existing blocked validation calendar. Snapshot their explicit resolved values
when translating the proposal to scenarios. Mask probability concerns episodes,
not the fraction of cells, and retains the existing pattern mixture.

### Direct quantiles and new architectures: 240

Use histories 8/12, widths 32/64, spatial none/distance, learning rates
0.0003/0.001, and covariates none/Kinsa/ILINet. Covariates use compact summaries.
All learn finalized future labels for all six targets, with no artificial
masking and the native-Q95-normalized quantile loss initially. The architectures:

1. Existing independent pathogen MLP, changing only the output to ordered quantiles.
2. Shared per-series two-block MLP with source identity and rank-4 pathogen
   adjustments. The shared encoder receives each series' own history; a small
   gated residual summarizes the allowed other series. Start its gate near zero.
3. Small residual time-mixing MLP, two blocks, sharing its per-series weights;
   use the same restricted context gate and output head as architecture 2.
4. Small residual basis MLP: two blocks that remove a fitted history component
   and add a four-week forecast component. This is an N-BEATS-inspired proposal,
   not a reproduction of a published model.
5. The shared MLP in architecture 2, replacing the last-value anchor with a
   damped recent-growth anchor. Estimate a slope from the latest three available
   transformed observations with their actual time gaps; damp successive weekly
   growth increments by 0.5. Fall back to zero slope with fewer than two points.

Use a median plus positive lower/upper increments and a monotone native-unit
output transform. Nonnegative admissions and ED proportions in [0,1] remain
ordered. Quantile loss uses the 23 submitted levels and the WIS-equivalent
weighting. Shared models preserve target/geography weights; independent fits
retain the corresponding within-component weights. Count parameters and record
training/inference time. Do not assume equal widths imply equal parameter counts.

### Label/input scope: 160

The 20 fixed contexts are the five quantile architectures × histories 8/12 ×
widths 32/64, with no spatial exchange, no covariates, learning rate 0.001 and
no artificial masking. For each, cross:

- Labels: flu admissions only; flu admissions plus flu ED (admission weight 1,
  ED weight 0.5, renormalized within the trained targets).
- Inputs: flu admissions only; flu admissions plus flu ED; all three admission
  series; all six admission/ED series.

Remove excluded information during both fitting and evaluation, including
derived dynamics, context messages and preprocessing statistics. Missing-value
masks remain available, but excluded series cannot leak through their features.
Compare to the corresponding all-six-label models in the architecture group.
For the independent pathogen architecture, removing other fitted components
should not itself improve the flu component; that comparison measures the
cost of fitting unnecessary components, not cross-pathogen parameter transfer.

### Reporting treatments: 120

Use the 20 contexts above with flu-admission labels and all-six target inputs.
Their unchanged-finalized-training/raw-evaluation controls already occur in
the scope group. For each context evaluate:

1. Retrain with finalized first-two-season inputs and archived/proxy recent-season
   inputs; retain finalized future labels.
2. Retrain with half-strength empirical admission reporting errors across all
   training seasons; ED inputs and all prediction labels stay finalized.
3. Keep the forecaster fixed; use a 50/50 distribution mixture from raw histories
   and histories corrected by the existing seasonal/context nowcaster.
4. Same fixed-forecaster mixture using the existing synthetic-trained neural nowcaster.
5. Same mixture using a small neural nowcaster fitted on real report-to-mature
   admission pairs only.
6. Same neural nowcaster as 5, pretrained to reconstruct hidden recent admission
   weeks from finalized permitted training histories, then fitted on exactly
   the same real pairs; use the same fine-tuning exposure as 5.

For forward evaluation, reporting evidence comes only from 2024–25; historical
pretraining uses 2022–23/2023–24/2024–25. For the retrospective fold, reporting
evidence comes from 2025–26 and historical pretraining uses
2022–23/2023–24/2025–26. Real reporting pairs require both the archived input and
its mature label to have been published by the relevant fitting cutoff. Use a
12-week reference age consistently and call it a maturity approximation, not
guaranteed final truth. Missing pairs and finalized proxies do not supervise
reporting errors. The error donor and corrector must also respect inner held-out
validation blocks. Only admissions are corrected; ED histories remain raw.

Rows 3–6 change evaluation-history handling, not forecaster weights. For
quantile outputs, mix the two distributions and invert the mixture CDF; do not
average their quantiles. Specify monotone piecewise-linear interpolation and
constant tails outside [0.01,0.99] as the initial interpolation assumption.
Report this tail assumption and compare it on the matched raw distribution.
Use each output pathogen's own deadline for correction caches. For the flu-only
rows this is the FluSight deadline. This does not reopen the project's accepted
historical cross-Hub correction approximation for older runs.

### Loss alignment: 80

Use the same 20 contexts, flu-admission labels and all-six inputs, unchanged
finalized training histories and raw reported evaluation histories. Compare
four new losses with their existing native-Q95 reference in the scope group:

1. Native-unit quantile loss divided by training-only baseline forecast error.
2. Quantile loss on log(1 + count), without magnitude normalization.
3. Log quantile loss divided by training-only baseline log forecast error.
4. Equal normalized native and log quantile loss, each using its own training Q95.

Estimate baseline error from a last-observed-value forecast on blocked
out-of-fold training tasks, pooled by horizon across states, with a separate
national scale. Floors are 1 count for native errors and 0.01 for log errors.
The log Q95 floor is 0.01. All are proposed numerical stabilizers. Preserve
equal-season and existing geography weights in these controlled loss comparisons;
thus they do not exactly reproduce the equal-task primary evaluation aggregation.
Never estimate training denominators from evaluation-season outcomes or Hub errors.

### Boosted quantile trees: 40

Train pooled flu-admission quantile trees on finalized permitted seasons with
their finalized four-week future labels. Use histories 8/12, inputs flu
admissions+ED/all six, depths 2/3, and minimum-leaf sizes 10/20/40/80/160.
Use log-count pinball loss, learning rate 0.05, up to 500 boosting iterations,
training-block early stopping, 80% row subsampling and seeds 42/43. Fit separate
horizon/quantile models sharing the same feature definition: current level,
recent changes, curvature, calendar, population, availability and report age.
Use only data available in each context. Sort quantiles to repair crossings.
No historical ILI augmentation is included. This is a pooled boosted comparator,
not a claim to reproduce Flusion or to obtain its published performance.

## Execution and selection after implementation

First run the reference and one example of each new code path; verify plausible
scientific errors only: target masks, unchanged labels, deadline and seasonal
isolation, native/log loss mathematics, quantile ordering and mixture weights.
Then run all 1,000 rows with both seeds. The fixed design avoids choosing the
architecture contexts from the evaluation scores midway through the sweep.

Use the four L40 GPUs and the shared manager queue initially, several small fits
per GPU. Keep matched comparisons on that GPU type. Benchmark representative
fits before giving a runtime estimate; parameter count alone does not predict
the cost of independent models or boosted horizon/quantile fits.

Rank only complete two-seed configurations on identical primary tasks. Show
both seed scores, paired differences, horizon and seasonal results. A small
gap is not convincing merely because 1,000 alternatives were searched. After
screening, separately propose five-seed confirmation of 3–5 candidates and an
equal-weight forecast ensemble; these are outside the 1,000-row budget.

Implementation requirements include the new heads/encoders, true target/input
scope, training losses, direct-quantile artifacts, reporting adapters and the
primary rank selector. Several overnight reporting mechanisms exist only in
saved research patches. Integrate them into the shared planner/scorer rather
than launching old private workflows with different evaluations. No speculative
CLI flags or launch commands are supplied as though they work today. The later
launch must include the exact manager plan, sbatch, status and rank commands.

## Deliberately outside this first sweep

Long historical ILI forecasting pretraining is promising but is not the same
as the admission-history nowcaster pretraining above. The current panel starts
in 2022, and the longer archive's usable seasonal coverage has not been verified.
Do not count an unbuilt historical training dataset among these 1,000 rows.
Likewise defer large Transformers, broad spatial attention and additional
all-source combinations until these controlled comparisons justify them.

Decision log: user selected a design-only deliverable and flu-admission log WIS
on 5 October. The proposal preserves the new reported-input regime and two
seeds, makes the forward season primary, and separates pipeline configurations
from unique forecaster fits. No production default or evaluation code changed.
