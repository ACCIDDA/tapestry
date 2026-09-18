# B1 revision experiment

User-authorized implementation and launch, 2026-09-17. Builds on
`analysis/b1-revision-seasons/README.md` and the original B1 screens.
Experiment: `B1-revisions-20260917`; suite: `B1-revisions`.

## Questions and design

Does forecast-only natural validation improve model selection? Does a weaker recent
objective help? Can probabilistic revision learning help while preserving B's direct
forecast path? Does training with realistic revision errors improve transfer?

32 configurations × seeds 42/43/44/45/46 = 160 runs, 480 held-out-season folds.
All runs use cap 300, patience 30, 12 context weeks, width64, latent16, batch8,
learning rate .001, 128 training draws, 256 fixed validation draws and 1,024
held-out evaluation draws. Epochs are selected separately per component and models
are freshly refitted, following the existing protocol. Target MLP and pathogen MLP
are the two backbones. No larger networks, new covariates, provider/release-age
features, or mask curriculum are included.

For each backbone, cross seven formulations with revision augmentation off/on:

| Formulation | Training objective |
|---|---|
| B direct with final flags | L_future |
| C parallel recent head, 10% | L_future + .10 L_recent |
| C parallel recent head, 20% | L_future + .20 L_recent |
| Existing two-stage, 10% | L_future + .10 L_recent |
| Existing two-stage, 20% | L_future + .20 L_recent |
| Gated revision branch on B, 10% | L_future + .10 L_recent |
| Gated revision branch on B, 20% | L_future + .20 L_recent |

This produces 28 configurations. Add B gap-only and B without artificial masks on
each backbone, with augmentation off, for four controls. Main panel uses mixed
masking at episode rate .5; augmentation-on independently selects an episode with
probability .5. Keep target/pathogen results separate before assessing consistency.
Five seeds quantify fitting variability, not independent epidemic seasons.

10% and 20% mean coefficients relative to the future task, NOT 10%/20% of total
loss. Each task retains partition-wide scientific weights and training-only Q95
normalization. Visible supplied finals receive no recent loss; artificial hiding
restores reconstruction supervision. With no eligible recent targets, no weight
is reassigned to another task. Loss is native-unit fair CRPS, not Hub-relative WIS.

Every configuration selects epochs on **future loss only with natural inputs**:
no validation dropout and no revision augmentation. Fold leakage restrictions
remain in force. All formulations use the same future-eligible fold construction,
including two-stage. Thus natural nowcast support is common across recent-head
models; direct B has no recent output. Previous two-stage used combined selection
and extra recent-only episodes; new comparisons intentionally replace that recipe.
Old scores are contextual references, not matched controls for this experiment.

## Probabilistic nowcast and B-preserving branch

Recent finals remain uncertain given Wednesday inputs; use sampled recent outputs
with the same proper scoring rule as forecasting. C has no feedback. Existing
two-stage still uses sampled recent anchors. The new gated branch:

- Reuses B's direct backbone, future head, dynamics, input flags and future anchor.
- Uses a shared recent representation with distinct stochastic heads for visible
  report corrections and missing-value reconstruction.
- Anchors visible corrections on each recent week's own observed value; missing
  cells use the last visible focal value (or B's small prior) as an anchor.
- Uses an independent recent latent and future latent; pairing retains the sampled
  recent/future relationship. Dependence across separately fitted components is
  not modeled. Marginal CRPS/WIS do not establish joint calibration.
- Sends both recent residuals and availability/finality indicators into a small
  adjustment network. A context-dependent sigmoid gate controls four horizon
  adjustments to B's future residuals. The adjustment output is zero-initialized;
  gate starts near .10, so the initial future prediction matches B to floating-point
  tolerance while permitting learning. This is joint training from scratch, not
  transfer from an already fitted B checkpoint or a frozen B model.
- Passes visible finals through exactly in native units. They contribute zero
  correction to the branch. Artificially hidden values and flags do not enter it.
- Uses existing positive-count / bounded-ED inverse transforms. The branch does
  not require the forecast to inherit all recent uncertainty. It can learn to use
  or suppress the adjustment. The branch may still worsen forecasts; this is a test.

Nominal report ages are fixed at 11 and 4 days before Wednesday and already encoded
by input positions. No extra nominal-age feature is needed. Actual source update
ages can vary but are deliberately excluded at the user's request; no metadata
ablation is part of this experiment. Wednesday issuance predicts the following
Saturday and three further Saturdays, preserving the existing date convention.

## Revision augmentation and leakage boundaries

Build a residual bank separately inside each component's inner fitting partition,
and rebuild from the permitted refit partition for final fitting. A donor is one
issuance block spanning two recent weeks, all six channels and all locations.
Only available provisional inputs with permitted recent reference labels qualify.
Supplied finals, held-out/inner-hidden labels and missing reports never supply errors.

Residual = normalized transformed report − normalized transformed final, using
only that fitting partition's scales. For a chosen recipient episode, pick one
donor block and construct synthetic reports as recipient final + donor residual
in transformed space. Apply only where both donor and recipient qualify for the
same channel/location/age and where training dropout leaves the recipient visible.
Inverse-transform to native counts/proportions. Other inputs, labels, availability
and final flags remain unchanged. This replaces a preliminary report; adding a full
residual to an already biased report would double the reporting bias.

One donor preserves the available cross-location/channel/age error dependence.
There is no interpolation, nearest-neighbor matching, season/provider conditioning,
or pooling of absent target/age slots. Unsupported cells retain the original report.
A fold with no genuine newest-week revision examples cannot acquire them through
this augmentation; that limitation is explicit. Zero counts use the existing count
transform; ED logits use the existing bounds; synthetic counts are projected to
nonnegative transformed values before inversion. These are material assumptions.
Residual transfer across eligible seasons is the augmentation hypothesis, not a
claim of stationarity established by the audit. Donors may be from the recipient's
own episode, which is harmless resampling of fitting information.

Separate seeded NumPy streams draw masking and revision donors. Saved mask audits
include epoch-by-episode donor indices and ordered training issuances. Manifests
record donor cell support by age/channel; training history records applied cell
counts. Validation, held-out evaluation and stress prediction never augment.

## Evaluation

Primary remains the existing natural-input forecast location-relative WIS against
Hub on frozen support, using the scientific season/target/geography weighting.
Recent/gap/outage forecasts are separate stress diagnostics. Calibration remains
reported. Evaluation uses 1,024 draws for every run, so differences against older
256-draw screens include an evaluation-budget change. New baselines are rerun.

Nowcasts receive a separate common-support ranking against **each target week's
own genuine preliminary report**, excluding supplied finals and cells without that
report. This diagnoses revision skill rather than persistence of a different week.
The saved baseline definition and score version distinguish it from historical
latest-visible-value rankings; the scorer rejects mixed baseline versions. The
shared scorer rejects nonpositive denominator groups rather than silently dropping
them; the pinned dataset audit found none on the combined natural recent support.
Forecast and nowcast rankings must never be combined into one score. Per-cell
recent mechanism/age results remain available for bias, coverage and reconstruction.

All inputs retain the retrospective finalized-history/fallback policy. No claim of
Wednesday-operational performance or untouched prospective validation is made.
The original seasonal audit uses 49,961 genuine recent reports but these are not
independent training examples; weeks/locations and the two ages are correlated.

## Plan, launch, monitor and resume

From `/proj/jlessler/projects/tapestry-all/tapestry`:

```bash
.venv/bin/python -m tapestry.models.manager plan -e B1-revisions-20260917 --suite B1-revisions --seeds 42 43 44 45 46 --eval-members 1024 --retrospective --device cuda
LANES=8 GPUS=6 sbatch --job-name=B1-revisions-L40 --array=0-3 --nodelist=g1803jles01 --time=2-00:00:00 data/experiments/B1-revisions-20260917/code/scripts/jlessler.sbatch B1-revisions-20260917
LANES=8 GPUS=6 sbatch --job-name=B1-revisions-H100 --array=0-1 --nodelist=g1803jles02 --time=2-00:00:00 data/experiments/B1-revisions-20260917/code/scripts/jlessler.sbatch B1-revisions-20260917
.venv/bin/python -m tapestry.models.manager status -e B1-revisions-20260917
.venv/bin/python -m tapestry.models.manager rank -e B1-revisions-20260917 --allow-incomplete
# Once complete:
.venv/bin/python -m tapestry.models.manager rank -e B1-revisions-20260917
```

Do not replan an active experiment: planning replaces its source snapshot. Repeat
launch commands after allocations end to resume pending work; `status` prints exact
resume guidance, including failed tasks. Shared queue claims seeds without duplicate
ownership. Notifications stay on. Forty-eight concurrent seed processes is the
existing screen's layout, not a new throughput guarantee. The two-day allocation
limit is not a completion-time estimate.

A separate four-configuration GPU pilot covers B augmentation, C augmentation,
two-stage augmentation and gated augmentation at two epochs with smaller draw
budgets. It is an execution check, not scientific evidence or part of the ranking.
Pilot command provenance and job IDs are recorded below after submission.

## Log

- 2026-09-17: implemented requested .10/.20 objectives, natural forecast selection,
  gated probabilistic revision extension, and fitting-only block augmentation.
  Preserved existing uncommitted formulation-suite changes. Excluded reporting
  metadata and mask curriculum from this experiment. Added targeted scientific
  checks for loss weights, final bypass, hidden values and augmentation units.

The fold-level `recent-diagnostics-*.csv` explicitly separates recent mechanism
(revision versus artificial reconstruction), reporting age, location, season and
target. It includes median MAE/bias, CRPS/WIS and 50%/95% coverage. Unchanged-report
MAE is populated only for visible genuine reports, never as an attainable baseline
for reconstruction of a masked value. These local native-unit diagnostics do not
replace the scientifically weighted global rankings.

Pilot provenance:

```bash
# Exact manager plan with four canonical scenarios:
bash analysis/b1-revision-seasons/pilot-plan.sh
LANES=4 GPUS=1 sbatch --job-name=B1-revisions-pilot --array=0-0 --nodelist=g1803jles01 --time=02:00:00 data/experiments/B1-revisions-pilot/code/scripts/jlessler.sbatch B1-revisions-pilot
.venv/bin/python -m tapestry.models.manager status -e B1-revisions-pilot
.venv/bin/python -m tapestry.models.manager rank -e B1-revisions-pilot --allow-incomplete
```

- Pilot submitted as Slurm array 1540158 on L40, four concurrent lanes. Unit
  scientific checks passed (nine objective/augmentation/bridge cases and two
  nowcast-baseline cases). The pilot snapshot precedes only the additional
  diagnostic CSV convenience export; its per-cell metrics and native forecasts
  already retain the information needed for those diagnostics.

- Pilot 1540158 completed successfully in 1:43: four runs, all three folds each,
  with 36 component-selection mask audits confirming zero artificial validation
  dropout. All fold manifests select forecast loss. Shared forecast and same-week
  nowcast rankings both completed; pilot scores are not scientific comparisons.
- Full experiment planned and submitted: L40 array **1540593** (0–3), H100 array
  **1540594** (0–1), eight lanes each, two-day allocation limit. Source snapshots
  are frozen. Use the commands above to monitor/resume; do not replan while active.

- 2026-09-18: all 160 runs complete (L40 1540593, H100 1540594, ~6:02 each). Both
  rankings produced; results published to the B1 results page under
  `docs/results/b1-overnight/index.md`, generated by `scripts/plot_b1_revisions.py`.
  Headline: the gated branch is the only formulation that beats the same-week
  preliminary report (0.890 at best, ~11%) while leaving B's forecast path intact;
  forecast leadership stays with the gap-only/no-mask controls (0.938). C parallel
  is ~2.7× worse than the report it corrects. Two scorer artefacts were found and
  are documented on the page rather than reported as results:
  (1) the nowcast `wis_ratio` averages per-location ratios, and two 2023-2024 RSV
  ED-visit locations (ID, VA) have a preliminary report equal to the final to
  floating-point tolerance, so their ~1e-10 denominator produces ~1e6 ratios that
  dominate the mean. The `ensemble_wis <= 0` guard in `totals.py` does not catch a
  small positive denominator. Those two of 819 location-cells are excluded; the
  pooled aggregation gives the same ordering. The forecast ranking is unaffected.
  (2) outage reconstruction collapses onto a shared fallback: 16 of 24 configurations
  (every gated and C run) agree seed-by-seed within 0.001 CRPS, so that panel is not
  ranked. Recent-stress reconstruction does differentiate and is reported instead.
