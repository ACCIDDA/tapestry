# Frozen forward formulation benchmark

Predeclared September 18, 2026 before fitting. This is forward development;
2025–26 has already been explored.

- Training information cutoff: **July 26, 2025 end of day UTC**. Labels and
  finalized input proxies are latest available by that cutoff, aged at least
  28 days (observation weeks through **June 28, 2025**).
- Test issuances: **July 30, 2025–July 29, 2026**, Wednesdays. Forecast target
  weeks: **August 2, 2025–August 1, 2026**, horizons 0–3. References pinned to
  September 16, 2026. All parameters and transformations remain frozen.
- Three candidates: Direct B with availability and supplied-final flags;
  shared joint model with a learned gated correction and forecast + .25 recent
  loss; independent recent-only nowcasters feeding sampled finalized-value
  hypotheses to forecast-only models trained on cutoff-final contexts.
- Six independent target MLPs per forecasting candidate; existing B0/B1 backbone,
  12 weeks, width 64, latent 16, fourth-root admission rates, logit ED, geography,
  dynamics, annual calendar, legacy decoder, global noise, no spatial attention.
- Seeds **42, 43, 44**. **100 fixed epochs**, batch 8, 128 training draws,
  256 predictive draws; Adam .001, no weight decay, gradient clipping 5.
  No validation selection, calibration, masking, revision augmentation or sweep.
  There is no internal held-out partition to leak through overlapping windows;
  every training input/label is bounded by the historical cutoff and maturity
  masks, with no test-season label in fitting or normalizers.

## Matched and unavoidable differences

Forecast architecture, eligible forecast labels, forecast epoch budget, scientific
loss weighting, seeds and evaluation tasks match. Existing native fair CRPS,
training-Q95 scales and season/target/geography weights are reused. Common input
transforms are fitted on deduplicated cutoff-final training contexts; common loss
scales use unique eligible training label weeks. No test data fit scales.

The independent pipeline costs six additional nowcaster fits per seed (also
100 epochs). Its recent-only networks use the same recent heads as the joint
model, with no forecast loss or forecast gradients. Unused forecast/gate modules
remain allocated in that reused backbone but receive no gradients from recent
loss; their outputs are discarded. Nowcaster labels can include a late training
issuance without a forecast label; forecast fits use the same forecast-eligible
issuances across all candidates.

All candidates fit older context on cutoff-final proxies and deploy on actual
Wednesday older context. Direct and joint use real recent reports during fitting.
The separate forecaster fits exact cutoff-final recent values, then receives 256
sampled recent hypotheses at deployment (one forecast draw per hypothesis).
All six target nowcasters supply recent inputs to each forecast model. Samples
from independently fitted targets are paired by member index; cross-target
nowcast dependence is not learned. Estimated recent inputs carry available and
supplied-reference flags for the finalized-input forecaster; these flags describe
the sampled hypothesis, not certified source finality. Sampling propagates
uncertainty but does **not** solve the training/deployment input mismatch.

Recent supervision intersects reference availability, maturity and source/regime
eligibility. NHSN observation weeks must end November 1, 2024 or later. NSSP uses
its own archive coverage. Visible revision and covered missing reconstruction
have distinct heads in the existing gate model and are evaluated separately at
4 and 11 days. Unknown archive coverage never supervises recent losses.
Older forecast examples remain eligible regardless of recent-loss eligibility.

## Scoring and declared denominator policy

Reuse `tapestry.evaluation.totals`: 23-quantile WIS, equal states/DC within 80%
geography weight, US 20%, admissions target weight 1 and ED .5, equal seasons.
Only one forward season exists. Seed means and sample SD quantify fitting
variability, not independent-season uncertainty; no location/seed significance
test is interpreted as season replication.

Primary ranking uses the existing frozen Hub ensemble tasks intersected once
with the new forward label support, with reference values replaced by the pinned
September 16 truth. Candidate tasks are identical. Ensemble quantiles are scoring
comparators only. Full-period WIS, training-Q95-scaled WIS and 50/80/90/95% coverage
also cover available labels outside the target-specific Hub subset. Target,
season, geography, location and horizon CSVs accompany matched tables and plots.

For visible recent reports, compare against **that observation week's own
preliminary value**, using aggregate location/target/age baseline MAE, not
pointwise division. Predeclare a near-zero denominator when mean baseline MAE /
training Q95 is **≤ 0.0001**. Mark relative MAE and WIS undefined for those cells;
report flags/counts and do not floor or replace denominators. Always report
native errors, MAE/Q95, WIS/Q95 and coverage on all eligible recent predictions.
Missing reports have no preliminary baseline and no revision ratio; report their
reconstruction scaled errors and coverage separately. Direct B has no recent
output and receives no fabricated reconstruction score. No post-hoc metric
substitution or nowcast-driven ranking is used.

## Availability assumptions

See [data provenance and coverage](../data/forward-2025.md). Native timestamps
are advertised availability and Git timestamps publication proxies; archive
interior completeness is assumed. These do not certify strict provider release
times or intraday availability. The 28-day finality convention is provisional.
Four-day NSSP visible correction training starts only June 18, 2025 and is sparse.
Interpret this as an archive-as-of benchmark conditional on those assumptions.

## Reproduce

```bash
.venv/bin/python -m tapestry.models.manager plan -e Forward-2025 --suite Forward-2025 --seeds 42 43 44 --device cuda
LANES=5 GPUS=2 sbatch --job-name=Forward-2025 --array=0-1 scripts/jlessler.sbatch Forward-2025
.venv/bin/python -m tapestry.models.manager status -e Forward-2025
.venv/bin/python -m tapestry.models.manager rank -e Forward-2025
```

Planning freezes source and input hashes. The ordinary manager, GPU dispatcher,
scorer and ranking remain the execution path. `rank` also writes the results page
at `docs/results/Forward-2025/index.md`. Existing runs resume through the manager;
check active Slurm allocations before resubmitting.

## Execution record

The one-epoch `Forward-2025-check` run verified all three fitting, sampling,
scoring and reporting paths. Its scores did not select the 100-epoch budget or
any hyperparameter. Four focused scientific checks passed. The full run started
as job **1647980** with three workers on one GPU. On the user's request for all
seeds concurrently, job **1648225** added six workers on a second GPU using the
same locked queue. No fits were restarted or duplicated. Report job **1648159**
waits for both allocations to succeed and executes the ordinary manager `rank`,
which writes the final tables, plots and report provenance. The working-tree
reporter preserves undefined missing-report baselines through aggregation.

The updated launch recipe above provides ten worker slots on two GPUs for nine
runs. For a standalone local launch, `manager run --parallel-seeds --fit-workers 9`
also schedules seeds independently; do not combine that local runner with an
active Slurm dispatcher on the same experiment.

- September 19, 2026: all nine runs completed. Saved predictions were analyzed
  and compared against four fixed B1 recipes on identical 2025–26 tasks, truth
  and scoring weights; no additional fitting. See the
  [ranking, fans, heatmaps and conclusions](../results/Forward-2025/analysis.md).
  The separate pipeline leads the forward candidates; legacy B1 comparisons
  retain retrospective information differences and are not causal ablations.
