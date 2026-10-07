# Proposed performance experiments — 5 October 2026

These are hypotheses, not measured improvements. The request is interpreted as
ideas for the current B2 probabilistic admission/ED forecasters and their
reporting-delay extensions. The working objective is lower Hub-relative WIS
with useful predictive coverage under Wednesday information availability.
No training or scoring jobs were launched for this note.

## Reference and evaluation assumptions

Use the pathogen-specific MLP with distance-weighted spatial sharing and no
additional covariates as the first reference. For forward development evaluation,
train on 2022–23, 2023–24 and 2024–25 finalized target histories, learn unchanged
finalized t+1 through t+4 labels, and evaluate 2025–26 Wednesday archived target
histories. Retain the documented B2 availability schedule and explicitly report
where absent archives are replaced by finalized-value proxies. Start with the
overnight recipe without artificial training masking or finality indicators.
Keep hardware, seeds, prediction draws, masks and scored tasks matched.

The 2025–26 season has already informed many decisions; it is a development set,
not an untouched test. Fit any calibration, mixture weights, loss scales or
selection rules only inside the permitted training seasons, using predictions
made without fitting on the validation outcomes. Temporal splits must also
respect label maturity and overlapping future-label windows. Record missing
historical targets rather than inventing observations. A 2024–25 evaluation
trained on 2025–26 is retrospective and should be secondary evidence.

The original finalized-evaluation B2 sweep is a different protocol: it supplied
finalized evaluation histories and the selected model used 20% training masking.
Its coverage findings motivate a fresh diagnostic; they do not establish the
coverage of the overnight archived-input models.

## Priority experiments

1. **Pool predictions from independently fitted seeds.** Reuse the reference
   checkpoints and draw equal numbers of complete forecast trajectories from
   each seed, then recompute quantiles and WIS. Averaging individual seed scores
   is not evaluation of this ensemble. Keep the total predictive draw count
   matched to individual-model comparisons. A second experiment can add a
   target-specific MLP with neighbor sharing and Kinsa, retrained on the same
   seasons, finalized histories and finalized future labels. Begin with equal
   weights; avoid selecting components from the 2025–26 ranking.

2. **Calibrate bias and predictive spread.** Reuse fitted forecasters. Fit a small
   postprocessing model on training-season out-of-fold predictions against
   finalized future labels: target/horizon bias and spread parameters, pooled
   across states, with shrinkage toward no adjustment. Use a monotone transform
   that respects nonnegative counts and bounded proportions. Apply it to the
   2025–26 forecast distribution without changing inputs. Inspect both median
   error and coverage; widening intervals alone may worsen WIS.

3. **Test training scales that reflect forecast difficulty.** Retrain the same
   reference model with unchanged inputs and labels, replacing historical Q95
   loss scales with positive, pooled scales estimated from a simple baseline's
   out-of-fold errors inside training seasons. Preserve the scientific target,
   geography and season weights. Q95 measures magnitude, whereas the evaluation
   denominator measures forecast error. This is an approximation to the evaluation
   objective, not exact Hub-relative WIS training. Never use evaluation-season
   Hub errors or individual realized outcome errors as training denominators.

4. **Predict departures from damped growth.** Retrain the reference architecture
   on the same inputs and finalized future labels, changing the decoder anchor
   from the latest observed level to a short, smoothed, damped growth projection.
   Let the network learn residual corrections around that projection. Keep zero
   growth as a matched alternative. This could improve rising/falling phases,
   but recent reporting revisions could also make growth extrapolation harmful.
   The earlier failed current-level anchoring pilot is not this treatment.

5. **Make correction strength depend on reporting reliability.** Keep the
   finalized-input-trained forecaster fixed. Compare the existing equal mixture
   of predictions from raw and corrected admission histories with a small set
   of mixture weights learned from training-season out-of-fold forecasts.
   First pool by pathogen and report age; only add a growth interaction if there
   is enough independent validation evidence. Leave ED histories unchanged.
   The separate nowcaster learns finalized recent-history labels from synthetic
   reports on permitted training trajectories, with reporting errors from 2024–25
   for the forward fold. This changes evaluation-history handling, not forecaster
   training. Do not use archive/proxy identity as a fictional operational feature.

6. **Test simpler output distributions.** Retrain the same encoder with a direct
   noncrossing quantile head at the 23 scoring levels, learning the same finalized
   future labels with the corresponding weighted quantile loss. Compare against
   the stochastic sample decoder with unchanged training inputs, scientific
   aggregation and 2025–26 archived evaluation histories. This asks whether the
   sample generator is needed for marginal forecast skill. A marginal quantile
   model alone does not provide coherent joint epidemic trajectories.

Start with seed pooling and calibration, then test the loss scales and growth
anchor separately. Check median bias, coverage, and target/horizon WIS alongside
the unchanged aggregate score. Larger inference sample counts are already under
study and should be standardized across comparisons, not counted as new training
evidence. These suggestions do not assume any new architecture or adjustment
will improve the strongest model.

## Evidence behind the priorities

- [Original B2 results and coverage](b-2-t0/index.md).
- [Completed reporting-delay experiments](overnight-b2-technical-results-20261005.md).
- [Nowcasting comparisons](nowcast-overnight-20261005/results.md).
- [Evaluation sample-count comparison](b2-eval-samples-20261005.md).

The reporting-delay results favor modest inference changes to the strongest
forecaster over further broad synthetic-input sweeps. The architecture documents
the Q95-normalized CRPS training objective, while the evaluation uses relative
WIS. These motivate the proposed tests; neither observation proves their benefit.

## Partial vintage archives: further nowcasting ideas

The follow-up request is interpreted as having finalized epidemic histories for
more seasons than have archived preliminary reports, and wanting to avoid
synthetically revising the entire forecaster training corpus. The proposals below
can leave the finalized-history-trained forecaster fixed. They are untested ideas.
The existing seasonal/context models already implement seasonal development,
partial pooling, causal adaptation, age-dependent level/growth losses, revision
momentum features and causal correction-strength selection. Those are existing
comparators, not new proposals.

For a concrete forward development experiment, fit epidemic-history components
on 2022–23/2023–24/2024–25 finalized data and fit reporting components only on
genuine 2024–25 archived report/label pairs. Evaluate reconstructed histories and
the fixed forecaster on 2025–26 archived reports against finalized recent and
future values, respectively. Keep absent pairs missing. Do not supervise
reporting behavior using finalized-value archive proxies. The frozen forecaster
learned finalized t+1..t+4 labels from unchanged finalized histories. Distinguish
a frozen reporting model from a separately evaluated online model that adds
2025–26 training pairs only after their reference reports actually mature.
Delay-12 reports are an approximate maturity endpoint, not guaranteed final truth.

1. **Fuse an epidemic prediction with a reporting observation model.** Use the
   full finalized training history to fit a short-gap trajectory predictor that
   supplies plausible recent values from older context. Separately estimate
   signed report errors by age/source from real archived pairs. Infer recent
   histories that agree with both the trajectory prediction and the observed
   preliminary reports, then feed sampled histories to the fixed forecaster.
   Start with a low-dimensional level/growth update over the latest one to three
   weeks, not unrestricted importance sampling over all locations and dates.
   This assumes the learned dynamics and reporting errors transfer to 2025–26.
   Uncertainty must be validated; calling the construction probabilistic does
   not establish calibration. Prevent double use of a preliminary observation
   as both prior evidence and an independent reporting likelihood.

2. **Fit a small reporting adapter on real pairs.** Pretrain a recent-history
   reconstruction encoder on older finalized histories using hidden recent
   values and unchanged finalized reconstruction labels. Freeze it initially,
   then fit a small age/source-conditioned nowcast head on actual archived
   2024–25 reports and their mature reference labels. The synthetic masks teach
   epidemic dynamics; they are not presented as simulated reporting revisions.
   This transfers a representation rather than transferring sampled reporting
   errors onto every old epidemic. Compare with an otherwise identical adapter
   without pretraining and with the existing seasonal/context nowcaster.

3. **Learn from intermediate revisions before maturity.** Supplement final-value
   supervision with real age-a to age-b report pairs when both are already
   available, even if the maturity report is not. Model signed changes and the
   remaining correction jointly with the mature pairs. Account for elapsed
   report ages and do not count missing releases as zero revisions. This could
   let the reporting component respond faster to a changed publication process.
   The seasonal adaptive-chain model already uses intermediate development;
   the new experiment would add this supervision to the neural reporting adapter,
   not rename the existing chain. Correlated pairs do not create independent
   observations, and intermediate changes alone do not identify final values.

4. **Infer an issuance-wide reporting disturbance.** Summarize revisions to older
   event weeks visible at the current issuance, pooling within a reporting
   source across locations and, where appropriate, pathogens. Use a heavily
   shrunk common disturbance to update the newest-week correction or uncertainty.
   This extends existing local revision-momentum features with a shared
   reporting factor. It assumes some simultaneous changes reflect reporting
   conditions; evaluate that assumption rather than equating it with fact.

First compare the real-pair adapter with and without finalized-history pretraining
against the seasonal/context model, then evaluate their reconstructed histories
through the same frozen forecaster. Report nowcast level/growth error, uncertainty
coverage and downstream WIS separately. A better reconstruction score does not
automatically imply better forecasts. Use the same actual archive support for
the direct nowcast comparison and disclose any proxies in downstream replay.
