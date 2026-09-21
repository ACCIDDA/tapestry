# Review of B1 revision results (2026-09-18)

Scope: correct the published interpretation and plots using saved scores from the
complete 160-run experiment. No training, new model predictions, or model-scoring
jobs are launched. Existing scientific forecast scores are unchanged. The registered
nowcast totals and ranking remain intact; post-hoc exclusions are explicitly labelled.

Reproduce from the repository root:

```bash
.venv/bin/python analysis/b1-revisions-review/audit.py
.venv/bin/python scripts/plot_b1_revisions.py
.venv/bin/python -m mkdocs build --strict
```

Outputs are under `docs/results/b1-overnight/revisions/`. The audit reads each saved
recent per-cell parquet and aggregates existing errors. It does not inspect plots.

Material assumptions and weighting:

- Scaled CRPS is cell CRPS divided by the saved training-only native Q95 loss scale
  for that fold, target and location. It is dimensionless and differs from the
  registered location-relative WIS endpoint.
- Mean error within location, target, season, reporting age and condition; then
  states/DC receive 80% and US 20% (renormalized if one group is absent), targets
  receive admissions 1 and ED .5, and available seasons receive equal weights.
  Ages remain separate. No missing target support is imputed. Diagnostic support
  differs by age and from Hub forecast support.
- Family tables average every configuration/seed equally within C, two-stage and
  gated (eight configurations, five seeds each). These are descriptive summaries
  of related fits, not predictive mixtures or independent observations.
- Matched reconstruction intersects natural genuine-report cells with the exact
  cells hidden under recent/outage stress, keyed by issuance, observation week,
  target, location and age. Supplied finals are excluded from this paired comparison.
  This measures the masking intervention, which can hide multiple inputs together;
  it does not isolate the causal contribution of one cell.
- Unchanged-report absolute error equals deterministic CRPS (and deterministic
  WIS), allowing a scaled CRPS comparison with the probabilistic model. The ratio
  of aggregate scaled scores does not divide by near-zero individual report errors.
- The adjusted nowcast WIS ratio uses the existing report's post-hoc native summed
  baseline WIS threshold 1e-6, not a newly validated universal tolerance. The audit
  retains a cutoff sensitivity and the smallest baseline groups. Near-perfect
  baselines still count in the scaled absolute-error diagnostics.
- A pooled native WIS ratio changes location importance and does not preserve the
  80/20 geography estimand. Family ordering can survive while individual ordering
  and whether two-stage beats the report change.
- Admission quantiles were rounded during export. A saved zero median does not
  imply that every unrounded trajectory is exactly zero. CRPS approaching the
  zero-prediction error corroborates near-zero distributions in the outage checks.
- Similar aggregate errors alone do not identify a model mechanism. Frozen code
  executes recent heads under outage and uses the small B0 anchor when focal
  history is absent. This supports investigating the anchor; it does not prove
  that an anchor change alone will fix the model.
- Paired contrast bars are SD, not confidence intervals. Five seeds reuse the
  same seasons. No statistical significance or equivalence is inferred from
  whether a mean ± SD bar includes zero.
- These are retrospective development seasons, with finalized older histories
  and supplied-final fallbacks. No new evidence on operational or prospective
  performance, or causal attribution to seasonal revisions, is introduced.

Log: replaced unsupported universal forecast/nowcast claims, mixed-unit
reconstruction summaries, the outage-scorer diagnosis, and misinterpretations of
pooled ordering and SD bars. Corrected both the page generator and workflow note
so regeneration preserves the corrections.
