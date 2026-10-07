# Model runs

Reports are ordered newest first. Dates identify the report, not necessarily the final training job. The Kinsa report date is its first recorded publication in this repository. Use the [workflow](../workflow.md) to plan, resume and analyze experiments.

## B4.polish and B5 (6–7 October 2026)

- [2026-10-07 · B4.polish and B5](b4-polish-b5-20261007/index.md): 2026–27 submission (A + B, five seeds, trained on all four seasons; production fits done), B5 exploration of 400 configurations, covariate representations, and new-seed confirmation; what worked and what did not.

## Completed B.3 pilot and B.4 flu study

- [2026-10-06 · B4.refineTop2](b4-refinetop2-20261006/index.md): sweep setting effects, ensembles of saved forecasts, training-season calibration, training-history treatments and four-week-total loss weights on the two flu leaders.
- [2026-10-06 · B.3 pilot and B.4 consolidated analysis](b4-flu-study-20261006/index.md): broad 600-configuration search, covariate/correction refinement, and matched output-head comparison; methods, results, figures and conclusions for all three experiments.

## Planned

- [2026-10-06 · B5 plan](b5-plan-20261006.md): stochastic nowcasting, multi-season training vintages, nowcaster and loss formulation (executed as [B5](b4-polish-b5-20261007/index.md)).
- [2026-10-05 · B3 sweep plan](b3-sweep-plan-20261005.md): 1,000 configurations × 2 seeds × both folds, with six training treatments for reporting delays.

## Completed reporting-delay study

- [2026-10-05 · End-of-run summary](overnight-b2-results-20261005.md): plain-language model descriptions, training procedure, results, and limitations.
- [Artificial vintaging](vintage-overnight-20261005/results.md) and [separate or joint nowcasting](nowcast-overnight-20261005/results.md): completed three-seed comparisons.
- [Matched comparison with Google Hub forecasts](google-comparison-20261005/results.md): count-scale results and the different official scoring rule.
- [B4 flu runs versus Google](b4-google-check-20261006.md): pooled WIS head-to-head on the 5,712 shared 2025–26 flu admission tasks.

## Nowcasting

All new nowcast evaluations include a [baselinenowcast point comparator](../reference/baselinenowcast.md).

- [2026-10-02 · Context-conditioned nowcaster](context-nowcast-20261001/index.md): recent level/growth scoring, causal residual training, and matched C1 forecasting under the documented availability schedule.
- [2026-10-01 · Seasonal online nowcasting](seasonal-nowcast-20261001/index.md).
  [Model versus actual revisions, one row per Wednesday](seasonal-nowcast-20261001/nowcast.html).
- [2026-10-01 · Local reporting triangles](reporting-triangle-20261001/index.md).

## Forecasting

- [2026-10-02 · Expanded B2 revision comparison](b2-weekend-20261002/index.md): 1,440 configurations, three seeds, two seasons; direct artificial revisions, trained nowcaster then forecaster, and joint recent reconstruction plus forecasting.

- [2026-10-02 · C1 with locally matched historical nowcast errors](c1-local-errors-20261002/index.md): four training-input treatments, three seeds, and separate 2024–25/2025–26 CV results.

- [2026-10-02 · B2 reporting-error augmentation: completed comparisons](b2-reporting-augmentation-20261001/index.md).

- [2026-10-01 · B2 input-encoding controls](b2-flag-controls-20261001/index.md).
- [2026-10-01 · B2 leaders: final, nowcast, and vintage input replay](b2-input-replay-20261001/index.md).
- [2026-09-30 · B2 — Covariates, spatial sharing and masking with finalized T-0 inputs](b-2-t0/index.md)
- [2026-09-25 · Covariate encoders and spatial sharing with reporting gaps](forecast-geography-v2/index.md)
- [2026-09-22 · Adding national Kinsa to finalized target histories](kinsa-finalized-5seeds/index.md)

## Archive

[Archived experiments](archive/index.md). These are historical results, not current workflow instructions.
