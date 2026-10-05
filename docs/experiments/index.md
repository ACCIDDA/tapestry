# Model runs

Reports are ordered newest first. Dates identify the report, not necessarily the final training job. The Kinsa report date is its first recorded publication in this repository. Use the [workflow](../workflow.md) to plan, resume and analyze experiments.

## Completed reporting-delay study

- [2026-10-05 · End-of-run summary](overnight-b2-results-20261005.md): plain-language model descriptions, training procedure, results, and limitations.
- [Artificial vintaging](vintage-overnight-20261005/results.md) and [separate or joint nowcasting](nowcast-overnight-20261005/results.md): completed three-seed comparisons.
- [Matched comparison with Google Hub forecasts](google-comparison-20261005/results.md): count-scale results and the different official scoring rule.

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
