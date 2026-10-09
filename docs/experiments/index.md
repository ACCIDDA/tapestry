# Model runs

One line per experiment phase, newest first. Each report's "Model choices" table says what
its models trained on, what they were scored on and with which seasons
([Model choices A–F](../reference/model-choices.md)). Submissions are in the
[submission log](../submissions.md); how to run an experiment is in the [workflow](../workflow.md).

## Influenza submission, October 2026

- **Submission choice (8 October):** [why B7 replaced the on-time System2 file](submission-choice-review-20261007.md), from the [B7 versus System2 comparison on real reports](b7-submission-comparison-20261007/index.md).
- **B7 production (7 October):** [three recipes retrained on all four seasons, ten seeds each, and their 7 October forecast](b7-production-20261007/index.md); this is the merged submission.
- **B7 folds (7 October):** [individual recipes retrained with the prescribed 2025–26 reporting errors, three held-out seasons](b7-folds-20261007/index.md) ([versus System2](b7-folds-20261007/vs-system2/index.md)).
- **B6 (7 October):** [search, extension and the System2 selection](b6-forecast-today-plan-20261007.md); results: [original campaign](b6-core-complete-1400/index.md), [extension](b6-extension-complete-1530/index.md), [System2](b6-system2-nine-1605/index.md), [saved ensembles](b6-saved-ensembles-20261007/index.md), [by horizon](b6-selected-horizons/index.md), [reporting regime on 7 October](b6-reporting-regime-20261007/index.md).
- **Season peak (7 October, stopped):** [whole-season paths versus peak loss](peak-study-20261007/index.md) and [autoregressive rollout](peak-rollout-20261007/index.md).
- **B4.polish and B5 (6–7 October):** [new seeds, training windows, B5 exploration and confirmation](b4-polish-b5-20261007/index.md) ([plan](b5-plan-20261006.md)).
- **B3 pilot and B4 flu study (5–6 October):** [consolidated analysis](b4-flu-study-20261006/index.md), [refineTop2](b4-refinetop2-20261006/index.md), [B3 pilot](b3-pilot-20261005/index.md) ([plan](b3-sweep-plan-20261005.md)), [B4 versus Google](b4-google-check-20261006.md).

## Reporting delays, 1–5 October 2026 (training routes since deleted)

- [End-of-run summary](overnight-b2-results-20261005.md) and [combined technical results](overnight-b2-technical-results-20261005.md); [artificial vintaging](vintage-overnight-20261005/results.md), [separate or joint nowcasting](nowcast-overnight-20261005/results.md), [matched comparison with Google](google-comparison-20261005/results.md), [256 versus 2,048 samples](b2-eval-samples-20261005.md).
- [Expanded B2 revision comparison](b2-weekend-20261002/index.md), [C1 with local nowcast errors](c1-local-errors-20261002/index.md), [B2 reporting-error augmentation](b2-reporting-augmentation-20261001/index.md), [input-encoding controls](b2-flag-controls-20261001/index.md), [input replay](b2-input-replay-20261001/index.md).
- Nowcasting: [context-conditioned nowcaster](context-nowcast-20261001/index.md), [seasonal online nowcasting](seasonal-nowcast-20261001/index.md), [local reporting triangles](reporting-triangle-20261001/index.md).

## September 2026

- [B2 covariates, spatial sharing and masking with finalized T-0 inputs](b-2-t0/index.md), [covariate encoders and spatial sharing](forecast-geography-v2/index.md), [national Kinsa](kinsa-finalized-5seeds/index.md).
- [Archived experiments](archive/index.md), including B0.
