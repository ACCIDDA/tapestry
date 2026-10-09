# baselinenowcast comparator

Every new `task=finalize`, `task=nowcast`, and pipeline nowcast-stage evaluation
includes an independent `baselinenowcast` **point** comparator. The goal is to
improve on this reference, not only on unchanged reports. This does not implement
the package's uncertainty estimation or sampling; beating the point comparator
does not establish superiority to its full probabilistic forecasts.

The equations follow upstream [estimate_delay.R](https://github.com/epinowcast/baselinenowcast/blob/e7eb9749622c732e0277349738b5c79141afa4b4/R/estimate_delay.R)
and [apply_delay.R](https://github.com/epinowcast/baselinenowcast/blob/e7eb9749622c732e0277349738b5c79141afa4b4/R/apply_delay.R).
The Python implementation independently computes incremental reports from archived
cumulative vintages, completes the triangle column by column using ratios of
summed increments to earlier exposure, and computes a delay CDF from the completed
column totals. It has no spatial pooling, identity prior, robust median fitting,
or learned correction-strength selection.

## Assumptions and adaptations

- Fit separately by signal, location and issuance using the latest 52 event weeks
  whose scheduled initial report date has passed. Delay 12 approximates maturity;
  these settings are fixed, not selected on evaluation scores. The window differs
  from Chromantis's 52 eligible pairs per delay.
- Only scheduled vintages available at issuance enter the fit, including causally
  available reports during held-out seasons. Reference truth never enters delay
  estimation. The current event week's visible reports may enter the triangle.
- Missing archived vintages are not zeros. Exclude an event row if any already-due
  vintage is missing or negative. Require at least one fully observed row and a
  finite positive delay CDF. Signed revision increments remain signed; a CDF above
  one permits downward corrections. These are low-level signed-revision equations,
  not a claim of matching all upstream preprocessing defaults.
- Counts use upstream's exposure denominator floor of one and successive increment expectation
  `(current cumulative + 1 - previous CDF) / previous CDF * next PMF`.
  Add each expected increment before processing the next delay. Continuous signals use a positive exposure
  denominator without that floor and `report / CDF(age)`, preserving invariance
  to rate units. This continuous extension is an adaptation, not upstream count
  semantics. Clip estimates to nonnegative and source-specific physical bounds.
  Preserve unrounded count expectations.
- At age 12 or older retain the visible report. Missing current reports or
  unsupported triangles retain the persistence anchor. Finalization uses its
  existing last-observed-in-12-weeks / training-median fallback. Target nowcasts
  use the latest visible value at or before the target week within 12 weeks,
  or zero if none exists. No future event week or final truth fills these gaps.
  Status and retained triangle-row counts accompany predictions; fallback cells
  remain in scores rather than silently changing evaluation support.

## Outputs and interpretation

Finalization saves `baselinenowcast`, `baselinenowcast_status`, and
`baselinenowcast_support` alongside each prediction in `finalizations.csv.gz`.
The method appears in all finalization metric/ranking tables and age-performance
plots; `baselinenowcast_scores.json` also records fallback coverage.

Target nowcasts save `baselinenowcast.npz` with point predictions, truth, masks,
dates, location order, status and support. `nowcast_scores.json` and
`nowcast-ranking.csv` include its normalized MAE and
`baselinenowcast_point_crps_ratio`: model fair CRPS divided by baseline point MAE
(less than one favors the model). MAE is the CRPS of a deterministic distribution;
both use the model's training-only Q95 scales and identical scientific cell
weights. A zero baseline denominator produces an undefined ratio. Pipeline files
live under `nowcast/`; forecasting continues to use its existing evaluation.

Existing attempts lacking baseline artifacts are incomplete under the updated
manager. Add the comparator without refitting using
`python -m tapestry.experiment.planner baseline -e NAME`, then run `status` and
`rank`. The command checks pinned inputs, aligns the saved scored cells, and
records separate baseline provenance without changing the original fit manifest.
For new fits, re-plan to refresh pinned source before `run`.

## Decision log

- 2026-10-01: Added the independent comparator to both per-signal finalization and
  neural nowcasts, including the pipeline stage, at the user's request. Implemented
  the point estimator first and explicitly distinguished continuous-signal
  adaptations and deterministic CRPS from the upstream uncertainty model.

- 2026-10-01 audit: fixed a count-expectation shortcut that did not reproduce
  upstream sequential application. Version `baselinenowcast_point_v2` matches
  the pinned R low-level functions on 27 real NHSN triangles (flu/COVID/RSV,
  NC/CA/US, three issuances), with maximum CDF error below 1e-15 and point
  error below 3e-12 admissions. Before correction the largest discrepancy was
  0.13 admissions. Regenerated the stored baseline.
- The configured comparator is not the upstream high-level default: upstream
  redistributes negative increments by default, and its default history allocation
  usually uses 18 weekly rows at maximum delay 12 with sufficient history, versus
  our 52-row signed-revision fit. Neither configuration has been selected using
  pre-evaluation validation. Continuous signals remain an explicit adaptation.
  Do not interpret this comparison as beating the default package.
