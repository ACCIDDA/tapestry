# Reporting regime visible on 7 October 2026

This audit uses the actual raw snapshots and Hub Git releases currently available
in the operational data root. It does not train on current-season future outcomes.
The first report is the value available at the FluSight Wednesday 23:00 Eastern
cutoff after each Saturday. The comparator is the report available four weeks
later, where that date has occurred. These later values are not asserted to be
final. Values are resolved by the same per-cell release rules as model inputs.

The newest flu ED Hub Git release is 7 October at 16:14:28 UTC (12:14 PM EDT),
covering the week ending 3 October. The Delphi snapshot collected earlier today
currently ends at the 30 September release, covering 26 September. Therefore the
newer Hub release supplies today's latest week. This observation does not establish
that Delphi always lags: acquisition times differ. The operational builder resolves
by publication time; it must not use only the older Delphi snapshot.

## Current revision pattern

For the six US early-season weeks in 2026 with four weeks of follow-up, the median
later/first ratio is 1.0263 for flu ED proportions and 1.0836 for admissions.
The corresponding 2025 early-season medians are about 1.0333 and 1.0654 (six ED
weeks and eight admissions weeks). This limited early-season evidence looks like
upward backfill with substantial geographic variation, not a demonstrated new
publisher schema or a fixed national correction factor.

Across current-season state/DC cells with four-week follow-up, 38.0% of ED values
revise upward and 3.0% downward; the remainder are unchanged at numerical tolerance
1e-8 in proportion units. The median ratio is 1.00, while its 90th percentile is
1.2385. National aggregation and state heterogeneity therefore give different
summaries. Ratios omit zero initial values. Counts use half-count tolerance to
avoid mistaking floating-point representation for revisions.

| US reference week | First ED proportion | Latest ED proportion | First admissions | Latest admissions |
|---|---:|---:|---:|---:|
| 5 September | 0.0019 (0.19%) | 0.0020 (0.20%) | 1,062 | 1,243 |
| 12 September | 0.0025 (0.25%) | 0.0026 (0.26%) | 1,503 | 1,540 |
| 19 September | 0.0031 (0.31%) | 0.0033 (0.33%) | 1,964 | 2,056 |
| 26 September | 0.0044 (0.44%) | 0.0046 (0.46%) | 2,515 | 2,621 |
| 3 October | 0.0058 (0.58%) | 0.0058 (0.58%) | 3,246 | 3,246 |

The newest week's equality means there has not yet been follow-up, not that it
needs no correction. Only the first row in this table has four-week follow-up.
Earlier 2026 weeks also enter the six-week summary. Comparable August/September
2024 first-release support is absent in this audit and is not imputed.

## Units and what the extension tests

The model's ED channel is a proportion in [0,1]. Delphi's percentages are divided
by 100; Hub proportions already use [0,1]. The existing residual tree supports
proportion-scale floors and bounded outputs, but its fitted channel list previously
excluded ED. The extension explicitly enables ED channels and fits them from
training-season report-to-mature pairs. It also tests ED revision augmentation in
training against the same augmentation plus correction. This avoids feeding a
corrector examples of noiseless finalized ED histories and calling them reported
training inputs. No blanket ratio from the table above is applied to forecasts.

The 2.6% aggregate ED revision does not predict the eventual forecast-score gain.
The model must show gains in matched held-out-season forecasts and inside the
ensemble. Early-season counts and proportions are small, and these six national
weeks cannot establish peak-season reporting behavior or a structural break.

Reproduce with `PYTHONPATH=src .venv/bin/python scripts/audit_b6_revisions.py`.
[Four-week summary](four-week-revision-summary.csv),
[ED cells](nssp_flu_proportion-revisions.csv),
[admissions cells](nhsn_flu_admissions-revisions.csv),
[ED source timestamps](nssp_flu_proportion-sources.csv).
