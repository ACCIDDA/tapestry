# First good B1 results

2026-09-17. A full-budget seed42 comparison used the existing dataset, all three
season folds, forecast-loss epoch selection followed by refitting, and 2,048
predictive trajectories. A reused a verified completed run; B and C ran on GPUs.

## Forecast performance

Scores use the shared frozen-support scorer: location-relative WIS against the
Hub ensemble, with the established geography/target weights and equal seasons.
The ensemble benchmark is 1.0; lower is better.

| Candidate | Natural score | Improvement over ensemble | Improvement over A |
|---|---:|---:|---:|
| A: rank-1 direct, masking 0.5 | 0.989394 | 1.06% | — |
| B: A + supplied-final flags | **0.941934** | **5.81%** | **4.80%** |
| C: B + parallel nowcast head | 0.958547 | 4.15% | 3.12% |

C uses forecast loss + 0.25 × nowcast loss, without feeding nowcast predictions
into forecasting. B beats the ensemble on all six target scores averaged across
seasons. Relative to A, it improves five targets; influenza admissions worsen
by about 2.05%.

| Input condition | A | B | C |
|---|---:|---:|---:|
| Natural | 0.989394 | **0.941934** | 0.958547 |
| Recent masking | 1.121713 | **1.078939** | 1.080418 |
| Gap | 0.989807 | **0.942306** | 0.958680 |
| Outage | 1.343990 | 1.316628 | **1.296242** |

Stress scores retain the same frozen tasks and ensemble benchmark; only the
model's inputs are stressed. B improves every stress point score relative to A.

## How much of the input is vintage data?

On the actual ensemble-scored support, deduplicating horizons and counting each
target/location/issuance once, **13,920 of 14,497 combinations (96.0%)** have
vintage reports for both recent weeks. **657 of 28,994 recent input cells (2.3%)**
use the supplied-final fallback. These are unweighted input counts, not shares
of the weighted score.

The audit joins `forecast-cells-natural.parquet` from B's seed42 attempt002 to
`build_b1_wednesday_calendar.npz`: issuance is Hub reference date minus three
days, locations map from FIPS to postal codes, and the two most recent cells
are classified using `X_available` and `X_final`.

Thus the comparison mostly uses vintage recent inputs. Older history is
finalized throughout, and missing recent reports can receive later finals.
Scoring uses the same finalized observations for the model and ensemble.
This is retrospective conditional forecasting, not a strict real-time backtest.
Hub forecast availability does not guarantee complete vintage input coverage.

## Calibration and uncertainty

Under the score's geography/target weighting and equal-season averaging:

| Candidate | 50% interval coverage | 95% interval coverage |
|---|---:|---:|
| A | 45.37% | 87.63% |
| B | 43.97% | 86.75% |
| C | 41.61% | 85.80% |

All three under-cover. B's lower WIS does not imply better calibration.

The primary paired 8-week temporal interval for B/A relative WIS change is
[-10.52%, +0.09%]; C/A is [-6.49%, +3.72%]. Both include zero. These intervals
condition on seed42's fitted models and on retaining scoring support: 827 of
2,000 block draws were excluded for missing positive-denominator support.
Four-/12-week sensitivity intervals favor B over A but do not replace the
preselected primary interval. There is no across-seed uncertainty estimate or
ten-fit predictive mixture yet, and these intervals do not test superiority
over the ensemble.

**Recommendation: continue iterating on B.** It has the best natural score and
is simpler than C. Retain A under the formal promotion rule: B's 4.80% gain
falls short of the fixed 5% threshold, the primary interval includes zero, and
the ten-seed experiment remains deferred.

## Execution and next settings

B job1491695 completed in 25:17; C job1491696 in 34:37; report job1491906 in
1:00. Completion notifications were enabled. B spent roughly half its time
fitting/selecting and half evaluating; final shared scoring took about 16 seconds.
The long daytime runs were cancelled and were not restarted overnight.

New code defaults to **256 evaluation trajectories**, with training and
validation budgets unchanged. This trades evaluation precision for faster
iteration. The results above and the old pinned plans remain at 2,048; use new
256-draw plans and reevaluate reused checkpoints consistently before combining
results. Mixed evaluation budgets are rejected by the decisive report.

Artifacts: `data/experiments/B1-seed42-report/` contains the score tables,
paired differences, calibration, temporal sensitivity, recent-head diagnostics,
and `seed-stress.png`. Exact original plan/launch/status/rank commands and job
IDs are recorded in [the execution note](../workflows/b1-one-seed.md).
