# B1 seed42 screen — 2026-09-17

All three candidates completed with full epoch selection/refitting, three folds,
matched stress masks, and 2,048 evaluation trajectories. A reused its verified
completed checkpoint. B job1491695 took 25:17, C job1491696 took 34:37, and the
shared comparison job1491906 took 1:00; all completed successfully. Slurm
completion notifications were enabled. No CPU training job ran.

Scores are the shared frozen-support ensemble-relative forecast WIS; lower is
better. Natural totals reproduce the original ranking calculation.

| Condition | A: direct | B: supplied-final flags | C: parallel auxiliary head |
|---|---:|---:|---:|
| Natural | 0.989394 | 0.941934 | 0.958547 |
| Recent | 1.121713 | 1.078939 | 1.080418 |
| Gap | 0.989807 | 0.942306 | 0.958680 |
| Outage | 1.343990 | 1.316628 | 1.296242 |

B improves natural WIS by 4.80% relative to A; C improves by 3.12%. B improves
all three stress point scores. C is best under outage (3.55% better than A),
but is 1.76% worse than B on natural inputs. B improves five of six target
scores; influenza admissions worsen by about 2.05%. C's strongest admission
result is RSV, but its RSV ED score is worse than both A and B.

Recommend **B for continued iteration**, while retaining **A under the formal
promotion rule**: B falls short of the fixed 5% natural improvement threshold,
and the primary paired 8-week temporal interval for B/A relative change is
[-10.52%, +0.09%]. C/A is [-6.49%, +3.72%]. Both include zero. These intervals
condition on the fitted seed42 models; they cannot estimate seed uncertainty.
The primary calculation excludes 827 of 2,000 resamples that lose required
positive denominator support, so the intervals are conditional on retaining
support. Four-/12-week sensitivity intervals for B/A exclude zero, emphasizing
sensitivity rather than changing the preselected primary rule. There is no
across-seed estimate or ten-fit predictive mixture from this screen.

Calibration remains weak. Using the score's target weights within each season,
equal seasons, and the shared all-geography coverage aggregation:

| Candidate | 50% interval coverage | 95% interval coverage |
|---|---:|---:|
| A | 45.37% | 87.63% |
| B | 43.97% | 86.75% |
| C | 41.61% | 85.80% |

These are retrospective supplied-final experiments, not an operational
as-of backtest. Lower WIS does not establish calibrated intervals.

Artifact directory: `data/experiments/B1-seed42-report/`, including paired
contrasts, target/season/geography calibration, block-length sensitivities,
recent-head diagnostics, and `seed-stress.png`.

## Evaluation budget and next execution

New B0/B1 manager plans and standalone fits now default to **256 evaluation
trajectories**; training/validation draws remain unchanged. Mixtures use the
actual saved draw count (2,560 total for ten fits evaluated at 256). Historical
reports read the count from the attempt, and comparisons reject mixed counts.
Fewer evaluation trajectories increase Monte Carlo error, especially in tail
quantiles; this is an explicit speed/precision choice, not a training change.

Completed results, original experiment plans, and their pinned source snapshots
remain historical 2,048-draw artifacts. They have not been relabeled or silently
modified. A new 256-draw comparison requires new plans and reevaluation of reused
checkpoints at the same count. Do not launch the old overnight resume commands
expecting 256. No new training or scoring jobs were launched for this update.
