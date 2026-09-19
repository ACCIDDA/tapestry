# Is 2025–26 unusually difficult for coverage?

**Not clearly from the saved evidence.** Undercoverage predates this season.
The same target-MLP B1 direct-with-final-flags, no-mask recipe gives the following
mean coverage of nominal 95% intervals across seeds 42–44. These are the existing
retrospective CV scores, using the usual geographic weights:

| Target | Season | B1 coverage | Hub ensemble coverage |
| --- | --- | ---: | ---: |
| Flu admissions | 2023–24 | 85.1% | 92.2% |
| Flu admissions | 2024–25 | 82.9% | 81.4% |
| Flu admissions | 2025–26 | 81.6% | 89.8% |
| COVID admissions | 2024–25 | 80.5% | 95.6% |
| COVID admissions | 2025–26 | 78.6% | 94.2% |

The model deteriorates slightly for these comparable targets, but its intervals
were already substantially too narrow or misplaced. The ensemble's influenza
coverage was worse in 2024–25. We do not have matching older-season scores for
all six targets, so comparing the six-target 2025–26 composite with earlier
one-/two-target composites would confound season and target mix.

The forward pipeline's 75.4% overall coverage is also a different experiment from
B1: cutoff-pinned references, actual Wednesday older histories, no later-final
filling, fixed epochs and an exact-training/estimated-deployment input mismatch.
The difference cannot be attributed to season difficulty alone. Each historical
CV fold fits a different model, historical task windows vary, and early folds
can train on later seasons. This comparison is descriptive, not several strict
forward-season replications or a test of outbreak severity.

This is still a useful season for development, especially for revealing failure
under the current reporting regime. It is not an untouched final validation
season and does not establish performance in another season. There is no evidence
here for discarding it because its coverage is low.

[Underlying season comparison](coverage-across-seasons.csv). Reproduce without
fitting or rescoring predictions:

```bash
.venv/bin/python scripts/summarize_forward_coverage.py
```

## Practical next experiments for coverage

These are proposals, not implemented fixes or demonstrated gains.

1. **Fit a small bias-and-width recalibrator using out-of-time predictions.**
   Generate rolling-origin forecasts entirely inside the historical training
   partition, honoring each origin's actual vintages, label release dates and
   maturity rule. Use only residuals whose labels became available before the
   final training cutoff. Start with a target-level median adjustment and a
   positive interval-width multiplier in the model's transformed space. Pool
   locations; add horizon effects only if supported. Pool or shrink sparse groups
   rather than fitting separate calibrators to every geography. Use time blocks
   as the validation units: overlapping horizons and many states do not create
   independent weeks. Freeze the fitted mapping before evaluating the season.
   Fit/check both median bias and tail width: about 80% of the winning pipeline's
   RSV-admission WIS is underprediction penalty, so symmetric widening alone may
   be inefficient. Preserve quantile ordering and native bounds on inversion.
2. **Evaluate a predictive mixture of the fitted seeds.** Pool equally weighted
   samples from each fit and then extract mixture quantiles; do not confuse this
   with averaging seed scores or quantile-wise averaging. It can represent
   between-fit uncertainty but may not fix shared bias. This is a new development
   comparison, not a result already established by the current tables. Increasing
   Monte Carlo draws alone mostly reduces sampling noise; it does not resolve a
   systematically misspecified predictive distribution.
3. **Test the pipeline's deployment-input mismatch directly.** Train its
   forecast-only component on independently generated, out-of-time nowcast
   samples as an ablation alongside exact finalized inputs. A nowcaster must
   never supply in-sample fitted answers to the rows used to assess that change.
   Include actual missing-report cases. The independent nowcaster's poor
   reconstruction coverage makes uncertainty propagation alone insufficient.
4. **Select training duration on honest temporal validation.** Compare fixed
   100 epochs with a pre-cutoff, blocked validation rule. Existing B1 used selected
   epochs, so its numerical advantage does not demonstrate that more training or
   more masking is the answer. This is secondary to diagnosing bias, width and
   the input mismatch.

Use the proper WIS objective alongside 50/80/90/95% coverage, interval width and
under/overprediction components. Coverage alone can be improved trivially by
making forecasts uninformatively wide; WIS makes that trade-off explicit.
[Bracher et al., evaluating epidemic forecasts](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008618).

For a future deployment protocol allowing updates, adaptive conformal methods
are another candidate for changing error distributions, using only labels that
have actually arrived. That would be a separate **adaptive** experiment, not the
current frozen-parameter benchmark. Long-run coverage control is not a guarantee
of exact coverage within every disease, location or season; delayed revisions
also need explicit handling.
[Gibbs and Candès, adaptive conformal inference](https://arxiv.org/abs/2106.00170).

The sparse pre-cutoff current-regime history, especially four-day NSSP visible
reports, limits reliable estimation of tail corrections. Calibration cannot
promise next-season coverage under an unknown distribution shift. No 2025–26
labels should be used to tune this benchmark and then re-advertise it as a clean
test. Those observations can support training for a later genuinely forward
season, with that protocol and cutoff specified separately.

Recorded September 19, 2026.
