# B0.0 follow-up: adopted objective and B0.1 direction

Updated 2026-09-15. The objective below is adopted in code; architecture ideas
remain proposals. Evidence is the [post-scaling crosses report](index.md), saved
ranking CSVs, and focused reads/checks of the model and scoring implementation.
Historical reported scores retain their original objective. The subsequent
[B0.1 crosses specification](../../design/b0.1.md) is the active experiment
plan: 172 sample-based configurations, with raw/rate inputs and direct quantile
prediction excluded. Earlier exploratory suggestions below do not override it.

## Adopted loss and selection choices

The six targets are flu/COVID/RSV admissions and ED proportions. Peak and
rate-change targets are outside this objective. B0 continues to use finalized
six-channel data; B1 adds vintages and masking, and B3 adds wastewater.

- Target weights: **[1,1,1,.5,.5,.5]**, with no extra flu preference.
- Geography: **80% states/DC**, equally across eligible jurisdictions, and
  **20% native US**, which is predicted directly.
- **Seasons count equally.** Inside a season, normalize the target weights over
  the available challenges. Missing historical ED comparisons get no weight.
- CRPS and WIS score **native, untransformed predictions**: admission counts and
  0–1 ED proportions. Input fourth roots/logits are inverted first. Admission
  quantiles are rounded for WIS.
- Training divides each cell's native CRPS by its **training-only
  channel/location Q95**, then uses season/target/geography weights. This is a
  magnitude-normalized surrogate, not ensemble-relative skill.
- Selection divides **each location's model WIS sum by its ensemble WIS sum**,
  using identical tasks within that target and season. Average state ratios,
  combine with US, combine targets within season, then average seasons.

See [design choices for loss](../../design/architecture.md#82-design-choices-for-b0-loss-and-weights)
and [selection](../../design/architecture.md#103-metrics) for the complete contract.

### What the Q95 division means

If a state's training flu-admission Q95 is 200, CRPS 20 contributes .10. If
another state's Q95 is 2,000, CRPS 200 also contributes .10. The original data
and forecasts still have count units; only their score contributions are scaled.
This treats comparable fractions of typical burden similarly. It does not
establish equal skill relative to a benchmark: a state can have small typical
burden but be difficult to forecast.

Explicit engineering assumptions: with fewer than 26 observed unique weeks,
blend local Q95 with pooled channel Q95 using local weight n/26; impose floors
of 1 admission and .001 ED proportion. Fit scales within each allowed training
partition. Fixed cell weights preserve the objective across minibatches. Missing
geography groups renormalize to the available group; the full corpus ordinarily
has both. Nonpositive ensemble denominators raise an error when ranking, rather
than silently removing a jurisdiction or assigning an arbitrary ratio.

### Consequences of equal seasons

Each of the three scored seasons now contributes one third. The oldest season
has only flu admissions, the middle flu/COVID admissions, and the newest all six.
Thus flu still has more *effective historical weight* because it is observed in
more challenge seasons: 31/54 for flu admissions, 13/54 COVID admissions, 2/27 RSV
admissions, and 1/27 for each ED target. There is no added flu coefficient; this
is the direct consequence of the chosen season-first aggregation and available
support. Per-target means remain diagnostics, not a way to reconstruct the
combined score. No unavailable challenge is imputed.

## What to keep from crosses

Use the early-stopped anchor and conv as contrasting development references.
The user has excluded raw and rate-only admission representations completely
from B0.1, including controls. The retained transforms are square root, fourth
root, and log1p. All candidates still receive native-unit count/proportion loss.

The data constraint is independent epidemic seasons, not merely tensor cells:
157 weeks and 52 correlated locations provide about two fitting seasons per
fold. No more data or augmentation is part of the current code change. More
seeds reduce training noise, not uncertainty about a new epidemic season.

## Promising architectural biases, in priority order

1. **Stochastic local trend with damping and a small learned residual.** Estimate
   current level and recent growth from context; let each sampled path vary in
   level and slope. Damping shrinks extrapolated growth with horizon, and a
   small residual allows departures/turning points. This encodes smooth local
   progression without forcing a single seasonal peak.
2. **Shared temporal representation with small disease-specific heads.** Keep
   transfer across six channels while allowing COVID to behave differently from
   winter flu/RSV. This tests negative transfer; benefit is not established by
   the current sweep's state/US head experiment.
3. **Multi-scale temporal features or a small multi-scale convolution.** Combine
   a few-week trend with a broader 8–12-week background rather than assuming one
   temporal scale captures both rapid changes and seasonal context. Compare
   against the plain MLP before adding spatial complexity.

All are proposed inductive biases, not claims of measured improvement. First
change the objective and stopping, so architecture effects can be interpreted.

### Level and slope uncertainty: concrete meaning

An illustrative transformed-space path is

`u_m(h) = mean_path(h) + a_m + h*b_m`,

where a sampled `a_m` shifts the whole path up/down and `b_m` changes its growth
or decline. The actual positive/bounded decoder maps the path back to native
counts/proportions. A path can begin near its neighbors and diverge by week four.
The terms' amplitudes depend on context, disease, and location; their dependence
can retain shared epidemic uncertainty. A damped h term can prevent explosive
extrapolation. A small curvature/residual term can represent turning points.

For intuition only, at current level 100 the four future medians might follow
110,120,130,140. Level uncertainty alone could shift this to 100,110,120,130.
Slope uncertainty could instead produce 105,110,115,120 or 115,130,145,160.
These illustrative sequences are not fitted forecasts or a claim about count
noise. A level/slope model alone cannot predict an unseen rebound. Its purpose
is to represent uncertainty in future change more directly. Marginal CRPS still
does not identify the full joint distribution; check trajectory diagnostics.

## Prospective confidence, by target

No current B0 recipe has established a reliable prospective advantage over the
ensemble. The finalized-input advantage, season reuse in selection, calibration
shortfall, and limited independent seasons prevent that claim. Confidence is
not equally low across targets, and marginal skill need not imply good paths.

| Target | Assessment before new fits |
|---|---|
| Flu admissions | Not confident. Strong candidate performance with stopping, but pronounced season sensitivity; rank-1's flu average is actually above parity. |
| Flu ED | Cautiously promising, not established. Broad retrospective wins, but only one benchmark season. |
| COVID admissions | Cautiously promising. Two benchmark seasons and several useful candidates; prospective/revision robustness unestablished. |
| COVID ED | No confidence in beating the ensemble. Every recipe loses; investigate this separately before adding capacity. |
| RSV admissions | Strongest encouraging retrospective margin, but only one benchmark season; cannot call it a reliable prospective win. |
| RSV ED | Encouraging retrospective performance across recipes, but likewise only one benchmark season. |

Selected saved **old-objective** mean WIS ratios (not the newly adopted
location-relative scores), read from the crosses ranking CSVs:

| Target | Anchor, patience 20 | Conv | Conv, legacy decoder |
|---|---:|---:|---:|
| Flu admissions | .882 | .992 | 1.004 |
| COVID admissions | .876 | .926 | .841 |
| RSV admissions | .696 | .832 | .793 |
| Flu ED | .894 | .906 | .835 |
| COVID ED | 1.736 | 1.174 | 1.353 |
| RSV ED | .826 | .715 | .700 |

These are evidence of opportunities and tradeoffs, not calibrated probabilities
of winning next season. In particular, the safe flu stopping recipe is poor
for COVID ED. One checkpoint need not be used for all six submissions.

## Implementation changes and follow-up

- Fixed shared-factor validation noise by passing explicit frozen draws, just
  like global/local noise. Validation preserves the training RNG.
- Corrected comments/help attributing direct-US uncertainty to cancellation of
  summed state errors. No aggregation of states produces the US forecast.
- Added channel/location loss scales, global season/target/location cell
  weights, location-preserving WIS totals, season-first ranking, and explicit
  scoring-version metadata. The version enters ranking directory hashes.
- Old totals without location columns need rescoring from saved forecasts.
  Fresh training experiments are necessary to assess the changed loss. No
  historical performance numbers are relabeled as new-objective results.

Next: fit the two reference families with stopping under the adopted objective,
then compare modest calibration and seed mixtures. Score a mixture's actual
quantiles. Diagnose COVID ED and growth/turning-point coverage before expanding
architecture. Saved inner calibration forecasts come from a model refitted
later; calibration transfer remains an assumption to evaluate on outer folds.

## Transform shortlist and additional experiments

Assessment from the saved historical ranking, 2026-09-15. These numbers use the
old pooled/target-first objective and the 50-epoch reference recipes. They do
not establish winners under the newly adopted loss, scoring, or early stopping.
Entries are combined six-target scores, so changes can affect other channels
through the shared model. Lower is better.

| Admission transform | Anchor | Conv |
|---|---:|---:|
| Fourth root | .9701 | .9215 |
| Square root | .9635 | .9975 |
| Log1p | .9646 | .9613 |
| Rate | .9905 | 1.0042 |
| Raw counts | .9914 | 1.0159 |

| ED transform | Anchor | Conv |
|---|---:|---:|
| Linear | .9701 | 1.0238 |
| Logit | .9714 | .9215 |
| Fourth root | .9604 | 1.0374 |

Fourth-root admissions are the strongest default across the two retained
families and especially for flu: anchor flu-admission score is .9477 with
fourth root versus 1.0369 with square root and 1.0256 with log1p. Square root
and log1p have slightly better anchor combined scores, but differences near
.006 are not persuasive evidence of a winner with three seeds. Log1p is the
main alternative; it also leads the historical raw-family transform changes.

Logit ED is strongly favored within conv. There is no resolved anchor ED
winner: fourth root has the best point estimate, but its approximately .01
combined advantage is small. Its anchor COVID ED score improves from 1.5925 to
1.0782, while flu ED worsens from .9681 to 1.0183. Do not interpret that as a
universal ED improvement or as an isolated per-channel transformation effect.

The earlier small-run proposal (superseded by B0.1) was two existing families (anchor/conv)
by two admission transforms (fourth_root/log1p) by two ED transforms
(linear/logit): eight recipes. Add anchor with fourth-root ED and anchor with
square-root admissions as two targeted checks; include no raw/rate control under the subsequently adopted exclusion. Fix 12-week context, the adopted objective, and an initial
300-epoch cap/patience 20 policy. Three seeds are a screen; expand finalists
before resolving small differences. These are proposed configurations, not a
new implemented experiment-manager suite.

Additional retained comparisons are simple training-side calibration and
seed/architecture mixtures, scored on the mixture's actual quantiles. Direct
quantile prediction was proposed and subsequently rejected by the user; it is
not part of B0.1. The architecture modes and independent-model controls remain
specified rather than implemented. The active 100/300-cap, six-reference grid
is in the [B0.1 plan](../../design/b0.1.md).

## Log

- 2026-09-15: initial assessment separated data constraints, utility weights,
  normalization, and ranking; proposed a bounded B0 follow-up.
- 2026-09-15: user adopted weights 1/.5, national weight 20%, equal-relative
  jurisdiction evaluation, and season-first averaging. Implemented these
  choices and both requested noise/comment corrections. Replaced the earlier
  alternative-weight proposals with the adopted design; architecture changes
  and performance assessments remain hypotheses and retrospective evidence.

- 2026-09-15: read saved transform-specific ranking rows; recorded conditional
  transform shortlist and optional calibration, mixture, and direct-quantile
  experiments. No new training or benchmark claim.

- 2026-09-15: superseded the small-run recommendation with the architecture-focused
  B0.1 specification, and removed raw/rate control recommendations at user request.

- 2026-09-15: revised B0.1 to six reference crosses and focused interaction panels,
  removed direct quantile prediction, and retained 100/300 epoch caps with stopping.
