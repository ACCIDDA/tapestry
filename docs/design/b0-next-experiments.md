# Five next experiments for Build B

The [named B0 experiment manager](../workflows/experiment-manager.md) now implements
a fixed 14-configuration comparison covering the existing feature/loss switches,
separate state/US heads, two modulated decoder blocks, latent size, and temporal
convolution. Spatial attention remains in the later B1 round. The recommendations
and historical results below provide the context; the new suite has not been run
as a full performance comparison.

Recommendations after reviewing the design, implementation, and saved scoringutils
results. These are hypotheses to test, not demonstrated improvements. Retain the
accepted six finalized channels and no wastewater for this round.

Implementation status: experiments 1–3 are now configurable in B0's train and
season-CV commands; see [running instructions](../workflows/training.md#first-three-experiments-implemented-switches).
Defaults retain the historical baseline. The full staged comparison is now complete:
[14 runs / 42 season fits and three-seed finalists](../results/b0-full-experiments.md). Extra season-to-date summaries and valid
observation counts below remain optional later extensions.

## Diagnostic correction: national versus state performance

The earlier headline averages include US, whose much larger count scale can dominate
mean WIS. On exactly the same ensemble-supported task sets, the influenza breakdown is:

| Evaluation season | B0 states/DC WIS | Ensemble states/DC WIS | B0 US WIS | Ensemble US WIS |
|---|---:|---:|---:|---:|
| 2023-24 | 30.495 | 35.483 | 2654.303 | 1468.111 |
| 2024-25 | 74.269 | 99.044 | 4966.941 | 4182.154 |
| 2025-26 | 53.839 | 64.729 | 3699.718 | 2787.996 |

Thus state/DC WIS is approximately 14%, 25%, and 17% lower than the ensemble;
US WIS is worse in every season. These remain finalized-input CV comparisons,
including fitting on later seasons in the first two folds. Source: each
`data/evaluation/b0_hub_comparison/flusight_flu_hosp_SEASON/scores.csv`, grouped by
model and US versus states/DC. No forecasts were regenerated for this diagnosis.

## 1. Normalize count histories and represent national support explicitly

Implement the design's population-based admissions rates and compare square-root
with fourth-root input transforms. Fit scaling/centering only on the fit seasons.
Include log population and a native-state/native-US indicator in the shared model.
Decode back to native admission counts before fair CRPS and scoringutils evaluation.
ED retains its own proportional units and bounded decoder.

Currently, one channel-wide Q95 scales untransformed counts, and the same local
mapping handles states and US without geography metadata. This is a plausible
explanation for the national weakness, not proof of its cause. Test whether a small
US-specific residual adjustment is needed only after the representation change.
Do not replace native US predictions with summed states unless the observation
identity and geographic membership have been verified (design sections 6.1, 7.1,
10.4). Score states/DC and US separately so a national gain cannot hide a state loss.

## 2. Give the local encoder more temporal information

Compare 8, 12, and 26 weeks with the same small MLP. Then add a small set of causal
features: recent slope/change in slope, last-observation age, valid-observation
count, weeks relative to Christmas, and observed season-to-date summaries with
coverage. Do not include realized season peaks or maxima from future weeks.

The current model has eight raw weeks, annual sine/cosine, and a latest-observation
anchor. Longer history and causal change features may help distinguish rising,
turning, and declining waves. This is the inexpensive experiment specified by
sections 6.3 and 7; it does not require a temporal transformer or seasonal rollout.
Test lookback separately before adding the feature bundle.

## 3. Match task weights to the requested multi-pathogen objective

Current scaled fair-CRPS weights are `[1,.1,.1,.1,.1,.1]`: influenza admissions
receive ten times the explicit weight of either COVID or RSV admissions. That was
an intentional influenza-first skeleton choice. It is not a neutral six-target fit.

Compare it with `[1,1,1,.1,.1,.1]`, using the same training-only channel scales and
architecture. Also run a flu-only supervised-loss ablation while preserving the
six input channels, to distinguish cross-signal conditioning from auxiliary-task
transfer. Report all channels and retain an influenza-priority option if balancing
harms its performance. See design sections 6.2 and 8.2.

## 4. Improve the stochastic decoder and calibrate by task/horizon

The design proposes two residual decoder blocks modulated by a shared 32-dimensional
latent; the skeleton uses one modulation, a 16-dimensional latent, and a small
non-residual decoder. Test the designed head while preserving independent draws
and count-scale fair CRPS. Keep eight training draws initially; compare 16 only as
a separate numerical experiment. Three independent training seeds establish whether
observed differences survive initialization, and a mixture of complete members
can be evaluated subsequently.

Do not apply one global interval-widening factor. For states/DC, 2023-24 flu lead-1
95% coverage is about 81%, whereas 2025-26 RSV lead-4 coverage is about 99%.
Calibration errors depend on source and horizon. If needed, fit parsimonious
source/horizon corrections only on inner out-of-sample predictions, then evaluate
both WIS and coverage. Never calibrate on the held-out season or append independent
observation noise by default. See sections 7.2, 7.3, and 12, and the
[FGN paper](https://arxiv.org/abs/2506.10772) for the model-perturbation/CRPS approach.
Weather results do not establish that this epidemic modification will help.

## 5. Add the single spatial-attention block of B1

After the cheap representation and task-weight checks, add one four-head attention
block over locations, with geographic/support metadata and proper masks. Keep the
shared focal encoder, shared decoder, and one latent spanning all output tasks.
Use the design's width 128 as a small candidate rather than jumping to a large model.

Current B0 cannot condition one state's forecast on another state's current
trajectory. B1 tests whether contemporaneous spatial differences provide useful
context for wave timing and national forecasting. Spatial benefit is a hypothesis;
compare against the same model without attention. Geographic metadata must be
present so attention can distinguish locations/support rather than treating every
location as interchangeable. See sections 7 and 7.1.

## Experiment discipline

Start with 1, 2, and 3; then test 4 and 5 individually, followed by a combination of
changes that helped. Keep the existing ensemble-supported scoring sets, all 23
quantiles, and fixed truth snapshots. Use three seeds for finalists. Select stopping
rules and calibration using chronological blocks inside the two fitting seasons,
not the season being evaluated. Fit scalers again for each such inner fold.

All three seasons have already been examined, so subsequent comparisons are
exploratory development, not a new untouched final test. A prospective season is
needed for clean confirmation. Retain the FGN-style generator for these experiments;
wastewater, simulations, diffusion, and more source acquisitions remain outside
this accepted pilot round.
