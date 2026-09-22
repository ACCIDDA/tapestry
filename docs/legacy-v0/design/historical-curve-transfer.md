# Historical surveillance transfer into B1

Concept proposal, September 20, 2026. Not implemented or experimentally validated.

## Scope and assumptions

“B-1” is interpreted as the existing B1 model family. The intended inductive biases are shared temporal dynamics, forecasts anchored to recent activity, explicit source/missingness semantics, and preservation of observed spatial and cross-channel alignment. The user has not specified which historical datasets will be added. ILI, ILI+, and FluSurv are examples, not an audited inventory or authorization to change model inputs.

Scale differences are not the only problem. ILI percentages, catchment hospitalization rates and NHSN admission counts measure different events, populations and delays. Multiplicative normalization cannot establish their interchangeability. The transfer hypothesis is that some temporal patterns are useful across surveillance systems after conditioning on source identity; amplitudes, delays and uncertainty can remain source-dependent.

## Minimal proposed experiment

Keep the modern B1 forecast task and its native outputs. Add a historical self-forecast task: a source predicts its own next four observations from its past, using an auxiliary source-specific input adapter and output head while sharing selected temporal encoder parameters with the modern model. Do not relabel historical ILI as historical NHSN admissions or occupy an existing admission/ED slot without an explicit source distinction.

Start with flu and the chosen existing B1 backbone. Independent target fits do not automatically share parameters; the proposal requires explicit sharing within the affected fit or a documented pretraining transfer. Historical examples update shared temporal parameters and their own head. They do not update the modern revision head, modern observation-specific output parameters or unobserved cross-pathogen relationships. Modern examples continue to train the complete modern forecasting path.

An initial representation is u(t)=x(t)/a, with positive a estimated exclusively from visible context or an allowed fitting partition. Retain a, source identity, units/population or denominator information where available, and calendar information as conditioning metadata. A forecast for the source can be decoded as x_hat(t+h)=a*u_hat(t+h), retaining a residual anchor within the existing working-space formulation. This is an invertible coordinate change conditional on a, not an assertion of exact model scale equivariance. Floors, nonlinear transformations and count-dependent noise require separate treatment. The choice of scale statistic and floor must be specified before a run; no numerical values are assumed here.

Per-location normalization must retain level/scale metadata if spatial mixing is expected to use relative amplitudes. Never normalize by the future peak or whole current season. Do not independently shift, stretch or splice locations into a fictitious spatial epidemic. Regional and national historical sources should first train a temporal-only task at native geography, without duplication into state observations or spatial losses for nonexistent state measurements.

Fit modern native-unit probabilistic losses as currently specified. Historical losses forecast each historical source in its own units with fitting-only source normalization. Normalize modern and auxiliary task aggregates separately before applying an explicit auxiliary weight, so additional historical rows do not silently redefine the modern scientific objective. A larger historical sample count does not justify larger task weight. Select the weight and stopping point on modern-target validation only.

## What B1 needs to establish first

- Usable predictions when related channels are unavailable, with availability and supplied-final flags treated correctly.
- A stable modern forecast baseline and an identifiable temporal encoder that can receive auxiliary gradients.
- Explicit source identifiers and reversible scaling, preserving levels and observed geography.
- Separate auxiliary labels and losses, with no artificial reporting-revision targets on finalized historical curves.

Compare the modern-only control, auxiliary historical training, and the same auxiliary task restricted to modern years to distinguish additional historical information from an auxiliary-supervision effect. Keep a modern-only route available to detect negative transfer. Evaluate modern forecasts, calibration and missing-channel conditions on identical support. Existing reused seasons remain development evidence rather than untouched evaluation.

Historical data can support hypotheses about temporal shape. It cannot by itself identify the modern ILI-to-admission relationship, modern ascertainment, hospital severity, reporting revisions, or dependence with absent COVID/RSV histories. Those remain learned from modern overlap or explicitly modeled with additional assumptions. Shared sampled noise and marginal CRPS also do not establish correct joint dependence.

## Research log

September 20, 2026: proposed source-aware auxiliary historical forecasting rather than synthetic relabeling of historical curves. Preserve native B1 objectives and distinguish transferable temporal structure from observation-system calibration. No code, datasets, jobs or existing experimental decisions changed.
