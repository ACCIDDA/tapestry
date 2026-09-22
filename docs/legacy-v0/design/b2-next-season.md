# B2 interpretation and next-season training

Research note, September 21, 2026. This is a proposed comparison, not an
implemented training change or a new launch. Evidence comes from the completed
[screen](../results/B2-screen/index.md), [Kinsa experiment](../results/B2-kinsa/index.md),
and current B2 input and fold code.

## What the covariate experiments establish

1. **Availability prevents an operational Kinsa or wastewater test.** Kinsa's
   PopHIVE archive begins April 6, 2026; NWSS vintages begin February 25, 2026.
   Neither has usable fitting and evaluation support within the same Wednesday
   fold. Biological lead, historical event dates, and archived release dates
   are different quantities. These archive starts do not establish when the
   original provider could first have supplied a signal.
2. **Kinsa is a weakly identified comparison.** Finalized aggregate WIS changes
   are +4.15% for the target recipe and -2.69% for the pathogen recipe. Inactive
   Wednesday Kinsa changes scores by +3.28% and +5.73%; adding the channel also
   changes initialization. These are a few contrasts, not a confidence interval
   or an estimated noise distribution. Kinsa enters only the national location;
   its aggregate improvement in the pathogen recipe comes from states, while
   national WIS worsens 2.08%. This is not persuasive evidence of an early signal
   being used successfully. A national symptom mixture also need not identify
   pathogen-specific hospital demand beyond the existing ED/admission histories.
3. **Claims have support, so lack of overlap is not a universal explanation.**
   Claims cover 206 of 208 channel/location pairs on both sides of every fold.
   Redundancy with the six existing histories and limited epidemic diversity
   are plausible explanations. Only about 80–85 inner fitting origins remain
   per fold; overlapping histories and correlated states are not independent
   seasons. These hypotheses have not been isolated experimentally.
4. **The inputs are already encoded, but their representation can improve.**
   B2 centers/scales covariates per fitting component and location, supplies
   availability masks, and concatenates their 12-week histories into its context
   MLP. It has no dedicated covariate denoising objective or trend summary.
   Standardization is not denoising. Test a small feature set (current level,
   trailing mean, recent slope, age of latest report) or a regularized residual
   correction before a larger encoder. Use trailing operations only; smoothing
   can erase lead. Log transforms for skewed positive wastewater levels are
   candidates, not a general prescription for bounded shares or percentile ranks.

## Proposed next comparison

Use the existing [forward benchmark](../workflows/forward-2025.md) as the basis
for chronological development. Its current parameters stay frozen throughout
the evaluation season; the proposed experiment would explicitly compare frozen
fits with periodic updates. The three-season B2 leave-one-season-out results
include fitting on later seasons when evaluating older ones and do not reproduce
the deployment sequence.

- Compare all prior eligible history, a predeclared recent-season weighting,
  and only the last season, on identical forward origins and targets. Last-season
  only is a data-scarcity baseline, not an assumed winner. Weighting values and
  update frequency must be fixed before running the comparison.
- At each update, admit only labels actually available by that issuance cutoff,
  with an explicit maturity policy. Fit normalization and any calibration on the
  same allowed data. Do not train against eventual final labels before release.
  Use chronological internal validation with label windows ending before the
  validation period; legitimate past context may overlap.
- First establish the no-covariate regimen. Then add one source at a time with
  an identical-width masked control, matched shared initialization and multiple
  seeds. Report native-US Kinsa effects separately and examine prespecified
  rising-phase and horizon contrasts alongside overall WIS and coverage.
- Retain older data as a candidate source of shared temporal structure, while
  adapting a small modern-target mapping on recent overlap. The
  [historical-curve proposal](historical-curve-transfer.md) describes source-aware
  auxiliary training. Historical ILI or Kinsa is not an admission label.

Assumptions: the intended next deployment is 2026–27; source archives remain
available; recent relationships may transfer better, but that is a hypothesis.
Kinsa's first archived vintage misses most of the preceding respiratory winter.
Last-season weighting alone cannot create that missing historical availability.
Backfilled finalized Kinsa may support a separately labeled retrospective
relationship study, not a certified historical Wednesday test. National Kinsa
remains national; broadcasting it as contextual information to state forecasts
would be a new modeling assumption and needs its own comparison.

The already explored 2025–26 season is development evidence. Lock the selected
protocol before genuinely prospective 2026–27 forecasts. A last-season-only
design still has many pooled locations but only one seasonal trajectory per
location, so it favors simple, regularized adaptation.

## Connection to the paper

[Martinson et al. (2026)](https://arxiv.org/html/2605.16238v1) evaluate forecasts
prospectively with weekly data updates. Their scarce-data RSV models use
gradient-boosted pipelines and cross-pathogen auxiliary inputs; influenza also
uses longer ILINet history. The study supports testing simple models and
auxiliary information under limited target history. It does not establish
last-season-only training as optimal. Weekly input updates should not be assumed
to imply one universal parameter-refitting schedule across their models.

## Decision log

September 21, 2026: prioritize availability-aware chronological training and
matched ablations over enlarging the covariate encoder. No model changes or jobs
were made for this note. Local and Longleaf work were committed and reconciled;
the source files common to both working trees were identical before merging.
