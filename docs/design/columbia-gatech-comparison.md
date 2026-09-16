# Tapestry compared with Columbia and Georgia Tech

Review date: 2026-09-15. This extends [the FluSight metadata screen](flusight-prior-art.md).

## Scope and evidence

The two models are CU-ARNB_Net and Gatech-ensemble_prob. Tapestry means the
current implementation in `src/tapestry/models/b0.py` and `objective.py`;
B0.1 additions are identified as proposals. This is a methods comparison, not
an empirical ranking. No new training or forecast scoring was performed.

Columbia's [metadata](https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-metadata/CU-ARNB_Net.yml)
has placeholder website/repository URLs and no paper citation. Searches by model
name, method name, and authors did not locate a dedicated paper. Following the
[submission PR](https://github.com/cdcepi/FluSight-forecast-hub/pull/2519) led to
Rami Yaari's public repositories. The
[CU ensemble README](https://github.com/ramiyaari/Flusight-CU-Ensemble/blob/d7787c58b235d037f67e23e6ae32db7ac9751cc3/README.md)
explicitly leaves the neural component's details and code availability TBD.
The inspected source/notebooks do not expose an identifiable ARNB implementation.
The author's separate transformer repository cannot be assumed to implement ARNB.
Thus architecture depth, exact anchor, dispersion parameterization, sharing,
training loss and calibration remain unverified.

Georgia Tech's [FluSight repository](https://github.com/gatech-isye-yanglab/flucast/tree/9c1d5a1a92583637f3b953a56a9c811af4f819ec)
contains only a README at the inspected commit. It links ARM, CATS and ICTSP.
Their papers and available generic implementations inform the comparison, but
are not the unpublished FluSight adaptation. In particular, the adjusted WIS
formula, quantile constraints, exact channel layout and ensemble weights are
not recoverable from that repository.

## Core comparison

| Dimension | Columbia ARNB | Georgia Tech probabilistic ensemble | Tapestry B0 |
|---|---|---|---|
| Forecast representation | Negative-binomial distribution | Direct quantile heads | Samples generated from latent Gaussian draws |
| Anchor | Recent observations; predicted log-space changes | Generic CATS/ICTSP implementations use last-value centering; exact FluSight handling unknown | Last valid focal observation, with positive/bounded inverse mappings |
| Learning objective | Not documented | FluSight metadata specifies MSE for central output and adjusted WIS for other quantiles | Fair CRPS computed after native-unit inversion, then normalized and weighted |
| Cross-series learning | Not documented in sufficient detail | Central purpose of ARM/CATS/ICTSP; operational location/channel organization unknown | Shared focal encoder plus six-channel local context, optional spatial attention |
| Outcomes | Influenza admissions; ILI input | Influenza forecast with hospitalization, ILI and Google Trends inputs | Jointly supervised flu/COVID/RSV admissions and ED proportions |
| Joint uncertainty | Dependence model not documented | Marginal quantile outputs alone do not define a joint forecast | Member identity spans locations, channels and horizons through shared/global and optional local noise |
| Missingness | Unknown | Operational treatment unknown | Explicit input and supervision masks |
| Vintage handling | Unknown for ARNB specifically | Unknown for this adaptation | Current B0 uses finalized retrospective data |

The Columbia and Georgia Tech operational claims above come from their
[ARNB metadata](https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-metadata/CU-ARNB_Net.yml)
and [ensemble metadata](https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-metadata/Gatech-ensemble_prob.yml).
Unknown does not mean absent. Inverse-WIS weighting of CU's larger ensemble is
not evidence that ARNB itself is trained with WIS.

## What the Georgia Tech papers add

[CATS, methods section](https://arxiv.org/html/2403.01673v2) preserves an original-series
forecast path and adds a correction informed by constructed auxiliary series.
It controls that information through channel gating, adaptive temporal cutoffs,
and a continuity penalty. The generic method uses MSE plus continuity loss.
This is a close conceptual precedent for separating focal dynamics from shared
context. Tapestry currently merges those representations before decoding; it
has no explicitly separable forecast-level context correction or dedicated
context gate. A residual attention block is not the same construction.

The [CATS tutorial](https://github.com/LJC-FVNR/CATS/blob/2c32c604c1f976dc3c3945461eda7afe9a264616/CATS-Tutorial.ipynb)
confirms last-value preprocessing, a forecast shortcut in `decoder`, channel
selection, temporal gating, and a continuity term in `forward`.
These observations concern the public tutorial, not deployed FluSight code.

[ARM, methods section](https://arxiv.org/html/2310.09488v2) combines adaptive
univariate preparation, random channel dropping during training, and multiple
convolutional kernels with channel attention. It directly addresses differences
between series before learning their relationships. That overlaps conceptually
with Tapestry's normalization, focal/context split, and proposed multiscale and
head-sharing experiments. Its stochastic dropping is a training regularizer,
not Tapestry's stochastic forecast generator. The inspected v2 is dated February
2026; it does not establish every detail of the 2024–25 deployed version.

[ICTSP, section 2.2](https://arxiv.org/html/2405.14982v2) represents historical
lookback/future examples as tokens and lets the current prediction attend to
those completed examples. This differs from Tapestry's current one-context-token
per location. The historical examples' outcomes lie inside the observed context;
they are not future labels from the forecast being issued.
The generic [ICTSP implementation](https://github.com/LJC-FVNR/In-context-Time-Series-Predictor/blob/91da21052adaf8c993745d751250b94dc340ed61/ICTSP/models/ICTSP.py)
constructs these tokens, and its
[training code](https://github.com/LJC-FVNR/In-context-Time-Series-Predictor/blob/91da21052adaf8c993745d751250b94dc340ed61/ICTSP/exp/exp_main.py)
selects MSE. Those files do not implement the hub's described WIS quantile head.

## Interpretation for Tapestry

The following are comparative deductions, not measured performance claims.

1. **Anchoring is established prior art.** Columbia is a direct epidemic precedent.
   Tapestry's sample decoder has a different distributional representation, but
   anchoring itself is not a novelty claim.
2. **Learning cross-series corrections is also established.** Georgia Tech's work
   makes the focal-versus-context distinction particularly explicit. B0.1's
   attention and head-sharing experiments study a useful application and design
   choice, rather than introducing cross-series neural learning.
3. **Sample generation is a meaningful distinction.** Tapestry maps context and
   random draws to complete output arrays. It does not constrain each marginal to
   a negative-binomial family or learn only a finite quantile grid. This flexibility
   adds estimation demands and is not automatically beneficial with few seasons.
4. **Joint sample identity is not validated dependence.** Current fair CRPS sums
   marginal scores. Incorrect cross-location, cross-horizon or cross-pathogen
   dependence can survive an excellent marginal score. Four-week totals, changes
   and appropriate multivariate scores are needed for a joint-distribution claim.
5. **Multi-pathogen sharing remains the clearer application distinction.** Neither
   available operational description establishes the same six jointly supervised
   outcomes. This is evidence about documentation, not proof of first use.
6. **The score names alone are a weak novelty argument.** Both Tapestry and the
   Georgia Tech adaptation directly optimize probabilistic forecast accuracy;
   their sample-versus-quantile parameterizations and actual objectives should be
   compared empirically. Fair CRPS and finite-grid adjusted WIS are not identical.

## Most informative comparisons

Keep inputs, folds, backbone and evaluation support fixed wherever possible.

- **Implicit samples versus a negative-binomial head:** first compare on the same
  admissions task. An NB head cannot directly represent ED proportions. Keeping
  six outputs requires a separate bounded-output head; that extra choice must be
  reported. Label this an ARNB-inspired comparator, not a reproduction.
- **Implicit samples versus monotone quantiles:** use B0.1's planned same-input
  quantile comparator to test whether sampling improves marginal skill. This is
  not an exact reproduction of Georgia Tech's undocumented adjusted-WIS head.
- **Unrestricted context versus gated correction:** a modest CATS-inspired control
  can test whether suppressing unhelpful context improves on current additive
  feature fusion. This is a new optional experiment, not an approved grid change.
- **Sharing versus conditioning:** independent models retaining all six inputs
  isolate parameter sharing. Separate own-pathogen/own-target input ablations are
  required to isolate the value of other pathogens as contemporaneous covariates.
  The planned independent-model controls alone do not answer the latter question.

No existing experiment plan or model implementation is changed by this review.

## Log

- 2026-09-15: Read the two models' public descriptions, three Georgia Tech
  architecture papers, generic CATS/ICTSP source, and Columbia repositories.
  Recorded the limits of operational reproducibility and proposed matched
  comparisons without assuming undocumented losses or dependence structures.
