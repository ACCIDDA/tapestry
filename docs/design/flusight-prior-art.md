# FluSight precedents for Tapestry

Metadata review dated 2026-09-15.

## Scope and assumptions

“Approach” is interpreted as the current B0 model and proposed B0.1 extensions:
shared learning over influenza/COVID/RSV hospital admissions and ED proportions,
temporal encoding and optional spatial exchange, anchored stochastic residual
outputs, global/local latent noise, and native-unit fair-CRPS training. B0.1's
additional head-sharing, attention, trend-decoder and quantile comparisons are
proposals, not all implemented capabilities. Vintage-aware training belongs to
later stages; the current pilot uses finalized retrospective data.

Screened all 100 YAML metadata files returned by the current main-branch
[FluSight model-metadata directory](https://github.com/cdcepi/FluSight-forecast-hub/tree/main/model-metadata).
Read method/input descriptions for candidate matches. This is a metadata screen,
not an exhaustive code or paper audit, historical Git audit, or review of the
separate 2021–2023 archive. Main-branch descriptions can change. Listing a model
here establishes a declared approach, not forecast quality or an exact first-use
date. Absence from a short description does not establish absence in the model.

## Closest precedents

All model links below point to the screened primary metadata.

| Model | Documented overlap | Distinction or unresolved detail |
|---|---|---|
| [CU-ARNB_Net](https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-metadata/CU-ARNB_Net.yml) | Neural residual forecasts anchored to recent observations; log-space changes, lags, seasonal encodings, growth features | Negative-binomial output distribution; does not describe Tapestry's implicit sample generator or six-target sharing |
| [Gatech-ensemble_prob](https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-metadata/Gatech-ensemble_prob.yml) | Attention-based multivariate neural forecasting; quantile heads trained with adjusted WIS | Median uses MSE; direct quantile output rather than fair-CRPS sample training |
| [LosAlamos-ThinMint](https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-metadata/LosAlamos-ThinMint.yml) | Transformer on lagged context; hospitalization and ED inputs, plus synthetic data from historical ILI | Metadata does not specify pathogen sharing, loss, spatial exchange, or probabilistic head |
| [UMass-flusion](https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-metadata/UMass-flusion.yml) | Multiple surveillance sources; shared coefficients across locations; current description includes spatial GBQR | AR/gradient-boosting quantile ensemble, not a latent-noise neural generator; metadata distinguishes season-specific versions |
| [GT-FluFNP](https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-metadata/GT-FluFNP.yml) | Context-dependent integration of information and uncertainty from multiple views | Sparse metadata; strong candidate for paper/code follow-up, not evidence of the exact Tapestry training scheme |
| [UNC_IDD-InfluPaint](https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-metadata/UNC_IDD-InfluPaint.yml) | Generative epidemic trajectories with spatial structure; conditioning on observed history | DDPM plus COPaint inpainting, rather than direct conditional generation trained with fair CRPS |
| [JHUAPL-Morris](https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-metadata/JHUAPL-Morris.yaml) | Neural N-HiTS forecasts trained with WIS; temporal/state features | No declared six-channel sharing or implicit generator |
| [NAU-FourCAT](https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-metadata/NAU-FourCAT.yml) | Transformer, Fourier calendar features, multi-horizon quantile regression with pinball/WIS objective | Direct quantiles; no declared fair-CRPS sample training |
| [OHT_JHU-nbxd](https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-metadata/OHT_JHU-nbxd.yml) | Multichannel TCN encoder, residual N-BEATS decoder, ensembles over lookbacks and initializations | Gamma error distribution and likelihood loss |
| [UVAFluX-CESGCN](https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-metadata/UVAFluX-CESGCN.yml) | Neural information exchange across states | Graph derived from transfer entropy; not the proposed learned location-target attention |
| [UM-DeepOutbreak](https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-metadata/UM-DeepOutbreak.yml) | Recurrent/self-attention network with regions as learning tasks | Post-hoc adaptive conformal uncertainty; explicitly says spatial correlation is not considered |
| [CFA-PyRenew_HE](https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-metadata/CFA-PyRenew_HE.yml) | Joint fitting of NHSN admissions and NSSP ED signals | Renewal model for influenza; two signals do not establish joint multi-pathogen neural learning |
| [UI_CompEpi-EpiGen](https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-metadata/UI_CompEpi-EpiGen.yml) | Deep/statistical ensemble trained with historical and VAE-generated synthetic data | Synthetic augmentation is not the same as using an implicit generator as the forecasting head |
| [DMAPRIME-HYB](https://github.com/cdcepi/FluSight-forecast-hub/blob/main/model-metadata/DMAPRIME-HYB.yml) | Multiscale CNN, recurrent encoder, temporal attention, quantile output, EHR covariates | South Carolina forecasting; no declared national six-target sample generator |

## Interpretation

Many ingredients have already been tried in FluSight. Neither neural forecasting,
attention, multichannel inputs, anchored residual outputs, generative trajectories,
nor direct probabilistic-score training is a defensible standalone novelty claim.

No screened YAML contains “CRPS” or “continuous ranked”. No screened description
explicitly identifies the entire combination of six jointly learned pathogen/outcome
channels, global/local latent-noise generation, and native-unit fair-CRPS training.
This supports a narrower candidate distinction, not a claim to be first.

The useful research claim would be empirical: whether cross-pathogen sharing,
cross-signal conditioning, spatial exchange, and direct sample training improve
forecast skill in controlled comparisons. Marginal CRPS/WIS cannot on their own
establish correctly learned joint dependence.

Prioritize CU-ARNB_Net for the residual decoder, Gatech-ensemble_prob and
JHUAPL-Morris for score-trained neural comparators, ThinMint for the input/encoder
combination, and GT-FluFNP for probabilistic multi-view architecture follow-up.
Flusion and InfluPaint remain the closest already-documented conceptual anchors.

## Log

- 2026-09-15: Added metadata-based precedent screen at the user's request; recorded
  interpretation of “approach”, scope limits, and distinction between overlap and
  an exact architectural match. No model code or experiments changed.

Follow-up: [Detailed Columbia and Georgia Tech comparison](columbia-gatech-comparison.md)
reads the linked architecture papers and public source, and distinguishes generic
architectures from unavailable FluSight-specific implementations.
