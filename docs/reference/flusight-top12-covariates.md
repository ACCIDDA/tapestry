# Inputs and covariates of the top twelve FluSight models

September 28, 2026. Scope: first twelve rows in the user-supplied 2025–2026 evaluation, including tied scores. These are documented inputs, not a reconstruction of every weekly deployment. Historical augmentation, contemporaneous predictors, derived features and incoming model forecasts have different roles.

Season-end metadata is pinned to [FluSight commit fd6cfaa](https://github.com/cdcepi/FluSight-forecast-hub/tree/fd6cfaabe10c147d0268c2b34087e0aa036df357/model-metadata). Code and paper details for the first eight are linked in [the model review](flusight-2025-2026-model-review.md). Features from older papers or optional code paths are qualified rather than assumed deployed in every 2025–2026 forecast.

| Order | Model | Documented raw inputs or upstream forecasts | Derived features and role |
|---|---|---|---|
| 1 | Google_SAI-FluEns | NHSN influenza admissions, historical ILINet, population | Component-dependent lags, seasonality, location indicators, normalized rates; some components map historical ILI into synthetic admission history. No Google search input listed. |
| 2 | OHT_JHU-nbxd | Admissions, dew point, NREVSS percent positive | Public code also includes temperature, outpatient surveillance and days since July 1. These are code defaults, not verified weekly production settings. |
| 3 | CMU-TimeSeries | Admissions, historical FluSurv and ILI; season-end code includes NSSP flu ED data | Population-normalized lags, seasonal windows, standardized historical training examples, climatological/linear components. |
| 4 | UGA_flucast-INFLAenza | Admissions, population, geographic adjacency | Inspected season code has state effects, cyclic seasonal effects, Christmas-relative effects, early-COVID indicator, common temporal and spatial-temporal latent effects. ED is separately forecast; this does not establish ED conditioning of admissions. |
| 5 | UMass-flusion | NHSN, FluSurv-NET, ILINet | Published earlier version uses population, source/location identifiers, season week, Christmas distance, lags, rolling means, slope/curvature. Version 1.1 adds spatial modeling; exact features should be checked against that implementation. Earlier ILI+ preprocessing includes test positivity. |
| 6 | NAU-vulPES | Component forecasts with NHSN, NSSP, HHS, ILIp and FluSurv inputs | Inherited through members; exact membership and feature allocation not established. |
| 7 | FluSight-ensemble | Eligible submitted quantiles | No single direct covariate vector; source information is inherited from contributors. |
| 8 | UVAFluX-FS_OptimWISE | Hub quantiles and observed admissions | Recent forecast performance, location and horizon determine regularized ensemble weights. |
| 9 | MIGHTE-Joint | Admissions with imputation, Google Trends | Weighted LightGBM/ARIMA/ARGO2 combination; metadata does not list exact search terms or lags. |
| 10 | MIGHTE-Nsemble v2 | Admissions; historical ILI mapped into admission history | Self and other-state lags; lags 1–12; rolling mean, SD, min/max over 2–12 weeks; differences and percent change; sine/cosine week/month; recent component WIS for adaptive weighting. |
| 11 | NEU_ISI-AdaptiveEnsemble | Admissions and Flu Scenario Modeling Hub projections across scenarios | Observed trends reject poorly matching simulated trajectories. Scenario assumptions enter indirectly through projections. |
| 12 | UGA_flucast-Scenariocast | NHSN admissions and Scenario Modeling Hub projections | Growth-rate patterns select nearest-neighbor projected trajectories. |

## Interpretation and limitations

### Training history versus forecast-time conditioning

Here, training history means examples used to estimate parameters or construct a historical trajectory library. Forecast-time conditioning means the recent observations, known calendar/static features or incoming forecasts used to make the current prediction. Weekly refitting can use both: this distinction is about statistical role, not whether a file is read during a weekly run. All recent observations mean the latest available reports, not knowledge of the future.

| Model | Training or reference history | Current forecast conditioning |
|---|---|---|
| Google_SAI-FluEns | Admissions and historical ILI augmentation | Recent admissions, calendar, population/location; current ILI not established across the ensemble |
| OHT_JHU-nbxd | Historical target and covariate sequences; optional historical hospitalization pretraining path | Recent admissions, dew point and test positivity; temperature/outpatient/calendar in public code defaults |
| CMU-TimeSeries | Admissions plus standardized historical ILI and FluSurv examples | Recent admissions plus NSSP ED in the auxiliary seasonal component; population/calendar |
| UGA INFLAenza | Past admissions estimate seasonal, spatial and temporal effects | Latest admissions update temporal/spatial state; known calendar, geography and population |
| UMass-flusion | Admissions plus ILI/FluSurv training examples | Published model conditions hospitalization forecasts on recent hospitalization features and known metadata; v1.1's exact additional spatial/current-source conditioning remains to be verified |
| NAU-vulPES | Inherited from constituent models | Current constituent forecasts; no verified target-specific raw input allocation |
| FluSight-ensemble | No fitted weights for its standard median aggregation | Current eligible quantile submissions |
| UVA FS_OptimWISE | Recent component forecasts and corresponding observed outcomes fit weights | Current component forecasts, location/horizon and fitted weights |
| MIGHTE-Joint | Historical admissions/imputation and component-specific predictor histories | Recent admissions and Google Trends; exact terms/lags unspecified |
| MIGHTE-Nsemble v2 | Admissions augmented with ILI-derived hospitalization history | Recent own/other-state admissions and calendar; recent errors update weights; live ILI not established |
| NEU_ISI-AdaptiveEnsemble | Scenario Modeling Hub trajectory library | Recent admissions select compatible trajectories |
| UGA Scenariocast | Scenario Modeling Hub trajectory library | Recent observed admission growth selects nearest trajectories |

Google's [published problem specification](https://github.com/google-research/google-research/blob/master/epi_forecasts/problem_statements/flu_problem_statement.txt) restricts its supplied historical ILINet tables to dates before October 15, 2022. This establishes historical augmentation in that task, not a live ILI feed. Broader paper prose does not establish an identical input policy for every deployed component. Flusion's [published formulation](https://arxiv.org/html/2407.19054v1) pools source-specific forecasting examples: fitting on multiple sources must not be interpreted automatically as conditioning each NHSN forecast on current values from every source. Its newer operational version requires separate verification.

Historical ILI and FluSurv can contribute additional training seasons rather than ordinary same-week covariates. ED and laboratory positivity can supply contemporaneous auxiliary signals. Scenario trajectories are upstream model outputs, not measurements. Population/calendar/geographic information can enter as offsets, stratification or latent structure rather than a flat feature matrix.

The documentation establishes Google Trends for MIGHTE-Joint, weather for OHT, ED input in CMU's season-end pipeline, and simulated-trajectory inputs for NEU_ISI and Scenariocast. Absence from metadata means not confirmed, not proof of non-use. Ensemble indirect inputs must not be counted as independent direct uses of a covariate.

## Version log

September 28, 2026: expanded the top-eight review to twelve. Current MIGHTE-Nsemble metadata is v3, explicitly for 2026–2027, and lists national WastewaterSCAN influenza A and national NSSP ED inputs with fixed ensemble weights. The ranked 2025–2026 season-end v2 metadata instead describes historical ILI transfer and adaptive weighting. Do not attribute v3 covariates to the ranked v2 model without historical implementation evidence. No datasets, model code or jobs changed.

September 28, 2026 follow-up: distinguished historical supervision/reference libraries from current forecast conditioning. Corrected the possible implication that every listed surveillance source is a contemporaneous predictor. Unverified current-source use remains explicitly unknown.
