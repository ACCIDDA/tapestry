# Other models

The leading FluSight models draw on both **historical training curves** and **current surveillance signals**, but those sources play different roles. Training on historical ILI does not necessarily mean using this week's ILI to forecast hospital admissions. This page compares those roles for the first twelve models in the supplied **2025–2026 FluSight evaluation**, including tied scores.

## Training history and forecast-time inputs

**Training history** supplies examples for estimating parameters or building a reference library. **Forecast-time inputs** condition the current prediction: recent reports, known calendar and geographic information, or forecasts from other models. A model can refit weekly using both; the distinction concerns statistical role, not when a file is read. “Recent” means the latest available observations, not future realized values.

The table summarizes documented inputs, not an audit of every weekly deployment. Links identify metadata or inspected implementations. Unverified uses are marked explicitly.

| Model | Historical training or reference data | Inputs conditioning the current forecast |
|---|---|---|
| **1. Google_SAI-FluEns** | Admissions plus historical ILINet, standardized or mapped into synthetic admissions. | Recent admissions, calendar, population and location, depending on component. **Live ILI use is not established across the ensemble.** No Google search data listed. [Metadata][google] |
| **2. OHT_JHU-nbxd** | Historical admissions and associated covariate sequences; public code supports pretraining on older hospitalization data. | Recent admissions, **dew point and influenza test positivity**. Public code defaults also include temperature, outpatient surveillance and days since July 1; these additional inputs are not verified for every submission. [Metadata][oht] · [Code][oht-code] |
| **3. CMU-TimeSeries** | Admissions plus historical ILI and FluSurv, standardized into additional training examples. | Recent admissions plus **NSSP influenza ED activity** in its auxiliary seasonal component; population and seasonal calendar. ED input is present in season-end code although omitted from the short metadata. [Code][cmu] |
| **4. UGA_flucast-INFLAenza** | Past admissions estimate seasonal, spatial and temporal effects. | Latest admissions update the epidemic state; population, state identity, geographic adjacency, seasonal position, Christmas-relative timing and an early-COVID indicator. Its separate ED forecast does **not** establish ED-to-admission conditioning. [Code][uga] |
| **5. UMass-flusion** | Admissions plus ILI/FluSurv examples pooled for training. Earlier ILI+ preprocessing incorporates test positivity. | In the published formulation: recent admission levels, lags, rolling means, slopes and curvature, plus calendar/location/source information. **Exact additional forecast-time inputs of the newer spatial version remain to be verified.** [Metadata][umass] · [Paper][flusion-paper] |
| **6. NAU-vulPES** | Inherited from members; metadata lists NHSN/HHS, NSSP, ILIp and FluSurv through component models. | Current component forecasts. Exact membership and which raw signals condition each member's hospitalization forecast are incompletely documented. [Metadata][nau] |
| **7. FluSight-ensemble** | No weight training for standard median aggregation; historical learning occurs within members. | Current eligible forecast quantiles. Surveillance sources enter indirectly through contributors. [Metadata][flusight] |
| **8. UVAFluX-FS_OptimWISE** | Recent component forecasts and subsequently observed admissions fit regularized weights over up to four weeks. | Current component quantiles, location/horizon and learned weights. Recent performance informs combination rather than acting as a raw disease signal. [Metadata][uva] |
| **9. MIGHTE-Joint** | Historical admissions, imputed history and component-specific predictor histories. | Recent admissions and **Google Trends**, through LightGBM/ARIMA/ARGO2 components. Exact search terms and lag choices are unspecified. [Metadata][mighte-joint] |
| **10. MIGHTE-Nsemble v2** | Admissions plus historical ILI converted into hospitalization history. | Recent own-state and other-state admissions; lags 1–12; rolling mean, SD, min/max over 2–12 weeks; differences, percentage changes and cyclical week/month features. Recent component errors update weights. **Live ILI use is not established.** [Metadata][mighte] |
| **11. NEU_ISI-AdaptiveEnsemble** | Flu Scenario Modeling Hub trajectory library, across scenarios. | Recent admissions reject projected trajectories that fail to match observed trends. Scenario assumptions enter indirectly through projections. [Metadata][neu] |
| **12. UGA_flucast-Scenariocast** | Scenario Modeling Hub trajectory library. | Recent admission **growth-rate patterns** select nearest-neighbor projected trajectories. [Metadata][scenario] |

## Historical transfer is not a live covariate feed

Google's [published task specification][google-task] supplies historical ILINet tables only for dates **before October 15, 2022**. It proposes standardizing those curves into extra training seasons or mapping them into synthetic admission histories. This establishes historical augmentation for that task, not an ensemble-wide current ILI feed. Broader paper descriptions do not establish identical input policies for every deployed component.

Flusion's [published formulation][flusion-paper] pools source-specific forecasting examples. ILI histories provide examples of ILI continuations; hospitalization histories provide examples of hospitalization continuations. Shared parameters transfer information between sources. It does not follow that each admission prediction consumes contemporaneous ILI or FluSurv observations. The 2025–2026 operational spatial version requires separate verification.

Scenario libraries are a third case: NEU_ISI and Scenariocast use **projected trajectories as a reference library**, with recent observations selecting compatible futures. These trajectories are model outputs, not additional surveillance measurements. Scenariocast's use of growth rates is particularly relevant when comparing epidemic curves with different absolute scales.

## What this means for Tapestry

The clearest documented additional current signals in this comparison are **ED activity for CMU**, **weather and laboratory positivity for OHT**, and **Google Trends for MIGHTE-Joint**. Other states' recent admission histories provide spatial information for MIGHTE-Nsemble. Population and calendar can enter through normalization, offsets, stratification or latent structure rather than as ordinary regression columns.

Two distinct experiments follow from this evidence:

- **More historical supervision:** use older ILI, FluSurv or simulated curves to learn transferable dynamics.
- **More current information:** condition forecasts on recent ED activity, positivity, weather or other signals available at issuance.

Evaluate these separately. Success from historical transfer does not establish the value of a new live covariate, and a richer live input set does not replace a longer training history. This is a research interpretation, not evidence that either change will improve Tapestry.

## Scope and version limitations

- Rankings refer to the supplied 2025–2026 evaluation, not the current season. Historical metadata is pinned to [FluSight commit fd6cfaa][metadata].
- MIGHTE-Nsemble's newer v3 metadata explicitly describes **2026–2027**, adding national WastewaterSCAN influenza A and national NSSP ED inputs with fixed weights. The ranked v2 documentation describes historical ILI transfer and adaptive weights. Do not attribute v3 inputs to v2 without implementation evidence.
- Absence from metadata means **unconfirmed**, not proof of non-use. Optional public code paths do not establish production settings. Inputs inherited through an ensemble are not separate direct uses by that ensemble.
- Historical training sources can be refreshed during weekly refitting without becoming current conditioning variables. Neither metadata nor this table certifies historical release-time availability.
- See the [detailed model review](flusight-2025-2026-model-review.md) for architectures, papers, performance and further implementation caveats.


[metadata]: https://github.com/cdcepi/FluSight-forecast-hub/tree/fd6cfaabe10c147d0268c2b34087e0aa036df357/model-metadata
[google]: https://github.com/cdcepi/FluSight-forecast-hub/blob/fd6cfaabe10c147d0268c2b34087e0aa036df357/model-metadata/Google_SAI-FluEns.yml
[google-task]: https://github.com/google-research/google-research/blob/master/epi_forecasts/problem_statements/flu_problem_statement.txt
[oht]: https://github.com/cdcepi/FluSight-forecast-hub/blob/fd6cfaabe10c147d0268c2b34087e0aa036df357/model-metadata/OHT_JHU-nbxd.yml
[oht-code]: https://github.com/CDDEP-DC/nbeats-xd/blob/87f255c68b13e5dfd721035c45c672cacf6790ee/data_utils/flu.py
[cmu]: https://github.com/cmu-delphi/exploration-tooling/blob/e86834be5f681fda60591dee0105d3d81df179c0/scripts/flu_hosp_prod.R
[uga]: https://github.com/brendandaisy/inla-forecasting-paper/blob/c980776fe342df05cc21546c2153bbd786500537/scripts/flusight-25-26/inflaenza-forecast.R
[umass]: https://github.com/cdcepi/FluSight-forecast-hub/blob/fd6cfaabe10c147d0268c2b34087e0aa036df357/model-metadata/UMass-flusion.yml
[flusion-paper]: https://arxiv.org/html/2407.19054v1
[nau]: https://github.com/cdcepi/FluSight-forecast-hub/blob/fd6cfaabe10c147d0268c2b34087e0aa036df357/model-metadata/NAU-vulPES.yml
[flusight]: https://github.com/cdcepi/FluSight-forecast-hub/blob/fd6cfaabe10c147d0268c2b34087e0aa036df357/model-metadata/FluSight-ensemble.yml
[uva]: https://github.com/cdcepi/FluSight-forecast-hub/blob/fd6cfaabe10c147d0268c2b34087e0aa036df357/model-metadata/UVAFluX-FS_OptimWISE.yml
[mighte-joint]: https://github.com/cdcepi/FluSight-forecast-hub/blob/fd6cfaabe10c147d0268c2b34087e0aa036df357/model-metadata/MIGHTE-Joint.yml
[mighte]: https://github.com/cdcepi/FluSight-forecast-hub/blob/fd6cfaabe10c147d0268c2b34087e0aa036df357/model-metadata/MIGHTE-Nsemble.yml
[neu]: https://github.com/cdcepi/FluSight-forecast-hub/blob/fd6cfaabe10c147d0268c2b34087e0aa036df357/model-metadata/NEU_ISI-AdaptiveEnsemble.yml
[scenario]: https://github.com/cdcepi/FluSight-forecast-hub/blob/fd6cfaabe10c147d0268c2b34087e0aa036df357/model-metadata/UGA_flucast-Scenariocast.yml
