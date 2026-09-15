# COVID-19 admissions · 2024-2025

[Canonical report and model names](../b0-configuration-comparison.md)

All models use the same frozen tasks. US is the native national prediction; states/DC are evaluated individually. Fans connect the four horizons from one forecast origin. Fans show the three best configurations across all six targets, ranked by mean seed score, plus the official ensemble (light blue) and this target/season’s best configuration (light red). Bands show 50%/95% intervals; black curves show truth. Each panel uses the middle-performing seed under its selection objective. Identical representative runs appear once; different middle seeds appear separately. Every fourth origin is illustrated. Admissions are counts; ED visits are proportions.

Overall top three (middle seeds): `residual2 · seed 44`, `latent32 · seed 43`, `balanced · seed 44`. Target/season best (middle seed): `conv_h12 · seed 44`. See the [model differences table](../b0-configuration-comparison.md#model-differences) and the canonical report’s equal-target WIS-ratio selection rule.

## Target/season configuration ranking

Arithmetic mean across seeds of each seed’s geometric WIS ratio across US and states/DC. All four horizons are included. Lower is better; 1 is ensemble parity.

| Variant | Mean seed score ± SD | Middle seed |
| --- | --- | --- |
| conv_h12 | 0.8946 ± 0.0233 | 44 |
| latent32 | 0.9007 ± 0.1280 | 43 |
| residual2 | 0.9551 ± 0.0174 | 42 |
| conv_h26 | 0.9749 ± 0.0219 | 44 |
| mlp_h26 | 1.0003 ± 0.0711 | 44 |
| mlp_h12 | 1.0769 ± 0.0684 | 43 |
| balanced | 1.1100 ± 0.1935 | 44 |
| state_us | 1.1123 ± 0.0665 | 43 |
| residual2_z32 | 1.1364 ± 0.0488 | 44 |
| mlp_h26_dynamics | 1.1411 ± 0.3256 | 43 |
| mlp_h8 | 1.1432 ± 0.3677 | 42 |
| baseline | 1.1615 ± 0.0733 | 43 |
| anchor | 1.1994 ± 0.1619 | 43 |
| flu_only | 1.7825 ± 0.8568 | 42 |

## US projection fans

[![COVID-19 admissions · 2024-2025: US projection fans](../../assets/b0_configuration_comparison/covid_covid_hosp_2024-2025/fans-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/covid_covid_hosp_2024-2025/fans-US.svg)

## North Carolina projection fans

[![COVID-19 admissions · 2024-2025: North Carolina projection fans](../../assets/b0_configuration_comparison/covid_covid_hosp_2024-2025/fans-37.svg){ loading=lazy }](../../assets/b0_configuration_comparison/covid_covid_hosp_2024-2025/fans-37.svg)

## States/DC WIS components

[![COVID-19 admissions · 2024-2025: States/DC WIS components](../../assets/b0_configuration_comparison/covid_covid_hosp_2024-2025/components-states_dc.svg){ loading=lazy }](../../assets/b0_configuration_comparison/covid_covid_hosp_2024-2025/components-states_dc.svg)

## US WIS components

[![COVID-19 admissions · 2024-2025: US WIS components](../../assets/b0_configuration_comparison/covid_covid_hosp_2024-2025/components-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/covid_covid_hosp_2024-2025/components-US.svg)

## States/DC relative WIS

[![COVID-19 admissions · 2024-2025: States/DC relative WIS](../../assets/b0_configuration_comparison/covid_covid_hosp_2024-2025/relative-wis-states_dc.svg){ loading=lazy }](../../assets/b0_configuration_comparison/covid_covid_hosp_2024-2025/relative-wis-states_dc.svg)

## US relative WIS

[![COVID-19 admissions · 2024-2025: US relative WIS](../../assets/b0_configuration_comparison/covid_covid_hosp_2024-2025/relative-wis-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/covid_covid_hosp_2024-2025/relative-wis-US.svg)

## States/DC WIS over time

[![COVID-19 admissions · 2024-2025: States/DC WIS over time](../../assets/b0_configuration_comparison/covid_covid_hosp_2024-2025/timeseries-states_dc.svg){ loading=lazy }](../../assets/b0_configuration_comparison/covid_covid_hosp_2024-2025/timeseries-states_dc.svg)

## US WIS over time

[![COVID-19 admissions · 2024-2025: US WIS over time](../../assets/b0_configuration_comparison/covid_covid_hosp_2024-2025/timeseries-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/covid_covid_hosp_2024-2025/timeseries-US.svg)
