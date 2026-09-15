# COVID-19 ED visits · 2025-2026

[Canonical report and model names](../b0-configuration-comparison.md)

All models use the same frozen tasks. US is the native national prediction; states/DC are evaluated individually. Fans connect the four horizons from one forecast origin. Fans show the three best configurations across all six targets, ranked by mean seed score, plus the official ensemble (light blue) and this target/season’s best configuration (light red). Bands show 50%/95% intervals; black curves show truth. Each panel uses the middle-performing seed under its selection objective. Identical representative runs appear once; different middle seeds appear separately. Every fourth origin is illustrated. Admissions are counts; ED visits are proportions.

Overall top three (middle seeds): `residual2 · seed 44`, `latent32 · seed 43`, `balanced · seed 44`. Target/season best (middle seed): `anchor · seed 42`. See the [model differences table](../b0-configuration-comparison.md#model-differences) and the canonical report’s equal-target WIS-ratio selection rule.

## Target/season configuration ranking

Arithmetic mean across seeds of each seed’s geometric WIS ratio across US and states/DC. All four horizons are included. Lower is better; 1 is ensemble parity.

| Variant | Mean seed score ± SD | Middle seed |
| --- | --- | --- |
| anchor | 1.1862 ± 0.0180 | 42 |
| mlp_h26_dynamics | 1.2260 ± 0.0845 | 44 |
| residual2_z32 | 1.2698 ± 0.2519 | 44 |
| mlp_h26 | 1.3130 ± 0.0982 | 42 |
| balanced | 1.3318 ± 0.1570 | 42 |
| mlp_h8 | 1.3449 ± 0.3498 | 44 |
| baseline | 1.3600 ± 0.1457 | 42 |
| mlp_h12 | 1.3833 ± 0.4240 | 44 |
| state_us | 1.4457 ± 0.1220 | 42 |
| residual2 | 1.4701 ± 0.1351 | 43 |
| latent32 | 1.5161 ± 0.1488 | 43 |
| conv_h26 | 1.6149 ± 0.1341 | 44 |
| flu_only | 1.7639 ± 0.5020 | 44 |
| conv_h12 | 2.0377 ± 0.1266 | 43 |

## US projection fans

[![COVID-19 ED visits · 2025-2026: US projection fans](../../assets/b0_configuration_comparison/covid_covid_prop_ed_visits_2025-2026/fans-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/covid_covid_prop_ed_visits_2025-2026/fans-US.svg)

## North Carolina projection fans

[![COVID-19 ED visits · 2025-2026: North Carolina projection fans](../../assets/b0_configuration_comparison/covid_covid_prop_ed_visits_2025-2026/fans-37.svg){ loading=lazy }](../../assets/b0_configuration_comparison/covid_covid_prop_ed_visits_2025-2026/fans-37.svg)

## States/DC WIS components

[![COVID-19 ED visits · 2025-2026: States/DC WIS components](../../assets/b0_configuration_comparison/covid_covid_prop_ed_visits_2025-2026/components-states_dc.svg){ loading=lazy }](../../assets/b0_configuration_comparison/covid_covid_prop_ed_visits_2025-2026/components-states_dc.svg)

## US WIS components

[![COVID-19 ED visits · 2025-2026: US WIS components](../../assets/b0_configuration_comparison/covid_covid_prop_ed_visits_2025-2026/components-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/covid_covid_prop_ed_visits_2025-2026/components-US.svg)

## States/DC relative WIS

[![COVID-19 ED visits · 2025-2026: States/DC relative WIS](../../assets/b0_configuration_comparison/covid_covid_prop_ed_visits_2025-2026/relative-wis-states_dc.svg){ loading=lazy }](../../assets/b0_configuration_comparison/covid_covid_prop_ed_visits_2025-2026/relative-wis-states_dc.svg)

## US relative WIS

[![COVID-19 ED visits · 2025-2026: US relative WIS](../../assets/b0_configuration_comparison/covid_covid_prop_ed_visits_2025-2026/relative-wis-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/covid_covid_prop_ed_visits_2025-2026/relative-wis-US.svg)

## States/DC WIS over time

[![COVID-19 ED visits · 2025-2026: States/DC WIS over time](../../assets/b0_configuration_comparison/covid_covid_prop_ed_visits_2025-2026/timeseries-states_dc.svg){ loading=lazy }](../../assets/b0_configuration_comparison/covid_covid_prop_ed_visits_2025-2026/timeseries-states_dc.svg)

## US WIS over time

[![COVID-19 ED visits · 2025-2026: US WIS over time](../../assets/b0_configuration_comparison/covid_covid_prop_ed_visits_2025-2026/timeseries-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/covid_covid_prop_ed_visits_2025-2026/timeseries-US.svg)
