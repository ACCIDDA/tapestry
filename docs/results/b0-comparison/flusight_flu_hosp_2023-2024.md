# Influenza admissions · 2023-2024

[Canonical report and model names](../b0-configuration-comparison.md)

All models use the same frozen tasks. US is the native national prediction; states/DC are evaluated individually. Fans connect the four horizons from one forecast origin. Fans show the three best configurations across all six targets, ranked by mean seed score, plus the official ensemble (light blue) and this target/season’s best configuration (light red). Bands show 50%/95% intervals; black curves show truth. Each panel uses the middle-performing seed under its selection objective. Identical representative runs appear once; different middle seeds appear separately. Every fourth origin is illustrated. Admissions are counts; ED visits are proportions.

Overall top three (middle seeds): `residual2 · seed 44`, `latent32 · seed 43`, `balanced · seed 44`. Target/season best (middle seed): `conv_h12 · seed 42`. See the [model differences table](../b0-configuration-comparison.md#model-differences) and the canonical report’s equal-target WIS-ratio selection rule.

## Target/season configuration ranking

Arithmetic mean across seeds of each seed’s geometric WIS ratio across US and states/DC. All four horizons are included. Lower is better; 1 is ensemble parity.

| Variant | Mean seed score ± SD | Middle seed |
| --- | --- | --- |
| conv_h12 | 0.8684 ± 0.1332 | 42 |
| mlp_h8 | 0.8773 ± 0.0304 | 44 |
| balanced | 0.8813 ± 0.0367 | 43 |
| residual2 | 0.9037 ± 0.0379 | 42 |
| residual2_z32 | 0.9184 ± 0.1075 | 42 |
| mlp_h12 | 0.9453 ± 0.0952 | 42 |
| anchor | 0.9615 ± 0.0655 | 44 |
| state_us | 0.9734 ± 0.0901 | 44 |
| flu_only | 1.0547 ± 0.1294 | 44 |
| latent32 | 1.0633 ± 0.0334 | 43 |
| conv_h26 | 1.1306 ± 0.0781 | 43 |
| mlp_h26 | 1.2206 ± 0.2771 | 43 |
| baseline | 1.3023 ± 0.0542 | 42 |
| mlp_h26_dynamics | 1.3824 ± 0.2129 | 43 |

## US projection fans

[![Influenza admissions · 2023-2024: US projection fans](../../assets/b0_configuration_comparison/flusight_flu_hosp_2023-2024/fans-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_hosp_2023-2024/fans-US.svg)

## North Carolina projection fans

[![Influenza admissions · 2023-2024: North Carolina projection fans](../../assets/b0_configuration_comparison/flusight_flu_hosp_2023-2024/fans-37.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_hosp_2023-2024/fans-37.svg)

## States/DC WIS components

[![Influenza admissions · 2023-2024: States/DC WIS components](../../assets/b0_configuration_comparison/flusight_flu_hosp_2023-2024/components-states_dc.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_hosp_2023-2024/components-states_dc.svg)

## US WIS components

[![Influenza admissions · 2023-2024: US WIS components](../../assets/b0_configuration_comparison/flusight_flu_hosp_2023-2024/components-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_hosp_2023-2024/components-US.svg)

## States/DC relative WIS

[![Influenza admissions · 2023-2024: States/DC relative WIS](../../assets/b0_configuration_comparison/flusight_flu_hosp_2023-2024/relative-wis-states_dc.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_hosp_2023-2024/relative-wis-states_dc.svg)

## US relative WIS

[![Influenza admissions · 2023-2024: US relative WIS](../../assets/b0_configuration_comparison/flusight_flu_hosp_2023-2024/relative-wis-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_hosp_2023-2024/relative-wis-US.svg)

## States/DC WIS over time

[![Influenza admissions · 2023-2024: States/DC WIS over time](../../assets/b0_configuration_comparison/flusight_flu_hosp_2023-2024/timeseries-states_dc.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_hosp_2023-2024/timeseries-states_dc.svg)

## US WIS over time

[![Influenza admissions · 2023-2024: US WIS over time](../../assets/b0_configuration_comparison/flusight_flu_hosp_2023-2024/timeseries-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_hosp_2023-2024/timeseries-US.svg)
