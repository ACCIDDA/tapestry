# Influenza admissions · 2024-2025

[Canonical report and model names](../b0-configuration-comparison.md)

All models use the same frozen tasks. US is the native national prediction; states/DC are evaluated individually. Fans connect the four horizons from one forecast origin. Fans show the three best configurations across all six targets, ranked by mean seed score, plus the official ensemble (light blue) and this target/season’s best configuration (light red). Bands show 50%/95% intervals; black curves show truth. Each panel uses the middle-performing seed under its selection objective. Identical representative runs appear once; different middle seeds appear separately. Every fourth origin is illustrated. Admissions are counts; ED visits are proportions.

Overall top three (middle seeds): `residual2 · seed 44`, `latent32 · seed 43`, `balanced · seed 44`. Target/season best (middle seed): `latent32 · seed 44`. See the [model differences table](../b0-configuration-comparison.md#model-differences) and the canonical report’s equal-target WIS-ratio selection rule.

## Target/season configuration ranking

Arithmetic mean across seeds of each seed’s geometric WIS ratio across US and states/DC. All four horizons are included. Lower is better; 1 is ensemble parity.

| Variant | Mean seed score ± SD | Middle seed |
| --- | --- | --- |
| latent32 | 0.7990 ± 0.0921 | 44 |
| mlp_h12 | 0.8563 ± 0.1473 | 43 |
| residual2 | 0.8643 ± 0.0651 | 42 |
| conv_h12 | 0.8654 ± 0.0748 | 43 |
| mlp_h26 | 0.8755 ± 0.0961 | 42 |
| flu_only | 0.8759 ± 0.0203 | 44 |
| state_us | 0.9129 ± 0.0677 | 44 |
| mlp_h26_dynamics | 0.9464 ± 0.2282 | 43 |
| residual2_z32 | 0.9469 ± 0.0338 | 42 |
| anchor | 0.9485 ± 0.0369 | 42 |
| baseline | 1.0113 ± 0.0527 | 43 |
| mlp_h8 | 1.0258 ± 0.2329 | 44 |
| conv_h26 | 1.0412 ± 0.1450 | 42 |
| balanced | 1.0564 ± 0.2908 | 44 |

## US projection fans

[![Influenza admissions · 2024-2025: US projection fans](../../assets/b0_configuration_comparison/flusight_flu_hosp_2024-2025/fans-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_hosp_2024-2025/fans-US.svg)

## North Carolina projection fans

[![Influenza admissions · 2024-2025: North Carolina projection fans](../../assets/b0_configuration_comparison/flusight_flu_hosp_2024-2025/fans-37.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_hosp_2024-2025/fans-37.svg)

## States/DC WIS components

[![Influenza admissions · 2024-2025: States/DC WIS components](../../assets/b0_configuration_comparison/flusight_flu_hosp_2024-2025/components-states_dc.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_hosp_2024-2025/components-states_dc.svg)

## US WIS components

[![Influenza admissions · 2024-2025: US WIS components](../../assets/b0_configuration_comparison/flusight_flu_hosp_2024-2025/components-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_hosp_2024-2025/components-US.svg)

## States/DC relative WIS

[![Influenza admissions · 2024-2025: States/DC relative WIS](../../assets/b0_configuration_comparison/flusight_flu_hosp_2024-2025/relative-wis-states_dc.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_hosp_2024-2025/relative-wis-states_dc.svg)

## US relative WIS

[![Influenza admissions · 2024-2025: US relative WIS](../../assets/b0_configuration_comparison/flusight_flu_hosp_2024-2025/relative-wis-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_hosp_2024-2025/relative-wis-US.svg)

## States/DC WIS over time

[![Influenza admissions · 2024-2025: States/DC WIS over time](../../assets/b0_configuration_comparison/flusight_flu_hosp_2024-2025/timeseries-states_dc.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_hosp_2024-2025/timeseries-states_dc.svg)

## US WIS over time

[![Influenza admissions · 2024-2025: US WIS over time](../../assets/b0_configuration_comparison/flusight_flu_hosp_2024-2025/timeseries-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_hosp_2024-2025/timeseries-US.svg)
