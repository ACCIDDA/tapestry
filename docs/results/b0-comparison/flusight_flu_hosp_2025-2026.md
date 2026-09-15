# Influenza admissions · 2025-2026

[Canonical report and model names](../b0-configuration-comparison.md)

All models use the same frozen tasks. US is the native national prediction; states/DC are evaluated individually. Fans connect the four horizons from one forecast origin. Fans show the three best configurations across all six targets, ranked by mean seed score, plus the official ensemble (light blue) and this target/season’s best configuration (light red). Bands show 50%/95% intervals; black curves show truth. Each panel uses the middle-performing seed under its selection objective. Identical representative runs appear once; different middle seeds appear separately. Every fourth origin is illustrated. Admissions are counts; ED visits are proportions.

Overall top three (middle seeds): `residual2 · seed 44`, `latent32 · seed 43`, `balanced · seed 44`. Target/season best (middle seed): `conv_h26 · seed 42`. See the [model differences table](../b0-configuration-comparison.md#model-differences) and the canonical report’s equal-target WIS-ratio selection rule.

## Target/season configuration ranking

Arithmetic mean across seeds of each seed’s geometric WIS ratio across US and states/DC. All four horizons are included. Lower is better; 1 is ensemble parity.

| Variant | Mean seed score ± SD | Middle seed |
| --- | --- | --- |
| conv_h26 | 0.8920 ± 0.1160 | 42 |
| residual2 | 0.9121 ± 0.0289 | 44 |
| conv_h12 | 0.9616 ± 0.1599 | 42 |
| balanced | 0.9653 ± 0.0598 | 42 |
| anchor | 0.9847 ± 0.1042 | 43 |
| latent32 | 0.9855 ± 0.0780 | 43 |
| state_us | 1.0125 ± 0.1103 | 43 |
| mlp_h8 | 1.0258 ± 0.0582 | 43 |
| flu_only | 1.0371 ± 0.1491 | 43 |
| residual2_z32 | 1.0517 ± 0.0758 | 44 |
| mlp_h26_dynamics | 1.0548 ± 0.0824 | 43 |
| mlp_h26 | 1.0734 ± 0.0764 | 44 |
| mlp_h12 | 1.0834 ± 0.0449 | 44 |
| baseline | 1.1132 ± 0.0577 | 44 |

## US projection fans

[![Influenza admissions · 2025-2026: US projection fans](../../assets/b0_configuration_comparison/flusight_flu_hosp_2025-2026/fans-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_hosp_2025-2026/fans-US.svg)

## North Carolina projection fans

[![Influenza admissions · 2025-2026: North Carolina projection fans](../../assets/b0_configuration_comparison/flusight_flu_hosp_2025-2026/fans-37.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_hosp_2025-2026/fans-37.svg)

## States/DC WIS components

[![Influenza admissions · 2025-2026: States/DC WIS components](../../assets/b0_configuration_comparison/flusight_flu_hosp_2025-2026/components-states_dc.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_hosp_2025-2026/components-states_dc.svg)

## US WIS components

[![Influenza admissions · 2025-2026: US WIS components](../../assets/b0_configuration_comparison/flusight_flu_hosp_2025-2026/components-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_hosp_2025-2026/components-US.svg)

## States/DC relative WIS

[![Influenza admissions · 2025-2026: States/DC relative WIS](../../assets/b0_configuration_comparison/flusight_flu_hosp_2025-2026/relative-wis-states_dc.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_hosp_2025-2026/relative-wis-states_dc.svg)

## US relative WIS

[![Influenza admissions · 2025-2026: US relative WIS](../../assets/b0_configuration_comparison/flusight_flu_hosp_2025-2026/relative-wis-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_hosp_2025-2026/relative-wis-US.svg)

## States/DC WIS over time

[![Influenza admissions · 2025-2026: States/DC WIS over time](../../assets/b0_configuration_comparison/flusight_flu_hosp_2025-2026/timeseries-states_dc.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_hosp_2025-2026/timeseries-states_dc.svg)

## US WIS over time

[![Influenza admissions · 2025-2026: US WIS over time](../../assets/b0_configuration_comparison/flusight_flu_hosp_2025-2026/timeseries-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_hosp_2025-2026/timeseries-US.svg)
