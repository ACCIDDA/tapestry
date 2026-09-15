# Influenza ED visits · 2025-2026

[Canonical report and model names](../b0-configuration-comparison.md)

All models use the same frozen tasks. US is the native national prediction; states/DC are evaluated individually. Fans connect the four horizons from one forecast origin. Fans show the three best configurations across all six targets, ranked by mean seed score, plus the official ensemble (light blue) and this target/season’s best configuration (light red). Bands show 50%/95% intervals; black curves show truth. Each panel uses the middle-performing seed under its selection objective. Identical representative runs appear once; different middle seeds appear separately. Every fourth origin is illustrated. Admissions are counts; ED visits are proportions.

Overall top three (middle seeds): `residual2 · seed 44`, `latent32 · seed 43`, `balanced · seed 44`. Target/season best (middle seed): `conv_h26 · seed 42`. See the [model differences table](../b0-configuration-comparison.md#model-differences) and the canonical report’s equal-target WIS-ratio selection rule.

## Target/season configuration ranking

Arithmetic mean across seeds of each seed’s geometric WIS ratio across US and states/DC. All four horizons are included. Lower is better; 1 is ensemble parity.

| Variant | Mean seed score ± SD | Middle seed |
| --- | --- | --- |
| conv_h26 | 0.9919 ± 0.0401 | 42 |
| baseline | 1.0559 ± 0.0540 | 42 |
| residual2 | 1.1029 ± 0.0110 | 42 |
| mlp_h26_dynamics | 1.1057 ± 0.0380 | 42 |
| conv_h12 | 1.1073 ± 0.0921 | 42 |
| anchor | 1.1225 ± 0.0635 | 42 |
| mlp_h26 | 1.1334 ± 0.0533 | 43 |
| residual2_z32 | 1.1349 ± 0.0324 | 44 |
| latent32 | 1.1530 ± 0.0578 | 43 |
| mlp_h12 | 1.1667 ± 0.0256 | 44 |
| mlp_h8 | 1.1692 ± 0.0264 | 43 |
| balanced | 1.2725 ± 0.0333 | 42 |
| state_us | 1.3429 ± 0.0559 | 42 |
| flu_only | 1.7319 ± 0.0407 | 42 |

## US projection fans

[![Influenza ED visits · 2025-2026: US projection fans](../../assets/b0_configuration_comparison/flusight_flu_prop_ed_visits_2025-2026/fans-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_prop_ed_visits_2025-2026/fans-US.svg)

## North Carolina projection fans

[![Influenza ED visits · 2025-2026: North Carolina projection fans](../../assets/b0_configuration_comparison/flusight_flu_prop_ed_visits_2025-2026/fans-37.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_prop_ed_visits_2025-2026/fans-37.svg)

## States/DC WIS components

[![Influenza ED visits · 2025-2026: States/DC WIS components](../../assets/b0_configuration_comparison/flusight_flu_prop_ed_visits_2025-2026/components-states_dc.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_prop_ed_visits_2025-2026/components-states_dc.svg)

## US WIS components

[![Influenza ED visits · 2025-2026: US WIS components](../../assets/b0_configuration_comparison/flusight_flu_prop_ed_visits_2025-2026/components-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_prop_ed_visits_2025-2026/components-US.svg)

## States/DC relative WIS

[![Influenza ED visits · 2025-2026: States/DC relative WIS](../../assets/b0_configuration_comparison/flusight_flu_prop_ed_visits_2025-2026/relative-wis-states_dc.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_prop_ed_visits_2025-2026/relative-wis-states_dc.svg)

## US relative WIS

[![Influenza ED visits · 2025-2026: US relative WIS](../../assets/b0_configuration_comparison/flusight_flu_prop_ed_visits_2025-2026/relative-wis-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_prop_ed_visits_2025-2026/relative-wis-US.svg)

## States/DC WIS over time

[![Influenza ED visits · 2025-2026: States/DC WIS over time](../../assets/b0_configuration_comparison/flusight_flu_prop_ed_visits_2025-2026/timeseries-states_dc.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_prop_ed_visits_2025-2026/timeseries-states_dc.svg)

## US WIS over time

[![Influenza ED visits · 2025-2026: US WIS over time](../../assets/b0_configuration_comparison/flusight_flu_prop_ed_visits_2025-2026/timeseries-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/flusight_flu_prop_ed_visits_2025-2026/timeseries-US.svg)
