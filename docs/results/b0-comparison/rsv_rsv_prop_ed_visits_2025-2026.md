# RSV ED visits · 2025-2026

[Canonical report and model names](../b0-configuration-comparison.md)

All models use the same frozen tasks. US is the native national prediction; states/DC are evaluated individually. Fans connect the four horizons from one forecast origin. Fans show the three best configurations across all six targets, ranked by mean seed score, plus the official ensemble (light blue) and this target/season’s best configuration (light red). Bands show 50%/95% intervals; black curves show truth. Each panel uses the middle-performing seed under its selection objective. Identical representative runs appear once; different middle seeds appear separately. Every fourth origin is illustrated. Admissions are counts; ED visits are proportions.

Overall top three (middle seeds): `residual2 · seed 44`, `latent32 · seed 43`, `balanced · seed 44`. Target/season best (middle seed): `conv_h12 · seed 43`. See the [model differences table](../b0-configuration-comparison.md#model-differences) and the canonical report’s equal-target WIS-ratio selection rule.

## Target/season configuration ranking

Arithmetic mean across seeds of each seed’s geometric WIS ratio across US and states/DC. All four horizons are included. Lower is better; 1 is ensemble parity.

| Variant | Mean seed score ± SD | Middle seed |
| --- | --- | --- |
| conv_h12 | 0.8093 ± 0.0294 | 43 |
| latent32 | 0.8548 ± 0.0386 | 43 |
| conv_h26 | 0.8746 ± 0.0121 | 44 |
| residual2 | 0.8813 ± 0.0110 | 42 |
| balanced | 0.8989 ± 0.0387 | 42 |
| residual2_z32 | 0.9194 ± 0.0773 | 44 |
| mlp_h12 | 0.9382 ± 0.0573 | 44 |
| anchor | 0.9621 ± 0.0246 | 44 |
| mlp_h26 | 0.9741 ± 0.1142 | 43 |
| mlp_h8 | 0.9755 ± 0.0918 | 44 |
| mlp_h26_dynamics | 1.0002 ± 0.0473 | 43 |
| state_us | 1.1013 ± 0.1099 | 42 |
| baseline | 1.4035 ± 0.2526 | 42 |
| flu_only | 1.4775 ± 0.0758 | 42 |

## US projection fans

[![RSV ED visits · 2025-2026: US projection fans](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/fans-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/fans-US.svg)

## North Carolina projection fans

[![RSV ED visits · 2025-2026: North Carolina projection fans](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/fans-37.svg){ loading=lazy }](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/fans-37.svg)

## States/DC WIS components

[![RSV ED visits · 2025-2026: States/DC WIS components](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/components-states_dc.svg){ loading=lazy }](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/components-states_dc.svg)

## US WIS components

[![RSV ED visits · 2025-2026: US WIS components](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/components-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/components-US.svg)

## States/DC relative WIS

[![RSV ED visits · 2025-2026: States/DC relative WIS](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/relative-wis-states_dc.svg){ loading=lazy }](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/relative-wis-states_dc.svg)

## US relative WIS

[![RSV ED visits · 2025-2026: US relative WIS](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/relative-wis-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/relative-wis-US.svg)

## States/DC WIS over time

[![RSV ED visits · 2025-2026: States/DC WIS over time](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/timeseries-states_dc.svg){ loading=lazy }](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/timeseries-states_dc.svg)

## US WIS over time

[![RSV ED visits · 2025-2026: US WIS over time](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/timeseries-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/timeseries-US.svg)
