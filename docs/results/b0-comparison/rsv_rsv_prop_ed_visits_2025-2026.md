# RSV ED visits · 2025-2026

[Report and configuration key](../b0-configuration-comparison.md)

All configurations and the official ensemble use the same frozen scoring tasks. US is the native national prediction; states/DC are evaluated individually, not summed.

## Four-week projection fans

Each blue fan connects the four horizons from a single forecast origin: median, 50% interval and 95% interval over black frozen truth. Every fourth available origin is shown for readability; all origins are exported and eligible shared tasks are scored. Missing horizons break the lines. ED values are proportions; admissions are counts.

### US projection fans

[![RSV ED visits · 2025-2026: US projection fans](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/fans-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/fans-US.svg)

### North Carolina projection fans

[![RSV ED visits · 2025-2026: North Carolina projection fans](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/fans-37.svg){ loading=lazy }](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/fans-37.svg)

## Scoring diagnostics

WIS components sum to total WIS. Relative WIS uses the official ensemble as the reference (one), whereas the original InfluPaint paper used FluSight-baseline. A zero reference WIS leaves that per-task ratio undefined; the leaderboard records valid ratio counts. Hub horizons 0–3 correspond to internal forecast leads 1–4.

### States/DC WIS components

[![RSV ED visits · 2025-2026: States/DC WIS components](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/components-states_dc.svg){ loading=lazy }](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/components-states_dc.svg)

### US WIS components

[![RSV ED visits · 2025-2026: US WIS components](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/components-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/components-US.svg)

### States/DC relative WIS by horizon

[![RSV ED visits · 2025-2026: States/DC relative WIS by horizon](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/relative-wis-states_dc.svg){ loading=lazy }](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/relative-wis-states_dc.svg)

### US relative WIS by horizon

[![RSV ED visits · 2025-2026: US relative WIS by horizon](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/relative-wis-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/relative-wis-US.svg)

### States/DC WIS over time

[![RSV ED visits · 2025-2026: States/DC WIS over time](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/timeseries-states_dc.svg){ loading=lazy }](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/timeseries-states_dc.svg)

### US WIS over time

[![RSV ED visits · 2025-2026: US WIS over time](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/timeseries-US.svg){ loading=lazy }](../../assets/b0_configuration_comparison/rsv_rsv_prop_ed_visits_2025-2026/timeseries-US.svg)
