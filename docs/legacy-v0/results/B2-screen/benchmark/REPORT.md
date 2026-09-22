# B2 secondary EpiBench evaluation

This is the **secondary EpiBench diagnostic report**. Use the [primary matched-control report](../index.md) for the scientific location-relative, season-first ranking.

This local mirror contains the ranking tables, provenance and SVG figures. Full Hubverse exports and per-task score CSVs remain on Longleaf at `/proj/jlessler/projects/tapestry-all/tapestry/data/experiments/B2-screen/comparison-0108ba6fb836/`. Dense 32-model plots emitted tight-layout warnings; visual inspection is left to the user.

32 saved runs; 32 configurations; 9 target/season comparisons.

All runs use the same frozen ensemble-supported forecast tasks and truth. Scores come from the full EpiBench config pipeline (including R scoringutils and relative WIS); the diagnostic plots call EpiBench plotting code. These are retrospective CV results with final outcomes; input modes are finalized or Wednesday as recorded per configuration. They are exploratory selections, not prospective rankings.

Configurations rank by the arithmetic mean of all-target seed objectives, including ED. Each seed objective is the geometric WIS ratio with equal target weight, then equal season/geography weight. Fans use the middle-performing seed. Lower is better. The secondary influenza objective is the geometric mean of WIS/ensemble-WIS ratios, equally weighting each season and US versus states/DC. The admissions objective first gives each admission target equal weight. Configuration results average the run objectives; different seed counts and selection on these folds limit comparisons. The plotted relative WIS is the mean of per-task ratios, a different statistic.

| Configuration | Variant | Seeds | All-target mean ± SD | Middle seed | Flu objective | Admissions objective |
|---|---|---:|---:|---:|---:|---:|
| b2-target-finalized-none-7472b43fd9c1 | target_gap__finalized__none | 1 | 0.8220 ± nan | 42 | 0.9226 | 0.8298 |
| b2-pathogen-wednesday-none-f823c4c36142 | pathogen_mixed__wednesday__none | 1 | 0.8253 ± nan | 42 | 0.8709 | 0.7819 |
| b2-pathogen-finalized-ww_wval_like+ww_pct_rank-8b3084f7a0a9 | pathogen_mixed__finalized__ww_wval_like+ww_pct_rank | 1 | 0.8351 ± nan | 42 | 0.9286 | 0.7814 |
| b2-pathogen-finalized-none-cc640d6629fd | pathogen_mixed__finalized__none | 1 | 0.8471 ± nan | 42 | 0.9643 | 0.7645 |
| b2-pathogen-finalized-ww_wval_like-fe02f6924a04 | pathogen_mixed__finalized__ww_wval_like | 1 | 0.8755 ± nan | 42 | 1.0303 | 0.7835 |
| b2-pathogen-finalized-inpatient-24b6f42c5bf2 | pathogen_mixed__finalized__inpatient | 1 | 0.8885 ± nan | 42 | 1.0825 | 0.8321 |
| b2-pathogen-wednesday-ww_wval_like-e96bf604d397 | pathogen_mixed__wednesday__ww_wval_like | 1 | 0.8966 ± nan | 42 | 0.8574 | 0.8495 |
| b2-target-wednesday-ww_wval_like-3faa8dce0825 | target_gap__wednesday__ww_wval_like | 1 | 0.9002 ± nan | 42 | 0.9683 | 0.8621 |
| b2-pathogen-wednesday-inpatient-67cc918c7063 | pathogen_mixed__wednesday__inpatient | 1 | 0.9006 ± nan | 42 | 0.9260 | 0.8818 |
| b2-target-wednesday-ww_pct_rank-bc903ddcd5ef | target_gap__wednesday__ww_pct_rank | 1 | 0.9012 ± nan | 42 | 1.0038 | 0.8642 |
| b2-pathogen-wednesday-ww_pct_rank-66c4464b9c8f | pathogen_mixed__wednesday__ww_pct_rank | 1 | 0.9017 ± nan | 42 | 0.8595 | 0.8592 |
| b2-pathogen-finalized-ww_pct_rank-9efeeee9cabe | pathogen_mixed__finalized__ww_pct_rank | 1 | 0.9049 ± nan | 42 | 1.0844 | 0.8160 |
| b2-target-finalized-ww_pct_rank-98316a7cdcaf | target_gap__finalized__ww_pct_rank | 1 | 0.9137 ± nan | 42 | 1.0801 | 0.8623 |
| b2-target-finalized-inpatient-73f8055cb7d8 | target_gap__finalized__inpatient | 1 | 0.9214 ± nan | 42 | 1.1047 | 0.8435 |
| b2-target-wednesday-none-555a37b06cf1 | target_gap__wednesday__none | 1 | 0.9279 ± nan | 42 | 0.9279 | 0.9205 |
| b2-target-wednesday-inpatient-bfaadacb6702 | target_gap__wednesday__inpatient | 1 | 0.9390 ± nan | 42 | 0.8445 | 0.9464 |
| b2-pathogen-wednesday-ww_wval_like+ww_pct_rank-b65c322374f6 | pathogen_mixed__wednesday__ww_wval_like+ww_pct_rank | 1 | 0.9700 ± nan | 42 | 0.9343 | 0.9663 |
| b2-pathogen-wednesday-outpatient-dcf72ce4aec1 | pathogen_mixed__wednesday__outpatient | 1 | 0.9788 ± nan | 42 | 0.9988 | 0.9077 |
| b2-target-wednesday-inpatient+outpatient+ww_wval_like+ww_pct_rank-fba755623401 | target_gap__wednesday__inpatient+outpatient+ww_wval_like+ww_pct_rank | 1 | 0.9828 ± nan | 42 | 1.0022 | 0.9391 |
| b2-pathogen-wednesday-inpatient+outpatient+ww_wval_like+ww_pct_rank-2c06ce178122 | pathogen_mixed__wednesday__inpatient+outpatient+ww_wval_like+ww_pct_rank | 1 | 0.9943 ± nan | 42 | 1.1394 | 0.9723 |
| b2-target-wednesday-inpatient+outpatient-0bf1abedd0c7 | target_gap__wednesday__inpatient+outpatient | 1 | 1.0002 ± nan | 42 | 1.0404 | 1.0031 |
| b2-target-finalized-ww_wval_like-61d2884d1dbf | target_gap__finalized__ww_wval_like | 1 | 1.0040 ± nan | 42 | 1.1112 | 0.9860 |
| b2-target-finalized-inpatient+outpatient-297850c63c92 | target_gap__finalized__inpatient+outpatient | 1 | 1.0067 ± nan | 42 | 1.0748 | 0.9847 |
| b2-target-wednesday-ww_wval_like+ww_pct_rank-6ce61cdd2655 | target_gap__wednesday__ww_wval_like+ww_pct_rank | 1 | 1.0086 ± nan | 42 | 0.9819 | 1.0325 |
| b2-pathogen-finalized-outpatient-62d52aba0b21 | pathogen_mixed__finalized__outpatient | 1 | 1.0141 ± nan | 42 | 1.1592 | 0.8675 |
| b2-target-finalized-outpatient-5a55f830178e | target_gap__finalized__outpatient | 1 | 1.0230 ± nan | 42 | 1.0559 | 0.9210 |
| b2-pathogen-wednesday-inpatient+outpatient-0d28f682703c | pathogen_mixed__wednesday__inpatient+outpatient | 1 | 1.0314 ± nan | 42 | 1.0714 | 0.9645 |
| b2-target-wednesday-outpatient-e6b01c356f8d | target_gap__wednesday__outpatient | 1 | 1.0314 ± nan | 42 | 1.0459 | 0.9972 |
| b2-target-finalized-ww_wval_like+ww_pct_rank-72966234cc15 | target_gap__finalized__ww_wval_like+ww_pct_rank | 1 | 1.0322 ± nan | 42 | 1.0272 | 0.9506 |
| b2-target-finalized-inpatient+outpatient+ww_wval_like+ww_pct_rank-ec7c6a53178a | target_gap__finalized__inpatient+outpatient+ww_wval_like+ww_pct_rank | 1 | 1.0781 ± nan | 42 | 1.0769 | 0.9003 |
| b2-pathogen-finalized-inpatient+outpatient+ww_wval_like+ww_pct_rank-558765d208c4 | pathogen_mixed__finalized__inpatient+outpatient+ww_wval_like+ww_pct_rank | 1 | 1.0859 ± nan | 42 | 1.1978 | 0.9114 |
| b2-pathogen-finalized-inpatient+outpatient-cd400ea8e74c | pathogen_mixed__finalized__inpatient+outpatient | 1 | 1.1162 ± nan | 42 | 1.1285 | 0.9694 |

[Configuration ranking (means and seed SD)](configuration_ranking.csv)

[Individual run ranking and identifiers](run_ranking.csv)

[Detailed target/season/geography/horizon leaderboard](leaderboard.csv)

[Configuration definitions and provenance](configurations.json)

Hubverse forecasts are stored under `hubverse/<hub>/model-output/<model_id>/`, one Parquet file per reference date with the saved quantile grid and horizons 0–3. Use `--csv` for CSV companions accepted directly by the EpiBench CLI. ED values are proportions. Every available origin is exported; fans display every fourth origin for readability, with median, 50% and 95% intervals. NC uses FIPS 37.

| Target / season | US fans | NC fans | State WIS components | State relative WIS | State WIS over time |
|---|---|---|---|---|---|
| wk inc flu hosp / 2023-2024 | [US](plots/flusight_flu_hosp_2023-2024/fans-US.svg) | [NC](plots/flusight_flu_hosp_2023-2024/fans-37.svg) | [Components](plots/flusight_flu_hosp_2023-2024/components-states_dc.svg) | [Relative WIS](plots/flusight_flu_hosp_2023-2024/relative-wis-states_dc.svg) | [Time series](plots/flusight_flu_hosp_2023-2024/timeseries-states_dc.svg) |
| wk inc flu hosp / 2024-2025 | [US](plots/flusight_flu_hosp_2024-2025/fans-US.svg) | [NC](plots/flusight_flu_hosp_2024-2025/fans-37.svg) | [Components](plots/flusight_flu_hosp_2024-2025/components-states_dc.svg) | [Relative WIS](plots/flusight_flu_hosp_2024-2025/relative-wis-states_dc.svg) | [Time series](plots/flusight_flu_hosp_2024-2025/timeseries-states_dc.svg) |
| wk inc flu hosp / 2025-2026 | [US](plots/flusight_flu_hosp_2025-2026/fans-US.svg) | [NC](plots/flusight_flu_hosp_2025-2026/fans-37.svg) | [Components](plots/flusight_flu_hosp_2025-2026/components-states_dc.svg) | [Relative WIS](plots/flusight_flu_hosp_2025-2026/relative-wis-states_dc.svg) | [Time series](plots/flusight_flu_hosp_2025-2026/timeseries-states_dc.svg) |
| wk inc flu prop ed visits / 2025-2026 | [US](plots/flusight_flu_prop_ed_visits_2025-2026/fans-US.svg) | [NC](plots/flusight_flu_prop_ed_visits_2025-2026/fans-37.svg) | [Components](plots/flusight_flu_prop_ed_visits_2025-2026/components-states_dc.svg) | [Relative WIS](plots/flusight_flu_prop_ed_visits_2025-2026/relative-wis-states_dc.svg) | [Time series](plots/flusight_flu_prop_ed_visits_2025-2026/timeseries-states_dc.svg) |
| wk inc covid hosp / 2024-2025 | [US](plots/covid_covid_hosp_2024-2025/fans-US.svg) | [NC](plots/covid_covid_hosp_2024-2025/fans-37.svg) | [Components](plots/covid_covid_hosp_2024-2025/components-states_dc.svg) | [Relative WIS](plots/covid_covid_hosp_2024-2025/relative-wis-states_dc.svg) | [Time series](plots/covid_covid_hosp_2024-2025/timeseries-states_dc.svg) |
| wk inc covid hosp / 2025-2026 | [US](plots/covid_covid_hosp_2025-2026/fans-US.svg) | [NC](plots/covid_covid_hosp_2025-2026/fans-37.svg) | [Components](plots/covid_covid_hosp_2025-2026/components-states_dc.svg) | [Relative WIS](plots/covid_covid_hosp_2025-2026/relative-wis-states_dc.svg) | [Time series](plots/covid_covid_hosp_2025-2026/timeseries-states_dc.svg) |
| wk inc covid prop ed visits / 2025-2026 | [US](plots/covid_covid_prop_ed_visits_2025-2026/fans-US.svg) | [NC](plots/covid_covid_prop_ed_visits_2025-2026/fans-37.svg) | [Components](plots/covid_covid_prop_ed_visits_2025-2026/components-states_dc.svg) | [Relative WIS](plots/covid_covid_prop_ed_visits_2025-2026/relative-wis-states_dc.svg) | [Time series](plots/covid_covid_prop_ed_visits_2025-2026/timeseries-states_dc.svg) |
| wk inc rsv hosp / 2025-2026 | [US](plots/rsv_rsv_hosp_2025-2026/fans-US.svg) | [NC](plots/rsv_rsv_hosp_2025-2026/fans-37.svg) | [Components](plots/rsv_rsv_hosp_2025-2026/components-states_dc.svg) | [Relative WIS](plots/rsv_rsv_hosp_2025-2026/relative-wis-states_dc.svg) | [Time series](plots/rsv_rsv_hosp_2025-2026/timeseries-states_dc.svg) |
| wk inc rsv prop ed visits / 2025-2026 | [US](plots/rsv_rsv_prop_ed_visits_2025-2026/fans-US.svg) | [NC](plots/rsv_rsv_prop_ed_visits_2025-2026/fans-37.svg) | [Components](plots/rsv_rsv_prop_ed_visits_2025-2026/components-states_dc.svg) | [Relative WIS](plots/rsv_rsv_prop_ed_visits_2025-2026/relative-wis-states_dc.svg) | [Time series](plots/rsv_rsv_prop_ed_visits_2025-2026/timeseries-states_dc.svg) |

Each plot directory also contains US diagnostics and EpiBench-compatible score CSVs. Zero ensemble WIS yields undefined relative WIS, with the valid ratio count retained in the leaderboard.
