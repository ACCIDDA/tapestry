# B0 configuration comparison

**15 runs, nine configurations, and nine target/season comparisons.** This report includes all 14 completed sweep runs and the original B0 CV run. R scoringutils evaluated 906,592 forecasts on identical frozen hub tasks; EpiBench generated the diagnostics. Each target/season page below embeds its eight figures.

## Interpretation

These are finalized-data retrospective cross-validation results. The first two folds train on later seasons; all folds were used for exploratory selection. Seed counts differ, so the ranking is descriptive rather than an untouched validation result.

Lower objectives are better; one means parity with the official hub ensemble. The influenza objective is the geometric mean of mean-WIS/ensemble-mean-WIS ratios, equally weighting each available season and US versus states/DC. The admissions objective additionally weights the three admission targets equally. Configuration scores average individual seed objectives. The relative-WIS figures instead average per-forecast ratios; these statistics need not agree.

## Configuration ranking

| Configuration | Example run | Seeds | Flu objective | Flu seed SD | Admissions objective | Admissions seed SD |
|---|---|---:|---:|---:|---:|---:|
| `B0-b1e265a36c11` | fourth_root_geo_s42 | 1 | 0.9211 | — | 0.9227 | — |
| `B0-fa4e4645314c` | fourth_root_geo_h12_dynamics_s44 | 3 | 0.9371 | 0.0186 | 0.9171 | 0.0322 |
| `B0-c20800f1557e` | fourth_root_geo_h12_balanced_admissions_s42 | 1 | 0.9381 | — | 0.9019 | — |
| `B0-41878487e0fd` | sqrt_geo_s42 | 1 | 0.9441 | — | 0.9318 | — |
| `B0-4d0946a0dfcf` | fourth_root_geo_h12_flu_only_s42 | 1 | 0.9601 | — | 1.4872 | — |
| `B0-96f18bffffa1` | fourth_root_geo_h12_s44 | 3 | 1.0281 | 0.1002 | 0.9750 | 0.0615 |
| `B0-8afe05f2b10e` | fourth_root_geo_h26_s42 | 1 | 1.0725 | — | 1.0829 | — |
| `B0-2a157841d7e4` | b0_season_cv_20260913 | 1 | 1.0731 | — | 1.1760 | — |
| `B0-323fbd9b1c6f` | baseline_s44 | 3 | 1.1058 | 0.0400 | 1.2057 | 0.0259 |

A dash denotes an undefined sample SD for a single seed. The fourth-root/geography configuration has the lowest mean influenza objective, but was tested at one seed. Among configurations repeated at three seeds, the twelve-week dynamics configuration has the lowest mean influenza objective. See the individual-run ranking to distinguish a strong seed from a stable formulation.

[Download configuration rankings](../assets/b0_configuration_comparison/configuration_ranking.csv) · [Individual runs](../assets/b0_configuration_comparison/run_ranking.csv) · [Detailed leaderboard](../assets/b0_configuration_comparison/leaderboard.csv)

## Figures by target and season

Every page contains national and North Carolina projection fans, plus WIS components, relative-WIS heatmaps, and WIS over time separately for US and states/DC. Click any figure to open its full-resolution SVG.

| Target | Season | Figures |
|---|---|---|
| Influenza admissions | 2023-2024 | [All eight figures](b0-comparison/flusight_flu_hosp_2023-2024.md) |
| Influenza admissions | 2024-2025 | [All eight figures](b0-comparison/flusight_flu_hosp_2024-2025.md) |
| Influenza admissions | 2025-2026 | [All eight figures](b0-comparison/flusight_flu_hosp_2025-2026.md) |
| Influenza ED visits | 2025-2026 | [All eight figures](b0-comparison/flusight_flu_prop_ed_visits_2025-2026.md) |
| COVID-19 admissions | 2024-2025 | [All eight figures](b0-comparison/covid_covid_hosp_2024-2025.md) |
| COVID-19 admissions | 2025-2026 | [All eight figures](b0-comparison/covid_covid_hosp_2025-2026.md) |
| COVID-19 ED visits | 2025-2026 | [All eight figures](b0-comparison/covid_covid_prop_ed_visits_2025-2026.md) |
| RSV admissions | 2025-2026 | [All eight figures](b0-comparison/rsv_rsv_hosp_2025-2026.md) |
| RSV ED visits | 2025-2026 | [All eight figures](b0-comparison/rsv_rsv_prop_ed_visits_2025-2026.md) |

### Example: influenza admissions, 2025–2026

The most recent influenza admissions season illustrates the plots; the full set of targets and seasons is linked above. Lower relative WIS is better.

[![States/DC relative WIS by model and horizon](../assets/b0_configuration_comparison/flusight_flu_hosp_2025-2026/relative-wis-states_dc.svg){ loading=lazy }](../assets/b0_configuration_comparison/flusight_flu_hosp_2025-2026/relative-wis-states_dc.svg)

The North Carolina fans below show the four-week projections for each run and the ensemble.

[![North Carolina four-week projection fans](../assets/b0_configuration_comparison/flusight_flu_hosp_2025-2026/fans-37.svg){ loading=lazy }](../assets/b0_configuration_comparison/flusight_flu_hosp_2025-2026/fans-37.svg)

## Identifiers, exports and reproduction

`B0-<12 hexadecimal characters>` identifies a configuration; `-s42` identifies its seed-42 realization. Configuration identity includes settings, dataset and model-source hashes; future settings participate automatically. Source revisions remain distinct even if their visible settings match. The original B0 is therefore separate from the later baseline.

[Configuration key](../assets/b0_configuration_comparison/configurations.csv) · [Complete configuration definitions](../assets/b0_configuration_comparison/configurations.json) · [Evaluation provenance](../assets/b0_configuration_comparison/manifest.json) · [Validation summary](../assets/b0_configuration_comparison/validation.json)

Four-week projections are retained locally in `data/evaluation/b0_configuration_comparison/hubverse/<hub>/model-output/<model_id>/`, with 6,750 CSV files and 6,750 Parquet companions containing the same 62,646,480 quantile rows. These large forecast archives are not copied into the documentation. The portable figures and ranking downloads above are included in the documentation build.

See the [configuration evaluation workflow](../workflows/configuration-evaluation.md) to score future runs and refresh this report.

## Validation

Nine tests passed, including export–score–rank–plot integration and CSV/Parquet agreement. All 14 sweep rescores exactly match the earlier WIS values. The comparison has 906,592 unique scored forecasts, identical support in all nine cases, and 72 valid SVG figures. EpiBench’s own forecast loader accepted an exported CSV and preserved zero-padded FIPS codes.
