# B0 evaluation with EpiBenchmark

**15 runs, nine configurations, and nine target/season comparisons.** This report includes all 14 completed sweep runs and the original B0 CV run. The full `epibench score --config-path` pipeline evaluated 906,592 forecasts on identical frozen hub tasks, including freshly rescored official ensembles. EpiBench also generated the diagnostics. Each target/season page below embeds its eight figures.

The original Python season-CV/persistence report has been removed. Its saved forecasts are included here under the original configuration identity. See [EpiBench integration and remaining gaps](../workflows/configuration-evaluation.md#what-epibench-still-needs-for-a-direct-frozen-benchmark).

Emily’s ten configs provide forecast dates and vintage inputs; see the [config review](../workflows/emily-configs.md). Our challenges remain unversioned custom scoring configs. This report retains the existing nine frozen task sets, finalized truth and finalized-data fits; Emily’s challenge ground truth is not used. Historical source archives are preserved; current exports and all new predictions save only five quantiles.

## Interpretation

These are finalized-data retrospective cross-validation results. The first two folds train on later seasons; all folds were used for exploratory selection. Seed counts differ, so the ranking is descriptive rather than an untouched validation result.

Lower objectives are better; one means parity with the official hub ensemble. The influenza objective is the geometric mean of mean-WIS/ensemble-mean-WIS ratios, equally weighting each available season and US versus states/DC. The admissions objective additionally weights the three admission targets equally. Configuration scores average individual seed objectives. The relative-WIS figures instead average per-forecast ratios; these statistics need not agree.

Scores below use five quantiles: **0.025, 0.25, 0.5, 0.75, 0.975**, and the official ensemble on the frozen tasks. The geometric-mean objectives are Tapestry aggregations of EpiBench scores. They are not the bundled EpiBench challenge scorecards, which use different dates, hub baselines, and challenge-specific task sets.

## Configuration ranking

| Configuration | Example run | Seeds | Flu objective | Flu seed SD | Admissions objective | Admissions seed SD |
|---|---|---:|---:|---:|---:|---:|
| `B0-b1e265a36c11` | fourth_root_geo_s42 | 1 | 0.9135 | — | 0.9215 | — |
| `B0-fa4e4645314c` | fourth_root_geo_h12_dynamics_s44 | 3 | 0.9323 | 0.0156 | 0.9176 | 0.0321 |
| `B0-c20800f1557e` | fourth_root_geo_h12_balanced_admissions_s42 | 1 | 0.9338 | — | 0.9030 | — |
| `B0-41878487e0fd` | sqrt_geo_s42 | 1 | 0.9419 | — | 0.9329 | — |
| `B0-4d0946a0dfcf` | fourth_root_geo_h12_flu_only_s42 | 1 | 0.9567 | — | 1.5080 | — |
| `B0-96f18bffffa1` | fourth_root_geo_h12_s44 | 3 | 1.0236 | 0.0993 | 0.9760 | 0.0607 |
| `B0-8afe05f2b10e` | fourth_root_geo_h26_s42 | 1 | 1.0821 | — | 1.0890 | — |
| `B0-2a157841d7e4` | b0_season_cv_20260913 | 1 | 1.0880 | — | 1.1898 | — |
| `B0-323fbd9b1c6f` | baseline_s44 | 3 | 1.1197 | 0.0391 | 1.2194 | 0.0257 |

A dash denotes an undefined sample SD for a single seed. Lowest mean influenza objective: `B0-b1e265a36c11` (1 seed(s)). Best configuration tested at multiple seeds: `B0-fa4e4645314c`. These are five-quantile scores; WIS values from earlier 23-quantile reports are not interchangeable.

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

Four-week projections are retained locally in `data/evaluation/b0_epibench_five_quantiles/hubverse/<hub>/model-output/<model_id>/`, with 6,750 CSV files and 6,750 Parquet companions containing the same 13,618,800 quantile rows, exactly five per forecast. These large forecast archives are not copied into the documentation. The portable figures and ranking downloads above are included in the documentation build.

See the [configuration evaluation workflow](../workflows/configuration-evaluation.md) to score future runs and refresh this report.

## Validation

All 906,592 unique per-forecast scores passed the frozen-support audit. Every candidate and ensemble was processed by the full EpiBench scoring command. Maximum WIS error against an independent five-quantile calculation: 7.28e-12. Relative WIS was checked against matched ensemble scores, including undefined zero denominators. Median AE and 50/95% coverage match the previous evaluation; WIS was recomputed on the new grid. The report includes 72 regenerated SVG figures. See the validation download for the exact audit results.
