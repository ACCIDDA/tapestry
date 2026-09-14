# Canonical B0 experiments

**14 variants × three seeds = 42 CV runs / 126 season fits.** Experiment `b0-rebuilt` evaluates 2,436,466 forecast units across nine target/season comparisons, including the official ensembles. All fits and the full EpiBench evaluation completed.

## Best model

**`conv_h12` (`B0-46059582726c`) has the lowest mean influenza and three-admission objectives.** It uses a temporal convolution encoder, 12 weeks of history, fourth-root count transformation, geography and dynamics features, shared prediction heads, and the original decoder with latent dimension 16.

Across seeds 42/43/44, its influenza WIS ratio is **0.8916 ± 0.0217** and its admissions ratio is **0.9189 ± 0.0505** (mean ± sample SD). These are **10.8%** and **8.1%** below ensemble parity on the respective aggregate objectives.

`residual2` is a close alternative: influenza ratio 0.8924 ± 0.0203 and admissions ratio 0.9240 ± 0.0190. The influenza gap between the two variants is only 0.0008, much smaller than their seed variability; three seeds do not establish a decisive winner.

| Seed | conv_h12 influenza ratio | conv_h12 admissions ratio |
| --- | --- | --- |
| 42 | 0.8781 | 0.8639 |
| 43 | 0.8801 | 0.9299 |
| 44 | 0.9166 | 0.9630 |

## Interpretation

Lower WIS is better; ratio 1 means ensemble parity. The influenza objective is the geometric mean of model mean-WIS / ensemble mean-WIS across the three seasons and two geography groups, equally weighted. The admissions objective first gives each admission target equal weight. Reported configuration values average the three seed objectives. Percentage improvements refer to these aggregates, not a pooled raw WIS. ED forecasts are reported separately and do not enter either selection objective.

This is the canonical rebuilt-input comparison. CDC observations were downloaded September 14, 2026; the training date range is September 2023–August 29, 2026 and hub commits are pinned. It is a new frozen experiment, not exact reproduction of the September 4 CDC snapshots. The first two CV folds train on later seasons and all three seasons informed development. **These are exploratory results, not prospective validation.**

Scoring uses quantiles 0.025, 0.25, 0.5, 0.75, and 0.975. Bias is the signed EpiBench/scoringutils quantile bias score, not an error in admission counts. Coverage is the fraction of truth values inside the prediction interval. States/DC metrics average individual location forecast tasks; US is the native national prediction. Every candidate uses the same frozen tasks as its ensemble.

## Variant ranking

| Variant | Configuration | Flu ratio ± SD | Admissions ratio ± SD |
| --- | --- | --- | --- |
| `conv_h12` | `B0-46059582726c` | 0.8916 ± 0.0217 | 0.9189 ± 0.0505 |
| `residual2` | `B0-06c2ef97395a` | 0.8924 ± 0.0203 | 0.9240 ± 0.0190 |
| `latent32` | `B0-c17ce3825219` | 0.9415 ± 0.0569 | 0.9278 ± 0.0245 |
| `mlp_h12` | `B0-a3090f862c4c` | 0.9530 ± 0.0291 | 0.9973 ± 0.0656 |
| `balanced` | `B0-a00d850b4623` | 0.9582 ± 0.0660 | 0.9654 ± 0.0995 |
| `state_us` | `B0-b8911ee633a5` | 0.9628 ± 0.0193 | 1.0515 ± 0.0622 |
| `anchor` | `B0-d4ead99b1ce0` | 0.9631 ± 0.0193 | 1.0298 ± 0.0449 |
| `residual2_z32` | `B0-7b6e9f118eb0` | 0.9697 ± 0.0573 | 1.0323 ± 0.0442 |
| `mlp_h8` | `B0-9e3ee86287ef` | 0.9698 ± 0.0810 | 1.0301 ± 0.1411 |
| `flu_only` | `B0-9edad5f6c6f8` | 0.9837 ± 0.0718 | 1.1982 ± 0.2206 |
| `conv_h26` | `B0-5fafced4a267` | 1.0121 ± 0.0301 | 1.0071 ± 0.0688 |
| `mlp_h26` | `B0-9519898b9184` | 1.0398 ± 0.0513 | 1.0426 ± 0.0383 |
| `mlp_h26_dynamics` | `B0-344ee9ec8f26` | 1.1038 ± 0.0548 | 1.0809 ± 0.1005 |
| `baseline` | `B0-a4e53e1214ca` | 1.1356 ± 0.0414 | 1.2218 ± 0.0267 |

## Model differences

Fan labels use variant names and seed numbers. Seeds 42/43/44 repeat the same configuration with different random initialization. The reference `anchor` uses a multilayer perceptron (MLP), 12 weeks of history, fourth-root counts, geography and dynamics features, shared prediction heads, the original (`legacy`) decoder, latent dimension 16, and influenza-first loss weighting. The table lists changes from that reference; `conv` means temporal convolution, `state_us` means separate state and national heads, and `residual2` means a two-block residual decoder.

| Variant name | Differences from anchor |
| --- | --- |
| `anchor` | Reference configuration (settings below) |
| `balanced` | loss weighting: balanced_admissions |
| `baseline` | history (weeks): 8; count transform: raw; geography features: False; dynamics features: False |
| `conv_h12` | encoder: conv |
| `conv_h26` | encoder: conv; history (weeks): 26 |
| `flu_only` | loss weighting: flu_only |
| `latent32` | latent dimension: 32 |
| `mlp_h12` | dynamics features: False |
| `mlp_h26` | history (weeks): 26; dynamics features: False |
| `mlp_h26_dynamics` | history (weeks): 26 |
| `mlp_h8` | history (weeks): 8; dynamics features: False |
| `residual2` | decoder: residual2 |
| `residual2_z32` | decoder: residual2; latent dimension: 32 |
| `state_us` | prediction heads: state_us |

The official ensemble is the hub reference forecast, not one of these fitted variants.

## Best model versus ensemble

Means across seeds, with all four horizons included. Coverage columns are percentages. The full download includes seed SD, each horizon, and every variant.

| Target | Season | Geography | Model WIS | Ensemble WIS | WIS ratio | Bias | 50% coverage | 95% coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COVID-19 admissions | 2024-2025 | US | 725.13 | 799.99 | 0.906 | -0.541 | 45.5 | 94.5 |
| COVID-19 admissions | 2024-2025 | states_dc | 21.096 | 23.889 | 0.883 | -0.316 | 44.0 | 84.9 |
| COVID-19 admissions | 2025-2026 | US | 627.57 | 591.94 | 1.060 | +0.138 | 44.8 | 88.9 |
| COVID-19 admissions | 2025-2026 | states_dc | 15.529 | 14.217 | 1.092 | +0.101 | 36.4 | 77.0 |
| COVID-19 ED visits | 2025-2026 | US | 0.00056984 | 0.00021825 | 2.611 | +0.620 | 17.9 | 84.2 |
| COVID-19 ED visits | 2025-2026 | states_dc | 0.00065844 | 0.00041397 | 1.591 | +0.519 | 26.4 | 70.6 |
| Influenza admissions | 2023-2024 | US | 1116.2 | 1341.6 | 0.832 | +0.090 | 51.4 | 97.8 |
| Influenza admissions | 2023-2024 | states_dc | 29.574 | 32.591 | 0.907 | +0.053 | 41.4 | 83.3 |
| Influenza admissions | 2024-2025 | US | 3453.2 | 3933.5 | 0.878 | -0.271 | 49.1 | 88.3 |
| Influenza admissions | 2024-2025 | states_dc | 79.22 | 92.814 | 0.854 | -0.188 | 40.9 | 81.0 |
| Influenza admissions | 2025-2026 | US | 2473.6 | 2553 | 0.969 | +0.113 | 45.8 | 86.0 |
| Influenza admissions | 2025-2026 | states_dc | 57.013 | 59.719 | 0.955 | +0.089 | 34.2 | 75.1 |
| Influenza ED visits | 2025-2026 | US | 0.0054614 | 0.004895 | 1.116 | +0.229 | 35.7 | 79.5 |
| Influenza ED visits | 2025-2026 | states_dc | 0.0066028 | 0.0060074 | 1.099 | +0.159 | 30.8 | 69.0 |
| RSV admissions | 2025-2026 | US | 517.62 | 621.33 | 0.833 | -0.372 | 44.3 | 91.7 |
| RSV admissions | 2025-2026 | states_dc | 14.114 | 14.777 | 0.955 | -0.326 | 37.6 | 76.9 |
| RSV ED visits | 2025-2026 | US | 0.00032518 | 0.00040692 | 0.799 | +0.170 | 54.7 | 98.2 |
| RSV ED visits | 2025-2026 | states_dc | 0.00061837 | 0.00075422 | 0.820 | -0.025 | 44.1 | 85.5 |

`conv_h12` beats the ensemble in all six influenza admission season/geography cells. It loses on COVID admissions in 2025–26 and on influenza/COVID ED targets. Its state-level 95% influenza coverage is only 75.1–83.3%, so the aggregate win does not imply well-calibrated uncertainty.

## Each candidate against its matched control

Paired seed differences (candidate minus control), reported as mean ± sample SD. Negative is better. The anchor/baseline contrast changes several features together; the other contrasts isolate the documented change.

| Candidate | Control | Δ influenza objective | Δ admissions objective |
| --- | --- | --- | --- |
| `anchor` | `baseline` | -0.1725 ± 0.0253 | -0.1920 ± 0.0471 |
| `state_us` | `anchor` | -0.0003 ± 0.0072 | +0.0216 ± 0.0246 |
| `residual2` | `anchor` | -0.0707 ± 0.0338 | -0.1058 ± 0.0268 |
| `latent32` | `anchor` | -0.0215 ± 0.0541 | -0.1021 ± 0.0213 |
| `residual2_z32` | `residual2` | +0.0773 ± 0.0721 | +0.1083 ± 0.0327 |
| `residual2_z32` | `latent32` | +0.0281 ± 0.0555 | +0.1045 ± 0.0220 |
| `mlp_h8` | `mlp_h12` | +0.0168 ± 0.0591 | +0.0328 ± 0.1068 |
| `mlp_h12` | `anchor` | -0.0101 ± 0.0316 | -0.0326 ± 0.0382 |
| `mlp_h26` | `mlp_h12` | +0.0868 ± 0.0797 | +0.0454 ± 0.0576 |
| `balanced` | `anchor` | -0.0048 ± 0.0752 | -0.0644 ± 0.1143 |
| `flu_only` | `anchor` | +0.0206 ± 0.0675 | +0.1684 ± 0.1973 |
| `conv_h12` | `anchor` | -0.0715 ± 0.0039 | -0.1109 ± 0.0156 |
| `mlp_h26_dynamics` | `mlp_h26` | +0.0640 ± 0.0943 | +0.0383 ± 0.0727 |
| `conv_h26` | `mlp_h26_dynamics` | -0.0917 ± 0.0421 | -0.0738 ± 0.0519 |

## Heads and uncertainty

For the following diagnostic summary, each influenza season receives equal weight after seed averaging. WIS ratios here are arithmetic means across seasons, so they differ from the geometric-mean selection objective.

| Variant | Geography | Mean WIS ratio | 50% coverage | 95% coverage |
| --- | --- | --- | --- | --- |
| anchor | US | 0.965 | 38.2% | 86.7% |
| anchor | states_dc | 0.965 | 34.0% | 75.7% |
| baseline | US | 1.512 | 17.2% | 51.9% |
| baseline | states_dc | 0.869 | 43.6% | 86.3% |
| conv_h12 | US | 0.893 | 48.8% | 90.7% |
| conv_h12 | states_dc | 0.905 | 38.8% | 79.8% |
| residual2 | US | 0.877 | 46.2% | 90.7% |
| residual2 | states_dc | 0.910 | 36.0% | 78.9% |
| state_us | US | 0.982 | 33.4% | 78.7% |
| state_us | states_dc | 0.951 | 37.4% | 80.4% |

**Separate heads do not resolve the state/US tradeoff.** Relative to the anchor, `state_us` slightly improves state influenza WIS and coverage, but worsens US WIS and reduces US 95% coverage from 86.7% to 78.7%. Its influenza objective is essentially unchanged, and its admissions objective is worse. The baseline still has better average state influenza WIS than these feature-rich variants, while its national forecasts are much worse.

**The richer decoder helps, but does not fix undercoverage.** At latent dimension 16, `residual2` improves both objectives and moves influenza coverage toward nominal levels at state and US scales. State 95% coverage rises from 75.7% to 78.9%; US coverage rises from 86.7% to 90.7%. Combining residual depth with latent dimension 32 reverses much of that gain: `residual2_z32` has poorer WIS objectives than either `residual2` or `latent32` alone.

## What to combine next

Test **`conv_h12` + `residual2` at latent dimension 16**, at all three seeds, against both individual variants. Both independently improve the anchor, but their combination remains untested. Balanced admission weights are a secondary isolated addition to that recipe. Keep the 12-week history: 26-week variants did not improve the selection objectives. There is no current evidence to add separate heads or combine the deeper decoder with latent dimension 32. Calibration and spatial attention remain outside this suite.

## Downloads

- [Metrics by variant, target, season, geography, and horizon: seed means and SD](../assets/b0_configuration_comparison/seed_summary.csv)
- [Paired matched-control objective differences](../assets/b0_configuration_comparison/matched_controls.csv)
- [Paired matched-control metric differences for every evaluation cell](../assets/b0_configuration_comparison/matched_control_details.csv)
- [Configuration ranking](../assets/b0_configuration_comparison/configuration_ranking.csv) · [Individual seeds](../assets/b0_configuration_comparison/run_ranking.csv)
- [Per-seed leaderboard, including ensembles](../assets/b0_configuration_comparison/leaderboard.csv)
- [Configuration definitions](../assets/b0_configuration_comparison/configurations.json) · [Evaluation provenance](../assets/b0_configuration_comparison/manifest.json)

The detailed tables contain WIS, bias, 50%/95% coverage, and WIS components for horizons 0–3 and all horizons combined. Horizon 0 means the first future week. Missing target/seasons are unavailable in the pinned ensemble-supported task sets, rather than failed runs.

## Figures by target and season

| Target | Season | Figures |
| --- | --- | --- |
| Influenza admissions | 2023-2024 | [Eight figures](b0-comparison/flusight_flu_hosp_2023-2024.md) |
| Influenza admissions | 2024-2025 | [Eight figures](b0-comparison/flusight_flu_hosp_2024-2025.md) |
| Influenza admissions | 2025-2026 | [Eight figures](b0-comparison/flusight_flu_hosp_2025-2026.md) |
| Influenza ED visits | 2025-2026 | [Eight figures](b0-comparison/flusight_flu_prop_ed_visits_2025-2026.md) |
| COVID-19 admissions | 2024-2025 | [Eight figures](b0-comparison/covid_covid_hosp_2024-2025.md) |
| COVID-19 admissions | 2025-2026 | [Eight figures](b0-comparison/covid_covid_hosp_2025-2026.md) |
| COVID-19 ED visits | 2025-2026 | [Eight figures](b0-comparison/covid_covid_prop_ed_visits_2025-2026.md) |
| RSV admissions | 2025-2026 | [Eight figures](b0-comparison/rsv_rsv_hosp_2025-2026.md) |
| RSV ED visits | 2025-2026 | [Eight figures](b0-comparison/rsv_rsv_prop_ed_visits_2025-2026.md) |

Projection fans show the three best configurations across all six targets, ranked by arithmetic mean seed score, plus the official ensemble (light blue) and the best configuration for the displayed target/season (light red). Each seed score uses the geometric mean of WIS ratios, weighting targets equally, then available season/geography cells equally. The season winner averages seed scores using both geography groups. Fans display the middle-performing seed: the median by overall score for the top three, or by target/season score for the season winner. Identical representative runs appear once; different middle seeds of the same configuration appear separately. Score ties use seed number; an even seed count uses the upper middle. Other figures show all 42 runs and the ensemble. Configuration IDs map to names in the ranking above; the `-s42`, `-s43`, and `-s44` suffixes identify seeds. Projection fans illustrate US and North Carolina. Relative-WIS plots average per-task ratios, whereas the tables use ratios of mean WIS.

### All-target configuration ranking for fans

| Variant | Mean seed score ± SD | Middle seed |
| --- | --- | --- |
| residual2 | 1.0197 ± 0.0103 | 44 |
| latent32 | 1.0293 ± 0.0311 | 43 |
| balanced | 1.0519 ± 0.0445 | 44 |
| anchor | 1.0572 ± 0.0264 | 43 |
| conv_h12 | 1.0587 ± 0.0247 | 43 |
| conv_h26 | 1.0604 ± 0.0277 | 43 |
| residual2_z32 | 1.0618 ± 0.0037 | 44 |
| mlp_h12 | 1.0640 ± 0.0092 | 44 |
| mlp_h8 | 1.0831 ± 0.0540 | 42 |
| mlp_h26 | 1.0852 ± 0.0359 | 43 |
| mlp_h26_dynamics | 1.0927 ± 0.0561 | 43 |
| state_us | 1.1625 ± 0.0432 | 43 |
| baseline | 1.2407 ± 0.0793 | 42 |
| flu_only | 1.3954 ± 0.1028 | 42 |

## Reproduction

Full artifacts are in `data/experiments/b0-rebuilt/comparison-da7685987135`. See [Longleaf setup](../longleaf-setup.md) for input rebuilding, GPU arrays, parallel scoring, and resume commands. Publish the completed comparison with:

```bash
.venv/bin/python scripts/publish_evaluation_docs.py --comparison data/experiments/b0-rebuilt/comparison-da7685987135
```

The pipeline completed its built-in task-support, finite-metric, and relative-WIS checks. This publication summarizes saved scores; it does not rerun fitting or scoring.
