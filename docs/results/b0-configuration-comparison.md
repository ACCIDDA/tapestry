# Canonical B0 experiments

**14 variants × three seeds = 42 CV runs / 126 season fits.** Experiment `b0-rebuilt` evaluates 2,436,466 forecast units across nine target/season comparisons, including the official ensembles. All fits and the full EpiBench evaluation completed. These are experiment counts; the ranking compares 14 configurations by their mean scores across three seeds.

## Best model

**`residual2` (`B0-06c2ef97395a`) has the lowest all-target score averaged across seeds.** The ranking includes all six admission and ED targets. Its settings are: encoder: mlp; history (weeks): 12; count transform: fourth_root; geography features: True; dynamics features: True; prediction heads: shared; decoder: residual2; latent dimension: 16; loss weighting: influenza_first.

Its all-target WIS ratio is **1.0197 ± 0.0103** (mean ± sample SD). This is **2.0% above** ensemble parity. Being the best fitted configuration does not imply beating the official ensembles overall. The overall fans illustrate its middle-performing seed, **44**.

The next configurations are `latent32` (1.0293 ± 0.0311), `balanced` (1.0519 ± 0.0445). Three seeds do not establish a decisive performance difference.

| Seed | residual2 all-target ratio | Secondary influenza ratio | Secondary admissions ratio |
| --- | --- | --- | --- |
| 42 | 1.0295 | 0.9159 | 0.9101 |
| 43 | 1.0089 | 0.8817 | 0.9162 |
| 44 | 1.0205 | 0.8797 | 0.9457 |

## Interpretation

Lower WIS is better; ratio 1 means ensemble parity. The primary all-target score first computes each seed’s geometric mean WIS ratio, weighting the six targets equally, then available season/geography cells equally within each target. Configuration scores are arithmetic means of these seed scores. The secondary influenza objective is the geometric mean of model mean-WIS / ensemble mean-WIS across the three seasons and two geography groups, equally weighted. The admissions objective first gives each admission target equal weight. Reported configuration values average the three seed objectives. Percentage improvements refer to these aggregates, not a pooled raw WIS. ED forecasts enter the primary all-target ranking; they are excluded only from the two secondary objectives. `conv_h12` leads the influenza objective and `conv_h12` leads the admissions objective; neither secondary objective determines the overall winner.

This is the canonical rebuilt-input comparison. CDC observations were downloaded September 14, 2026; the training date range is September 2023–August 29, 2026 and hub commits are pinned. It is a new frozen experiment, not exact reproduction of the September 4 CDC snapshots. The first two CV folds train on later seasons and all three seasons informed development. **These are exploratory results, not prospective validation.**

Scoring uses quantiles 0.025, 0.25, 0.5, 0.75, and 0.975. Bias is the signed EpiBench/scoringutils quantile bias score, not an error in admission counts. Coverage is the fraction of truth values inside the prediction interval. States/DC metrics average individual location forecast tasks; US is the native national prediction. Every candidate uses the same frozen tasks as its ensemble.

## Variant ranking

Ordered by all-target mean across seeds. SD is sample variability across the three seeds.

| Variant | Configuration | All-target ratio ± SD | Middle seed | Secondary flu ratio ± SD | Secondary admissions ratio ± SD |
| --- | --- | --- | --- | --- | --- |
| `residual2` | `B0-06c2ef97395a` | 1.0197 ± 0.0103 | 44 | 0.8924 ± 0.0203 | 0.9240 ± 0.0190 |
| `latent32` | `B0-c17ce3825219` | 1.0293 ± 0.0311 | 43 | 0.9415 ± 0.0569 | 0.9278 ± 0.0245 |
| `balanced` | `B0-a00d850b4623` | 1.0519 ± 0.0445 | 44 | 0.9582 ± 0.0660 | 0.9654 ± 0.0995 |
| `anchor` | `B0-d4ead99b1ce0` | 1.0572 ± 0.0264 | 43 | 0.9631 ± 0.0193 | 1.0298 ± 0.0449 |
| `conv_h12` | `B0-46059582726c` | 1.0587 ± 0.0247 | 43 | 0.8916 ± 0.0217 | 0.9189 ± 0.0505 |
| `conv_h26` | `B0-5fafced4a267` | 1.0604 ± 0.0277 | 43 | 1.0121 ± 0.0301 | 1.0071 ± 0.0688 |
| `residual2_z32` | `B0-7b6e9f118eb0` | 1.0618 ± 0.0037 | 44 | 0.9697 ± 0.0573 | 1.0323 ± 0.0442 |
| `mlp_h12` | `B0-a3090f862c4c` | 1.0640 ± 0.0092 | 44 | 0.9530 ± 0.0291 | 0.9973 ± 0.0656 |
| `mlp_h8` | `B0-9e3ee86287ef` | 1.0831 ± 0.0540 | 42 | 0.9698 ± 0.0810 | 1.0301 ± 0.1411 |
| `mlp_h26` | `B0-9519898b9184` | 1.0852 ± 0.0359 | 43 | 1.0398 ± 0.0513 | 1.0426 ± 0.0383 |
| `mlp_h26_dynamics` | `B0-344ee9ec8f26` | 1.0927 ± 0.0561 | 43 | 1.1038 ± 0.0548 | 1.0809 ± 0.1005 |
| `state_us` | `B0-b8911ee633a5` | 1.1625 ± 0.0432 | 43 | 0.9628 ± 0.0193 | 1.0515 ± 0.0622 |
| `baseline` | `B0-a4e53e1214ca` | 1.2407 ± 0.0793 | 42 | 1.1356 ± 0.0414 | 1.2218 ± 0.0267 |
| `flu_only` | `B0-9edad5f6c6f8` | 1.3954 ± 0.1028 | 42 | 0.9837 ± 0.0718 | 1.1982 ± 0.2206 |

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

The all-target winner is `residual2`. Means across seeds, with all four horizons included. Coverage columns are percentages. The full download includes seed SD, each horizon, and every variant.

| Target | Season | Geography | Model WIS | Ensemble WIS | WIS ratio | Bias | 50% coverage | 95% coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COVID-19 admissions | 2024-2025 | US | 784.83 | 799.99 | 0.981 | -0.592 | 41.0 | 99.5 |
| COVID-19 admissions | 2024-2025 | states_dc | 22.215 | 23.889 | 0.930 | -0.348 | 43.4 | 87.2 |
| COVID-19 admissions | 2025-2026 | US | 579.71 | 591.94 | 0.979 | +0.247 | 57.1 | 93.1 |
| COVID-19 admissions | 2025-2026 | states_dc | 14.716 | 14.217 | 1.035 | +0.127 | 38.9 | 80.5 |
| COVID-19 ED visits | 2025-2026 | US | 0.00039198 | 0.00021825 | 1.796 | +0.532 | 40.9 | 100.0 |
| COVID-19 ED visits | 2025-2026 | states_dc | 0.00049828 | 0.00041397 | 1.204 | +0.374 | 42.1 | 82.7 |
| Influenza admissions | 2023-2024 | US | 1174.9 | 1341.6 | 0.876 | -0.010 | 42.2 | 91.4 |
| Influenza admissions | 2023-2024 | states_dc | 30.401 | 32.591 | 0.933 | +0.018 | 36.5 | 79.8 |
| Influenza admissions | 2024-2025 | US | 3353.5 | 3933.5 | 0.853 | -0.514 | 39.5 | 88.3 |
| Influenza admissions | 2024-2025 | states_dc | 81.336 | 92.814 | 0.876 | -0.356 | 37.3 | 79.2 |
| Influenza admissions | 2025-2026 | US | 2306.8 | 2553 | 0.904 | -0.073 | 56.8 | 92.6 |
| Influenza admissions | 2025-2026 | states_dc | 54.99 | 59.719 | 0.921 | -0.035 | 34.3 | 77.9 |
| Influenza ED visits | 2025-2026 | US | 0.005469 | 0.004895 | 1.117 | +0.187 | 45.2 | 87.2 |
| Influenza ED visits | 2025-2026 | states_dc | 0.0065405 | 0.0060074 | 1.089 | +0.103 | 36.0 | 74.3 |
| RSV admissions | 2025-2026 | US | 524.77 | 621.33 | 0.845 | -0.288 | 51.7 | 96.8 |
| RSV admissions | 2025-2026 | states_dc | 14.319 | 14.777 | 0.969 | -0.271 | 36.7 | 79.8 |
| RSV ED visits | 2025-2026 | US | 0.00036281 | 0.00040692 | 0.892 | +0.061 | 57.1 | 100.0 |
| RSV ED visits | 2025-2026 | states_dc | 0.00065709 | 0.00075422 | 0.871 | -0.152 | 43.7 | 85.5 |

Target-specific gains and losses are shown above. The overall ranking gives each target equal weight; it does not imply a win in every target/season or well-calibrated uncertainty.

## Each candidate against its matched control

Paired seed differences (candidate minus control), reported as mean ± sample SD. Negative is better. The anchor/baseline contrast changes several features together; the other contrasts isolate the documented change.

| Candidate | Control | Δ all-target objective | Δ secondary influenza objective | Δ secondary admissions objective |
| --- | --- | --- | --- | --- |
| `anchor` | `baseline` | -0.1834 ± 0.1002 | -0.1725 ± 0.0253 | -0.1920 ± 0.0471 |
| `state_us` | `anchor` | +0.1053 ± 0.0230 | -0.0003 ± 0.0072 | +0.0216 ± 0.0246 |
| `residual2` | `anchor` | -0.0376 ± 0.0284 | -0.0707 ± 0.0338 | -0.1058 ± 0.0268 |
| `latent32` | `anchor` | -0.0280 ± 0.0181 | -0.0215 ± 0.0541 | -0.1021 ± 0.0213 |
| `residual2_z32` | `residual2` | +0.0421 ± 0.0067 | +0.0773 ± 0.0721 | +0.1083 ± 0.0327 |
| `residual2_z32` | `latent32` | +0.0325 ± 0.0338 | +0.0281 ± 0.0555 | +0.1045 ± 0.0220 |
| `mlp_h8` | `mlp_h12` | +0.0192 ± 0.0606 | +0.0168 ± 0.0591 | +0.0328 ± 0.1068 |
| `mlp_h12` | `anchor` | +0.0067 ± 0.0305 | -0.0101 ± 0.0316 | -0.0326 ± 0.0382 |
| `mlp_h26` | `mlp_h12` | +0.0212 ± 0.0402 | +0.0868 ± 0.0797 | +0.0454 ± 0.0576 |
| `balanced` | `anchor` | -0.0054 ± 0.0560 | -0.0048 ± 0.0752 | -0.0644 ± 0.1143 |
| `flu_only` | `anchor` | +0.3381 ± 0.0769 | +0.0206 ± 0.0675 | +0.1684 ± 0.1973 |
| `conv_h12` | `anchor` | +0.0015 ± 0.0021 | -0.0715 ± 0.0039 | -0.1109 ± 0.0156 |
| `mlp_h26_dynamics` | `mlp_h26` | +0.0075 ± 0.0210 | +0.0640 ± 0.0943 | +0.0383 ± 0.0727 |
| `conv_h26` | `mlp_h26_dynamics` | -0.0323 ± 0.0286 | -0.0917 ± 0.0421 | -0.0738 ± 0.0519 |

## Heads and uncertainty

This section examines influenza admissions only, not the overall ranking. Each influenza season receives equal weight after seed averaging. WIS ratios here are arithmetic means across seasons, distinct from the primary all-target objective.

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

**The richer decoder helps, but does not fix undercoverage.** At latent dimension 16, `residual2` improves both secondary admission objectives and moves influenza coverage toward nominal levels at state and US scales. State 95% coverage rises from 75.7% to 78.9%; US coverage rises from 86.7% to 90.7%. Combining residual depth with latent dimension 32 reverses much of that gain: `residual2_z32` has poorer WIS objectives than either `residual2` or `latent32` alone.

## What to combine next

Use **`residual2`** as the overall reference for follow-up experiments. Compare any combination of decoder, latent-dimension, weighting, or encoder changes against its individual components at all three seeds using the all-target mean. `conv_h12` remains useful as a secondary admission-focused comparator. The existing results do not measure the performance of untested combinations. Calibration and spatial attention remain outside this suite.

## Downloads

- [Metrics by variant, target, season, geography, and horizon: seed means and SD](../assets/b0_configuration_comparison/seed_summary.csv)
- [Paired matched-control objective differences](../assets/b0_configuration_comparison/matched_controls.csv)
- [Paired matched-control metric differences for every evaluation cell](../assets/b0_configuration_comparison/matched_control_details.csv)
- [All-target configuration ranking, means and SD across seeds](../assets/b0_configuration_comparison/configuration_ranking.csv) · [Individual seed scores](../assets/b0_configuration_comparison/run_ranking.csv)
- [Per-seed leaderboard, including ensembles](../assets/b0_configuration_comparison/leaderboard.csv)
- [Configuration definitions](../assets/b0_configuration_comparison/configurations.json) · [Evaluation provenance](../assets/b0_configuration_comparison/manifest.json)

The detailed tables contain WIS, bias, 50%/95% coverage, and WIS components for horizons 0–3 and all horizons combined. Horizon 0 means the first future week. Missing target/seasons are unavailable in the pinned ensemble-supported task sets, rather than failed runs.

Ranking downloads are recomputed from the saved leaderboard using the current all-target rule. The original scoring manifest records the unchanged forecast and evaluation provenance.

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
