# B7 recipe ensembles against System2 on real Wednesday reports

Written 7 October 2026, 22:55 EDT. No model was retrained.

## Setup

**Shared evaluation.** Both sides are scored on the same evaluation as the B6 System2 selection:
- archived Wednesday reports, with the existing finalized-value substitutions;
- each model's own saved correction;
- the frozen FluSight ensemble task grid and frozen truth:
  - 2024–25: reference dates 23 November 2024 to 31 May 2025;
  - 2025–26: 22 November 2025 to 30 May 2026;
  - horizons 0–3.

States/DC are pooled, excluding the US and Puerto Rico. WIS is in raw units, and the ratio is
pooled model WIS divided by FluSight-ensemble WIS on the same tasks. Lower is better. The
two seasons count equally.

The method is that of `scripts/compare_b6_b7.py` (another agent's
[b7-submission-comparison-20261007](../../b7-submission-comparison-20261007/index.md)). This page
adds A_blocks3 and B5 ILI.

**System2.** These are the saved B6 fold forecasts:
- evaluation on 2024–25 trains on 2022–23, 2023–24 and 2025–26;
- evaluation on 2025–26 trains on 2022–23, 2023–24 and 2024–25;
- each recipe keeps its B6 input treatment (admission histories artificially revised, ED mostly at
  latest values, one FluSurv member);
- the predictive distributions are mixed.

**B7.** The B7-fast fold checkpoints for the same two training splits:
- training admission and ED histories artificially revised with 2025–26 errors;
- replayed on the same archived reports with each fold's saved synthetic correction tree
  (`scripts/replay_b7_reported.py`, replaced on 8 October 2026 by `planner replay --inputs reported`);
- the B_width256 replay is identical to the other agent's independent replay.

**Seeds and labels.** Seeds 44 and 45 unless "5 seeds". All models learn latest future flu
admissions and ED.

## Results

| Ensemble | Log, 2024–25 | Log, 2025–26 | Log, mean | Log ratio to Hub ensemble | Native, mean | US log | ED 2025–26 ×1000 | 95% coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B7 B5 ILI + B_width256 | 0.2533 | **0.2749** | **0.2641** | **0.788** | 62.9 | 0.155 | **5.04** | 0.927 |
| B7 A_blocks3 + B5 ILI + B_width256 | 0.2499 | 0.2785 | 0.2642 | 0.788 | 62.1 | 0.153 | **5.04** | **0.932** |
| B7 A_blocks3 + B_width256 | **0.2487** | 0.2824 | 0.2656 | 0.792 | 64.0 | 0.157 | 5.12 | 0.923 |
| B7 A_width192 + B_width256 | 0.2535 | 0.2810 | 0.2673 | 0.797 | 64.5 | 0.156 | 5.15 | 0.912 |
| B7 B_width256 alone | 0.2591 | 0.2797 | 0.2694 | 0.803 | 69.1 | 0.172 | 5.38 | 0.888 |
| B7 B5 ILI alone | 0.2626 | 0.2774 | 0.2700 | 0.805 | 61.9 | 0.158 | 5.12 | 0.908 |
| B7 A_blocks3 alone | 0.2500 | 0.2926 | 0.2713 | 0.810 | 63.6 | 0.159 | 5.30 | 0.913 |
| System2 nine, 2 seeds | 0.2605 | 0.2836 | 0.2720 | 0.811 | **59.5** | **0.149** | 5.09 | 0.924 |
| System2 eight (no FluSurv), 2 seeds | 0.2615 | 0.2859 | 0.2737 | 0.816 | 59.6 | 0.150 | 5.09 | 0.918 |
| System2 nine, 5 seeds | 0.2640 | 0.2843 | 0.2742 | 0.818 | 60.7 | 0.152 | 5.10 | 0.922 |
| System2 eight (no FluSurv), 5 seeds | 0.2647 | 0.2860 | 0.2753 | 0.821 | 60.6 | 0.152 | 5.10 | 0.918 |
| FluSight ensemble | 0.3422 | 0.3292 | 0.3357 | 1 | 81.9 | 0.221 | 6.52 | 0.861 |

"Log" is state/DC WIS on log(x+1) admissions, the closest proxy for CDC's states-only log ranking.
"Native" is state/DC admission-count WIS. Coverage is for state/DC admissions.

## Reading

- On log admissions, every B7 two- or three-recipe mixture beats System2 in both seasons:
  - with matched seeds, by 1.7–2.9% (0.2641–0.2673 against 0.2720);
  - against the five-seed System2, by 2.5–3.7%.
- On admission counts, System2 is better by 2–8% (2.4–6.3% against five seeds, 4.4–8.3% against two).
- On ED and coverage, the two are level or B7 is slightly better.
- **The 2025–26 gap is partly inflated.** B7's correction trees learned 2025–26's own reporting
  errors, and here they correct 2025–26's real reports. The 2024–25 season has no such leak, and
  B7's log advantage over the five-seed System2 there is larger (4.0–5.8%).
- **The three B7 mixtures with B5 ILI or A_blocks3 differ by up to 0.6% on log**, which is within seed
  noise. The three-recipe mixture is the best balance: it ties on log and is the best B7 option on
  counts, ED and coverage.
- **Seed cohorts move System2 by about 1%:** two seeds scored 0.2720, five seeds 0.2742.

## Assumptions and limits

- Both seasons informed recipe selection repeatedly. B7 recipes were chosen partly from these
  seasons, and the choice among B7 mixtures was made on them today.
- All recipes use the calendar input. The early-season risk from the audit applies to both sides.
- October and early November are not on the Hub grid.
- Pooled WIS ratios are not CDC's pairwise relative WIS.

Reproduce on the B7 checkout:
1. `PYTHONPATH=src .venv/bin/python scripts/replay_b7_reported.py REPLAY A_blocks3 B_width256 B5_confirmed_candidate_1`
2. `PYTHONPATH=src .venv/bin/python scripts/score_b7_vs_system2.py REPLAY OUT`
