# B7-fast: seed and recipe ensembles, 2024 lapse weeks excluded

Written 7 October 2026, 22:50 EDT. No model was retrained here. The page combines the saved B7-fast
forecasts into ensembles and scores them again.

## What was scored

- **Models.** These are the B7-fast fits: each recipe was retrained three times, each time holding out
  one of 2023–24, 2024–25 and 2025–26 and training on the other three of the four completed seasons
  (2022–23 to 2025–26). Seeds 44 and 45.
  - Training inputs: admission and ED histories artificially revised with errors from 2025–26 reports.
    Kinsa unchanged. FluSurv omitted.
  - Labels: latest future flu admissions and ED.
- **Evaluation inputs.** The same three artificial 2025–26-style revision draws of each held-out season
  as the B7-fast report, corrected by each fit's own correction tree (the "corrected" view). Scores
  average the three draws.
- **Ensembles.** A recipe's two seeds get equal weight, and so does every recipe in a group. Two
  combination rules:
  - `vincent`: average the quantiles;
  - `mixture`: mix the predictive distributions.
- **Tasks.** October–May reference dates, horizons 0–3. **Change from the B7-fast report:**
  admissions tasks whose target week falls in the 2024 NHSN reporting lapse (weeks ending 11 May to
  2 November 2024) are dropped. That removes May 2024 from 2023–24 and the October-to-early-November
  2024 targets from 2024–25. Those labels count only the 1,800–2,700 hospitals that reported during
  the lapse; with them included, October 2024 and May 2024 log WIS was 0.52–0.82, against 0.17–0.45
  in other months. ED tasks are kept.
- **Scores.** Lower WIS is better. States/DC are pooled; the US is separate. The three seasons
  count equally. "Log" means WIS on log(x+1) admissions, the closest available proxy for CDC's
  states-only log-scale ranking.

## Results

| Ensemble (rule) | State/DC log adm. | 2023–24 | 2024–25 | 2025–26 | Oct–Nov only | State/DC native adm. | US log adm. | State/DC ED ×1000 | 95% coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B5 ILI + B_width256 (vincent) | **0.2698** | 0.243 | 0.282 | 0.285 | 0.333 | 51.5 | 0.145 | 4.31 | 0.922 |
| A_blocks3 + B_width256 (mixture) | 0.2715 | 0.243 | 0.280 | 0.292 | 0.333 | 51.6 | 0.151 | 4.28 | 0.939 |
| A_blocks3 + B5 ILI + B_width256 (vincent) | 0.2710 | 0.245 | 0.280 | 0.289 | 0.339 | 50.7 | 0.146 | 4.27 | 0.931 |
| A_blocks3 + B_width256 (vincent) | 0.2719 | 0.246 | 0.278 | 0.292 | 0.336 | 52.2 | 0.153 | 4.31 | 0.928 |
| A_width192 + A_blocks3 + B_width256 + B_blocks3 (vincent) | 0.2736 | 0.243 | 0.285 | 0.293 | 0.344 | 51.5 | 0.148 | 4.27 | 0.926 |
| All six base recipes (vincent) | 0.2741 | 0.246 | 0.287 | 0.290 | 0.345 | 51.6 | 0.147 | 4.29 | 0.925 |
| A_width192 + B_width256 (vincent) | 0.2744 | 0.247 | 0.284 | 0.293 | 0.335 | 52.9 | 0.154 | 4.32 | 0.910 |
| B_width256 alone | 0.2769 | 0.254 | 0.288 | 0.289 | **0.324** | 56.5 | 0.168 | 4.51 | 0.900 |
| A_blocks3 + B_blocks3 (mixture) | 0.2790 | 0.243 | 0.297 | 0.297 | 0.360 | 50.9 | 0.145 | 4.31 | 0.941 |
| A_width192 + B_blocks3 (mixture) | 0.2803 | 0.244 | 0.300 | 0.297 | 0.361 | 51.2 | 0.144 | 4.27 | 0.930 |
| B5 ILI alone | 0.2847 | 0.263 | 0.300 | 0.291 | 0.366 | 50.4 | 0.142 | 4.38 | 0.920 |
| A_width192 alone | 0.2855 | 0.253 | 0.299 | 0.304 | 0.361 | 51.8 | 0.150 | 4.31 | 0.902 |
| B_blocks3 alone | 0.2873 | 0.244 | 0.311 | 0.307 | 0.375 | 53.2 | 0.148 | 4.46 | 0.919 |
| A_blocks3 alone | 0.2874 | 0.267 | 0.289 | 0.306 | 0.366 | 51.7 | 0.165 | 4.36 | 0.930 |
| B_width192 alone | 0.2923 | 0.266 | 0.314 | 0.297 | 0.365 | 57.8 | 0.173 | 4.65 | 0.897 |
| B_width192, calendar removed, alone | 0.3153 | 0.304 | 0.338 | 0.304 | 0.427 | 61.2 | 0.185 | 4.81 | 0.893 |

"Oct–Nov only" is the state/DC log-admission WIS for October and November reference dates, with
the three seasons counted equally. Coverage is the share of state/DC admission outcomes inside the
nominal 95% interval. All rows, including both rules for every group, are in
[ensemble-scores.csv](ensemble-scores.csv), [month-scores.csv](month-scores.csv) and
[three-recipe-scores.csv](three-recipe-scores.csv).

Recipe definitions:
- **A_blocks3:** width-96 sampled MLP with Kinsa and a three-block residual decoder, trained on
  corrected artificial histories.
- **A_width192:** the same recipe at width 192 and the default learning rate.
- **B5 ILI:** `B5_confirmed_candidate_1`, a sampled width-96 MLP with Kinsa, pretrained on pre-August-2022
  state ILI, with a ten-week history and multiscale features.
- **B_width256, B_blocks3, B_width192:** neighbor-sharing MLPs that predict quantiles directly,
  trained on uncorrected artificial histories plus reconstruction of recent latest values.
  B_width256 has width 256 and learning rate 0.0005.

## Reading

- **B_width256 is the best quantile recipe here.** Every pair containing it beats the same partner
  paired with B_blocks3 or B_width192, by 2–4% on state/DC log WIS and by 6–8% in October–November.
  The earlier B6 evaluation supports this only in part. B6 used five seeds, training histories
  where only admissions were artificially revised, and real Wednesday reports for 2024–25 and
  2025–26, late November to May. On state/DC log WIS it put B_width192 and B_width256 level
  (0.2749 and 0.2751), both ahead of B_blocks3 (0.2796).
- **Adding a sampled recipe helps everything except October–November.** Compared with B_width256
  alone, A_blocks3 + B_width256 (mixture) has:
  - 2% lower log WIS;
  - 9% lower native WIS;
  - 10% lower US log WIS;
  - 5% lower ED WIS;
  - 95% coverage of 0.94 instead of 0.90.

  It is 3% worse in October–November.
- **The sampled partner hardly matters.** With A_blocks3, B5 ILI or both, the scores agree within
  0.2–0.8%, which is less than the noise from two seeds.
- **Calendar.** With the B_width192 recipe, removing the calendar input made it 8% worse overall and
  17% worse in October–November. All three held-out seasons peak between late December and early
  February, so this does not test an early season like 2026–27.
- **The absolute level is optimistic.** Scored on the Hub's tasks, every single B7 fit ranks 1st to
  5th of the 30–46 Hub models on log admissions in each season (pairwise relative WIS 0.49–0.73).
  Two reasons:
  - the artificial evaluation errors come from the same 2025–26 set that training and the
    correction trees used;
  - for 2023–24 and 2024–25, the models trained on later seasons.

  Use these scores to rank recipes, not to predict live accuracy.

## Assumptions

- Two seeds per recipe. Differences below about 1% between ensembles are not resolved.
- Dropping the lapse target weeks only changes scoring. Every fit still trained on the lapse labels.
- The A_blocks3 calendar-removal and four-week-correction arms did not run; they were not relaunched with `--retry-failed`.
- Reproduce on the B7 checkout with `scripts/score_b7_ensembles.py OUT`, then
  `scripts/month_b7_ensembles.py OUT`.
