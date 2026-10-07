# B4 flu runs versus Google_SAI-FluEns, 2025–26 (rescoring of saved scores only)

Question (6 October 2026): does the B4 flu model regularly beat Google on the season Google submitted?

**What is compared.** Every B4 flu model is a width-32–96 MLP (or shared encoder in part of the 600 sweep) trained on 2022–23, 2023–24 and 2024–25 with finalized future flu labels (forward fold) and evaluated on 2025–26 using each week's Wednesday Hub reports (finalized fills where no archive exists). Three evaluation-input views of the same fitted model: **raw** (Wednesday reports as published), **corrected** (newest flu-admission weeks replaced by the run's tree nowcaster), **half** (50/50 mixture of the raw and corrected forecast distributions). refineTop2 also has **calibrated** (corrected view with interval widths rescaled on held-out training weeks). Google is Google_SAI-FluEns, from its submitted FluSight quantiles.

**Score.** Mean WIS per task, pooled over flu admission tasks for the 50 states and DC, reference dates 2025-11-22 to 2026-05-30, horizons 0–3: 5,712 tasks for both B4 and Google. Ratio = B4 / Google; **below 1 means B4 is better**. Natural = admission counts; log = log(count + 1). Configuration ratio uses the two-seed mean WIS.

| Experiment | Configurations | Log, raw | Log, corrected | Natural, raw | Natural, corrected | Best log ratio (corrected) |
|---|---:|---:|---:|---:|---:|---:|
| 600-config sweep | 600 | 1 | 11 | 204 | 174 | 0.960 |
| Covariate refine | 24 | 0 | 3 | 23 | 18 | 0.966 |
| Output heads | 8 | 0 | 3 | 8 | 7 | 0.966 |
| refineTop2 retrains | 21 | 0 | 5 | 19 | 16 | 0.964 |
| refineTop2 ensembles of saved forecasts | 14 | 4 | 14 | 14 | 14 | 0.933 |

Cells give the number of configurations beating Google. The full table, including the half and calibrated views, median ratios and individual seed runs, is in `b4-google-check-20261006/summary.csv`, written by `compare.py`.

**Assumptions and limits.**
- Google's pooled WIS (natural 59.27, log 0.2989) comes from [google-comparison-20261005](google-comparison-20261005/results.md). It is scored against frozen Hub truth from 8 July 2026. B4 raw WIS is scored against the panel's finalized truth saved with the forecasts. We assume these agree on the 5,712 tasks. This was not checked value by value.
- The task count and date range match. We assume this is the same task set; it was not checked key by key.
- This is a pooled-mean head-to-head, not CDC's pairwise relative WIS against all qualifying Hub models. On the natural scale, pooled WIS is dominated by large states.
- All B4 configurations, leaders and ensemble members were chosen using 2025–26 results. "Beats Google" here is therefore a development result, not a prospective one.
- The Google comparison scored only 2025–26 (the season of the CDC report naming Google the top submission). Whether Google also submitted for 2024–25 was not checked.
