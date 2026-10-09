# 2026-09-19 · 2025–26 forward evaluation with a fixed training cutoff

Archived experiment. Results and figures describe the recorded protocol; current execution instructions are in the Workflow page.

## Best models by season

Top three configurations by the reported combined score (all configurations if fewer than three). Season columns are seed means. Lower is better; 1 is the matched Hub ensemble.

| Model | 2025-2026 | Combined |
|---|---:|---:|
| B1 B no-mask | 0.9411 | 0.9411 |
| B1 B gap-only | 0.9487 | 0.9487 |
| B1 gated 0.20 | 0.9615 | 0.9615 |
| Hub ensemble | 1.0000 | 1.0000 |
| B0 reference | No known result | No known result |

The saved matched-Hub score rows cover 2025–26 only, so the combined and season scores coincide. No known B0 reference verified on this exact forward-evaluation support.

## Protocol

<!-- protocol:start -->

Training cutoff: July 26, 2025 end of day UTC; mature labels through June 28. Test Wednesdays July 30, 2025–July 29, 2026; forecast targets August 2, 2025–August 1, 2026. Parameters frozen. This season was previously explored; this is **forward development**, not an untouched final test.

Experiment specification (not retained in the experiment archive) · Data assumptions (not retained in the experiment archive)

<!-- protocol:end -->

## Season splits

No known season-split graph or description.

## Findings

<!-- write-up: kept across regenerations -->

### Matched Hub-supported forecast tasks

| candidate | seeds | combined_mean | combined_sd | states_dc_combined_mean | US_combined_mean |
| --- | --- | --- | --- | --- | --- |
| separate | 3 | 1.0638 | 0.0365 | 1.0749 | 1.0193 |
| direct | 3 | 1.2402 | 0.0977 | 1.2255 | 1.2989 |
| joint | 3 | 1.2722 | 0.0145 | 1.2472 | 1.3720 |





Mean and sample SD describe fitting variability only. One season provides no independent season replication; seeds and locations are not independent seasons. Native WIS cannot be compared across targets' units. Hub support varies by target and is a subset of the full test period. [Matched target/season/geography scores](matched-hub-scores.csv).

### Full forward-period diagnostics

| candidate | target | wis | scaled_wis | covered_50 | covered_95 |
| --- | --- | --- | --- | --- | --- |
| direct | wk inc covid hosp | 220.4361 | 0.0403 | 0.2811 | 0.6638 |
| direct | wk inc covid prop ed visits | 0.0014 | 0.0498 | 0.2330 | 0.5644 |
| direct | wk inc flu hosp | 395.3301 | 0.0619 | 0.3976 | 0.7766 |
| direct | wk inc flu prop ed visits | 0.0051 | 0.0862 | 0.2921 | 0.6806 |
| direct | wk inc rsv hosp | 143.0794 | 0.0731 | 0.2305 | 0.5798 |
| direct | wk inc rsv prop ed visits | 0.0006 | 0.0535 | 0.3889 | 0.7631 |
| joint | wk inc covid hosp | 218.1275 | 0.0412 | 0.2592 | 0.6503 |
| joint | wk inc covid prop ed visits | 0.0017 | 0.0590 | 0.1699 | 0.4757 |
| joint | wk inc flu hosp | 476.6145 | 0.0715 | 0.3315 | 0.7176 |
| joint | wk inc flu prop ed visits | 0.0047 | 0.0795 | 0.2994 | 0.7045 |
| joint | wk inc rsv hosp | 143.8573 | 0.0716 | 0.2423 | 0.5800 |
| joint | wk inc rsv prop ed visits | 0.0005 | 0.0458 | 0.3779 | 0.7897 |
| separate | wk inc covid hosp | 232.8096 | 0.0425 | 0.2953 | 0.7221 |
| separate | wk inc covid prop ed visits | 0.0016 | 0.0551 | 0.2829 | 0.6419 |
| separate | wk inc flu hosp | 341.3595 | 0.0591 | 0.4218 | 0.7973 |
| separate | wk inc flu prop ed visits | 0.0041 | 0.0694 | 0.3726 | 0.7913 |
| separate | wk inc rsv hosp | 121.7592 | 0.0647 | 0.2856 | 0.6379 |
| separate | wk inc rsv prop ed visits | 0.0005 | 0.0448 | 0.3510 | 0.7691 |

All candidates share identical available forecast labels. Tables use equal locations within states/DC (80%) and US (20%). Target-native WIS, stable training-Q95-scaled WIS and coverage are reported separately from the Hub-relative ranking.



[Target and seed diagnostics](forecast-by-target-seed.csv) · [Location diagnostics](forecast-by-location.csv) · [Horizon diagnostics](forecast-by-horizon.csv)

### Recent predictions

12 candidate/seed/target/age/location ratios were undefined under the predeclared threshold (mean preliminary absolute error / training Q95 ≤ 0.0001). These are not assigned zero or epsilon denominators. No revision ratio exists for missing reports.



[Recent diagnostics](recent-summary.csv) · Location diagnostics and denominator flags (`recent-by-location.csv`, removed 9 October 2026: over 1 MB)


Archive dates are accepted availability proxies, with assumed interior completeness; strict provider-publication availability is not independently certified. Finalized values are 28-day-mature cutoff proxies. Four-day NSSP visible correction training begins only June 18, 2025. Older-history fitting uses cutoff-final proxies; deployment uses Wednesday reports. The separate pipeline additionally learns forecasting on exact cutoff-final recent inputs and deploys on estimates; sampling propagates uncertainty but does not eliminate that mismatch. No calibration, artificial masking or test-driven epoch selection was used.

[**Analysis, ranking against B1, fan plots and heatmaps**](index.md)

### Forward benchmark: ranking, fans and comparison with B1

**The separate nowcast-to-forecast pipeline is the strongest of the three forward candidates.** Its mean relative WIS is **1.064**, versus **1.240** for Direct B and **1.272** for Joint gated: improvements of **14.2%** and **16.4%**, respectively. It wins both paired comparisons for each of seeds 42, 43 and 44, and has the lowest mean in all six targets. These are fitting replicates of **one already-explored season**, not three independent season tests.

All nine forward runs are complete. No additional training was performed for this analysis. Saved B1 predictions were rescored with the shared scorer on **exactly the same 2025–26 tasks, pinned September 16, 2026 truth and Hub ensemble**. All 21 candidate/seed evaluations cover the same **37,844 forecast tasks**, and rescoring reproduces the original forward ranking. Relative WIS below 1 beats the ensemble; above 1 loses. The mean is a mean of model scores, not the score of an ensemble of seeds.

### Calibration and failure modes

The separate pipeline's nominal 95% forecast intervals cover only **75.4%**, versus **67.7%** for Direct B and **65.4%** for Joint gated. Its nominal 50% intervals cover **33.8%**. B1 intervals also undercover: approximately **81–84%** at the 95% level. These figures use the same matched tasks and scientific weighting as the ranking; the main results page additionally reports full-period coverage on all eligible labels.

The separate pipeline remains weakest relative to the ensemble at the shortest forecast horizon: **1.223 at horizon 0**, improving to **0.990 at horizon 3**. Horizon-specific denominators differ, so these are relative skill comparisons, not decreasing native error with distance. Its RSV-admission WIS is still 1.280, and about **80%** of that scientifically weighted WIS comes from underprediction penalties. Thus interval width is not the only issue; low predictions also need attention. This is a score decomposition, not a causal explanation of the model's behavior.



[Geography heatmap: separate versus direct](heatmap-geography.png) · [Horizon scores](comparison-horizons.csv) · [WIS decomposition](comparison-wis-components.csv)

### Recent correction is different from missing-report reconstruction

The table below uses training-only Q95 scaling, target weights 1 for admissions and 0.5 for ED, and states/DC 80%, US 20%. It retains all eligible groups, including near-zero baseline groups. It is a stable diagnostic, not a replacement for the predeclared location-relative recent ratios. Recent evaluation follows the test issuance partition; its observation weeks begin July 19, 2025, before the first forecast target week.

| Candidate | Recent task | Age (days) | MAE/Q95 | WIS/Q95 | Report MAE/Q95 | 95% coverage (%) |
| --- | --- | --- | --- | --- | --- | --- |
| joint | reconstruction | 4 | 0.047 | 0.033 | nan | 65.181 |
| joint | reconstruction | 11 | 0.048 | 0.034 | nan | 64.124 |
| joint | revision | 4 | 0.018 | 0.013 | 0.023 | 80.306 |
| joint | revision | 11 | 0.011 | 0.008 | 0.009 | 81.529 |
| separate | reconstruction | 4 | 0.064 | 0.048 | nan | 47.697 |
| separate | reconstruction | 11 | 0.063 | 0.048 | nan | 45.384 |
| separate | revision | 4 | 0.018 | 0.013 | 0.023 | 78.358 |
| separate | revision | 11 | 0.011 | 0.008 | 0.009 | 80.504 |

Four-day visible-report median errors improve relative to each observation week's own preliminary report. At eleven days the median corrections have **higher** scaled MAE than simply retaining that report, even though probabilistic WIS is modestly lower. The model's probabilistic objective and point accuracy should not be conflated.

The independent nowcaster reconstructs missing reports **worse** than the joint model: scaled MAE is about **0.064 versus 0.047** at four days and **0.063 versus 0.048** at eleven days. Its missing-report 95% coverage is only **48% / 45%**, versus **65% / 64%** for joint. Therefore better downstream forecasts do not demonstrate better nowcasting across all tasks. The pipeline contrast also changes forecast training inputs and adds independent fitting; it does not isolate the causal effect of nowcasting alone.

Missing reports have no report baseline or revision ratio (`nan` in the table means undefined, not zero). The predeclared near-zero rule flags two unique groups—Missouri flu ED at eleven days and Tennessee RSV admissions at eleven days—appearing in 12 candidate/seed instances. Relative ratios remain undefined for these groups. No threshold was selected after seeing these results.



[Recent scaled diagnostics](recent-scientific-summary.csv) · Predeclared ratio flags and location diagnostics (`recent-by-location.csv`, removed 9 October 2026: over 1 MB)

### What the B1 comparison means

The **2025–26 held-out B1 forecasts exist**. Four named target-MLP recipes were chosen for interpretation rather than by searching for the best 2025–26 result:

| B1 comparator | Fitting and evaluation differences |
| --- | --- |
| B no-mask | Closest backbone/input-channel recipe; cap 100 with validation-selected epochs, no masking, 256 predictive draws. |
| B gap-only | Previously strong control; cap 300 with selected epochs, 50% gap-only masking, 1,024 draws. |
| Gated 0.20 | Analogous gated formulation; recent weight 0.20, cap 300, mixed 50% masking, no revision augmentation, 1,024 draws. |
| Joint two-stage | Earlier jointly fitted recent-to-future formulation; recent weight 0.20, cap 300, mixed 50% masking, no revision augmentation, 1,024 draws. This is **not** the independently fitted forward pipeline. |

B1 used retrospective finalized older histories, supplied reference finals for some missing recent inputs, and later-finalized fitting references. It did not enforce the July 26, 2025 information cutoff or restrict NHSN revision fitting to the new reporting regime. Its held-out-season masking prevents ordinary cross-validation label overlap, but does not make the underlying revisions historically available. On these exact tasks, supplied-final flags account for **0–4.9%** of recent target input cells depending on target and age; they are not the majority of recent reports. Older finalized context, reference revisions, training-label maturity and selected epochs also differ. We cannot attribute the score gap specifically to recent final filling.

Rescoring aligns tasks, truth, quantiles and scientific weights; it cannot remove information already used during B1 fitting or prediction. Thus **0.941 for B1 B versus 1.064 for Forward Separate is not evidence that B1 would win at a Wednesday deployment cutoff**. Conversely, Forward Separate's small numerical advantage over old jointly trained two-stage B1 (1.064 versus 1.093) is not a controlled proof that independence alone is responsible.

[B1 input final-flag fractions](b1-final-input-fractions.csv) · [Configurations, refit epochs, paths and provenance](analysis-provenance.json)

### Conclusions for next-season development

1. **Keep the separate pipeline as the leading forward candidate**, with Direct B as the control. Its improvement is consistent across the three fits and six target averages in this development season. The evidence does not favor the present jointly gated forecast recipe.
2. **Do not treat the winning pipeline as calibrated or deployment-validated.** It remains above ensemble WIS overall, has substantial forecast undercoverage, and has weak missing-report reconstruction. Future calibration or input-mismatch adjustments must be fitted inside a training partition; these test results should not tune an apparent final-test winner.
3. **Treat correction and reconstruction separately.** The eleven-day median can damage an already useful visible report; missing values require a different evaluation from revision correction. Sparse pre-cutoff four-day NSSP revision support remains a serious limitation.
4. **Use older B1 as a descriptive reference, not a fair operational rival.** An operational comparison would require refitting a B1 recipe under the same historical information cutoff. These results do not establish that reporting-regime mixing caused prior failures or that nowcasting is generally unhelpful.

The exact-training/estimated-deployment mismatch remains present in the independent pipeline; sampling propagates uncertainty but does not resolve that mismatch. All forward candidates share a cutoff-final older-history fitting versus Wednesday-history deployment mismatch. Archive timestamps and completeness remain availability assumptions, and the 28-day maturity definition is a provisional finality proxy. This is forward development on an explored season, not an untouched final test. No seed- or geography-based significance claim is made.



| Candidate | Relative WIS | Seed SD | States/DC | US | 95% coverage |
| --- | --- | --- | --- | --- | --- |
| B1 B no-mask | 0.9411 | 0.0136 | 0.9681 | 0.8334 | 0.8109 |
| B1 B gap-only | 0.9487 | 0.0269 | 0.9761 | 0.8391 | 0.8319 |
| B1 gated 0.20 | 0.9615 | 0.0555 | 0.9811 | 0.8829 | 0.8283 |
| Forward Separate | 1.0638 | 0.0365 | 1.0749 | 1.0193 | 0.7541 |
| B1 joint two-stage | 1.0932 | 0.1345 | 1.0904 | 1.1043 | 0.8403 |
| Forward Direct B | 1.2402 | 0.0977 | 1.2255 | 1.2989 | 0.6774 |
| Forward Joint gated | 1.2722 | 0.0145 | 1.2472 | 1.3720 | 0.6544 |

| candidate | kind | age_days | scaled_ae | scaled_wis | baseline_scaled_ae | covered_95 |
| --- | --- | --- | --- | --- | --- | --- |
| joint | reconstruction | 4 | 0.0466 | 0.0326 | nan | 0.6518 |
| joint | reconstruction | 11 | 0.0477 | 0.0344 | nan | 0.6412 |
| joint | revision | 4 | 0.0183 | 0.0133 | 0.0230 | 0.8031 |
| joint | revision | 11 | 0.0111 | 0.0082 | 0.0087 | 0.8153 |
| separate | reconstruction | 4 | 0.0637 | 0.0476 | nan | 0.4770 |
| separate | reconstruction | 11 | 0.0632 | 0.0481 | nan | 0.4538 |
| separate | revision | 4 | 0.0178 | 0.0132 | 0.0230 | 0.7836 |
| separate | revision | 11 | 0.0106 | 0.0079 | 0.0087 | 0.8050 |

<!-- end write-up -->

## Forecast fans

### Fan plots

Fans show **seed 42**, selected in advance for visualization, with medians and central **50%/95% intervals** every fourth test issuance. Black curves are pinned reference observations. They are not a seed ensemble or a plot of fitting variability. All candidates use the same displayed origins and truth; no favorable seed or date was selected. US, North Carolina and California are fixed illustrative geographies, not representative performance samples. Full fan sheets include all seven compared models:

- [Influenza admissions: US, NC, CA](fans-flu_hosp.png)
- [COVID admissions: US, NC, CA](fans-covid_hosp.png)
- [RSV admissions: US, NC, CA](fans-rsv_hosp.png)
- [Influenza ED visits: US, NC, CA](fans-flu_prop_ed_visits.png)
- [COVID ED visits: US, NC, CA](fans-covid_prop_ed_visits.png)
- [RSV ED visits: US, NC, CA](fans-rsv_prop_ed_visits.png)

Admissions remain counts; ED panels display percentages (the scorer uses proportions). Each fan is a four-week forecast, not a continuous forecast trajectory across origins. Some B1 origins at the season boundary are unavailable; no fan is fabricated there. Primary ranking uses only the common frozen support.

<figure class="report-figure" markdown="1">

<figcaption>Forecast ranking</figcaption>

<div class="report-plot" markdown="1">

![Forecast ranking](ranking.png){style="width: 1050px"}

</div>

[Open original figure](ranking.png)

</figure>

<figure class="report-figure" markdown="1">

<figcaption>Matched target results</figcaption>

<div class="report-plot" markdown="1">

![Matched target results](targets.png){style="width: 1200px"}

</div>

[Open original figure](targets.png)

</figure>

<figure class="report-figure" markdown="1">

<figcaption>US influenza forecast fans</figcaption>

<div class="report-plot" markdown="1">

![US influenza forecast fans](fans-us-flu-overview.png){style="width: 1200px"}

</div>

[Open original figure](fans-us-flu-overview.png)

</figure>

<figure class="report-figure" markdown="1">

<figcaption>fans flu_prop_ed_visits</figcaption>

<div class="report-plot" markdown="1">

![fans flu_prop_ed_visits](fans-flu_prop_ed_visits.png){style="width: 1200px"}

</div>

[Open original figure](fans-flu_prop_ed_visits.png)

</figure>

<figure class="report-figure" markdown="1">

<figcaption>fans rsv_hosp</figcaption>

<div class="report-plot" markdown="1">

![fans rsv_hosp](fans-rsv_hosp.png){style="width: 1200px"}

</div>

[Open original figure](fans-rsv_hosp.png)

</figure>

<figure class="report-figure" markdown="1">

<figcaption>fans covid_hosp</figcaption>

<div class="report-plot" markdown="1">

![fans covid_hosp](fans-covid_hosp.png){style="width: 1200px"}

</div>

[Open original figure](fans-covid_hosp.png)

</figure>

<figure class="report-figure" markdown="1">

<figcaption>fans covid_prop_ed_visits</figcaption>

<div class="report-plot" markdown="1">

![fans covid_prop_ed_visits](fans-covid_prop_ed_visits.png){style="width: 1200px"}

</div>

[Open original figure](fans-covid_prop_ed_visits.png)

</figure>

<figure class="report-figure" markdown="1">

<figcaption>fans rsv_prop_ed_visits</figcaption>

<div class="report-plot" markdown="1">

![fans rsv_prop_ed_visits](fans-rsv_prop_ed_visits.png){style="width: 1200px"}

</div>

[Open original figure](fans-rsv_prop_ed_visits.png)

</figure>

<figure class="report-figure" markdown="1">

<figcaption>fans flu_hosp</figcaption>

<div class="report-plot" markdown="1">

![fans flu_hosp](fans-flu_hosp.png){style="width: 1200px"}

</div>

[Open original figure](fans-flu_hosp.png)

</figure>

## Score diagnostics

<figure class="report-figure" markdown="1">

<figcaption>Coverage</figcaption>

<div class="report-plot" markdown="1">

![Coverage](coverage.png){style="width: 1200px"}

</div>

[Open original figure](coverage.png)

</figure>

<figure class="report-figure" markdown="1">

<figcaption>Recent scaled errors</figcaption>

<div class="report-plot" markdown="1">

![Recent scaled errors](recent.png){style="width: 1200px"}

</div>

[Open original figure](recent.png)

</figure>

<figure class="report-figure" markdown="1">

<figcaption>Matched target heatmap</figcaption>

<div class="report-plot" markdown="1">

![Matched target heatmap](heatmap-comparison.png){style="width: 1200px"}

</div>

[Open original figure](heatmap-comparison.png)

</figure>

<figure class="report-figure" markdown="1">

<figcaption>Horizon heatmap</figcaption>

<div class="report-plot" markdown="1">

![Horizon heatmap](heatmap-horizons.png){style="width: 1200px"}

</div>

[Open original figure](heatmap-horizons.png)

</figure>

<figure class="report-figure" markdown="1">

<figcaption>Recent-task heatmap</figcaption>

<div class="report-plot" markdown="1">

![Recent-task heatmap](heatmap-recent.png){style="width: 1200px"}

</div>

[Open original figure](heatmap-recent.png)

</figure>

<figure class="report-figure" markdown="1">

<figcaption>heatmap geography</figcaption>

<div class="report-plot" markdown="1">

![heatmap geography](heatmap-geography.png){style="width: 1200px"}

</div>

[Open original figure](heatmap-geography.png)

</figure>

## Matched comparisons

No known result or figure.

## Full ranking

### Ranking on identical tasks

| Candidate | Relative WIS | Seed SD | States/DC | US | 95% coverage (%) |
| --- | --- | --- | --- | --- | --- |
| B1 B no-mask | 0.941 | 0.014 | 0.968 | 0.833 | 81.088 |
| B1 B gap-only | 0.949 | 0.027 | 0.976 | 0.839 | 83.194 |
| B1 gated 0.20 | 0.961 | 0.056 | 0.981 | 0.883 | 82.833 |
| Forward Separate | 1.064 | 0.036 | 1.075 | 1.019 | 75.413 |
| B1 joint two-stage | 1.093 | 0.134 | 1.090 | 1.104 | 84.035 |
| Forward Direct B | 1.240 | 0.098 | 1.226 | 1.299 | 67.736 |
| Forward Joint gated | 1.272 | 0.015 | 1.247 | 1.372 | 65.440 |

**The forward rows form the controlled formulation comparison. B1 rows are descriptive historical comparators with different information and fitting protocols.** This is a comparison of four named B1 recipes, not an exhaustive reranking of every B1 configuration. Seeds 42–44 are used throughout, even where five B1 fits exist.



The separate pipeline beats the Hub ensemble narrowly for flu admissions (0.957), COVID admissions (0.983) and flu ED visits (0.993), but not RSV admissions (1.280), COVID ED visits (1.057) or RSV ED visits (1.087). Its combined score remains **6.4% worse than the ensemble**. The improvement over Direct B appears in both states/DC (1.2255 → 1.0749) and US (1.2989 → 1.0193). It improves mean location scores in 36/52 flu-admission geographies, 46/52 COVID-admission, 51/52 RSV-admission, 45/52 flu-ED, 48/52 COVID-ED and 49/51 RSV-ED geographies. These counts describe breadth, not independent statistical replication.

Joint gated is 2.6% worse than Direct B in the mean and loses in two of three paired seeds; the ordering reverses for seed 44. Its small seed SD does not establish generalization. There is no forecast advantage for this joint recipe in the current benchmark.

[All seed scores](comparison-seeds.csv) · [Paired forward differences](paired-forward-seeds.csv) · [Target/season/geography scores](comparison-targets.csv) · [Matched support dates/counts](comparison-support.csv)

## Appendix

No known result or figure.
