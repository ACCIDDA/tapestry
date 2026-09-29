# Covariates help when their representation is right

**Completed 2026-09-25: all 189 runs, 63 configurations, three seeds and three
held-out seasons.** The best configuration is the independent-location pathogen
MLP with compact summaries of the mixed covariate bundle. It improves on its
no-covariate control in all three seeds, but does not beat the Hub ensemble overall.
The main practical result is to keep simple covariate processing and avoid adding
geographic attention by default.

<!-- no-mask-followup:start -->
## Follow-up: removing artificial masking mostly hurt

**All 96 follow-up runs completed: 32 selected formulations × three seeds.**
These were the top half of the original 63-formulation ranking. Every new run
uses the same pinned model code, data hashes, seeds, scoring support and training
settings as its counterpart, except `mask_rate=0.5` becomes `mask_rate=0`.
**Wednesday reporting availability remains unchanged.**

Removing artificial masking worsened the three-seed mean for **29 of 32
formulations**. Only **18 of 96 paired seeds** improved, and **16 formulations
worsened in all three seeds**. Across these selected formulations, the mean
seed-paired relative WIS change was **+6.0%** (median across formulations +5.4%);
positive means removing masking made the score worse.

| Formulation | With artificial masking | Without | Paired WIS change on removal | Seeds improved by removal |
| --- | ---: | ---: | ---: | ---: |
| **Independent, mixed summaries** | **1.036** | **1.121** | **+8.2%** | **0/3** |
| Independent, new flu sources smoothed | 1.049 | 1.115 | +6.3% | 0/3 |
| Independent, Kinsa summaries | 1.063 | 1.208 | +13.8% | 0/3 |
| Pooled, claims summaries | 1.067 | 1.109 | +4.1% | 1/3 |
| Independent, claims smoothed | 1.071 | 1.096 | +2.2% | 1/3 |
| Independent, no covariates | 1.124 | 1.101 | **−1.8%** | 2/3 |
| Attention, no covariates | 1.119 | 1.127 | +1.9% | 1/3 |

The percentages are means of seedwise ratios, not ratios of the displayed means;
this distinction matters when a control has large seed variation. Formulations
are counted as improved when their mean WIS decreases. The three improved means
are independent/no covariates, attention/claims summaries, and attention/new flu
shared encoder. The latter's gain is approximately 0.1%, too small to treat as a
compelling result from three seeds.

![Paired artificial-masking comparison for all 32 formulations](no-mask/matched-masking.png)

### The original leader remains the model to keep

Mixed summaries without artificial masking get worse both for states/DC
(**+7.8%**) and native US (**+9.7%**), with all three seeds worsening in each.
Its three-seed mean worsens in every held-out season:

| Season | Masked mixed summaries | No-mask mixed summaries |
| --- | ---: | ---: |
| 2023–24 | 1.019 | 1.082 |
| 2024–25 | 0.878 | 0.991 |
| 2025–26 | 1.210 | 1.289 |

The best no-mask mean is independent/smoothed claims at **1.096 ± 0.065** seed SD.
It is only **0.5% better** than the new no-covariate control in the paired
comparison (two of three seeds improve), and it is worse than its own masked
version, 1.071. None of the 32 no-mask configuration means beats the Hub ensemble.
The masked mixed-summary model remains the overall leader at 1.036.

Removing masking also worsened the leading models' interval coverage. Under the
same location/target/season weights, nominal 95% coverage changed:

- Mixed summaries: **84.0% → 81.7%**.
- Smoothed new flu sources: **84.6% → 80.0%**.
- Kinsa summaries: **85.2% → 81.7%**.

Thus the better coverage under masking is not just accompanied by an inferior
WIS tradeoff: the masked versions are better on both measures for these leaders.
All remain undercovered relative to 95%.

### Paired fan plots, directly on this page

These compare the **same mixed-summary formulation and seed 42**, with and without
artificial masking, against the Hub ensemble. The selected seed is fixed, not
chosen for looking good. Bands are 50% and 90%; the coverage statistics above
refer to separately scored 95% intervals. US and North Carolina illustrate behavior;
the numerical conclusions use all scored locations and all three seeds.

![US admissions: original leader with and without artificial masking](no-mask/fans-US-hosp.png)

![US ED: original leader with and without artificial masking](no-mask/fans-US-ed.png)

![North Carolina admissions: original leader with and without artificial masking](no-mask/fans-NC-hosp.png)

![North Carolina ED: original leader with and without artificial masking](no-mask/fans-NC-ed.png)

### What this changes in the interpretation

**Keep artificial masking for the current leading covariate formulations.** The
result is consistent with masking acting as useful regularization or encouraging
use of covariates when target histories are incomplete. This experiment cannot
distinguish those mechanisms. Its effect is conditional: the no-covariate
independent model improves slightly without masking.

**Artificial masking is not the main explanation supported by this experiment
for losing the old B0 performance.** Removing it does not restore B0; even the
unmasked no-covariate control scores 1.101. Full finalized target-history
availability versus Wednesday masks, the changed panel, and exact B0 recipe
reproduction remain separate unresolved comparisons. We should not switch back
to blaming natural reporting gaps: the scored-date availability audit below
already shows most latest targets are present.

**Selection limits the conclusion.** We deliberately reran the best half under
masking, so this comparison favors formulations selected for doing well in that
condition. It is not an unbiased estimate of masking's effect over all 63 models,
and the 96 paired seeds are not 96 independent epidemiological datasets. The
consistent deterioration of the three prior leaders supports retaining their
current settings; a universal claim that masking always helps would be unjustified.
No new training or calibration was launched for this analysis.

Evidence: [32 matched configuration comparisons](no-mask/matched_configurations.csv),
[96 paired seeds by geography](no-mask/paired_seeds.csv),
[season comparisons](no-mask/paired_seasons.csv),
[coverage](no-mask/coverage95.csv), and
[comparison provenance](no-mask/comparison_provenance.json).
Ranking `ranking-981191c4dc56` was generated in patron-node report job **2466709**.
Every matched seed was checked for identical frozen target/season/location/horizon
support, and identical dataset/population/frozen-manifest hashes and evaluation
member count. Early stopping is unchanged but may select different epochs when
masking changes; that is part of the training-procedure comparison.

Reproduce status and ranking on Longleaf:

```bash
.venv/bin/python -m tapestry.experiment.planner status -e forecast-no-mask-top32-v3
.venv/bin/python -m tapestry.experiment.planner rank -e forecast-no-mask-top32-v3
```

Reproduce the paired tables, chart and fans from the downloaded score/forecast files:

```bash
.venv/bin/python analysis/forecast-geography-v2/compare_masking.py \
  data/experiments/forecast-no-mask-top32-v3/ranking-981191c4dc56
.venv/bin/python analysis/forecast-geography-v2/masking_fans.py
```

The [manager plan and original launch commands](../../design/forecast-no-mask.md#manager-commands)
remain recorded for reproduction; do not re-plan completed runs just to regenerate
this analysis.
<!-- no-mask-followup:end -->

## B0 comparison and reporting availability

**This is not a matched reproduction of B0.** B0 had finalized target histories
without the historical Wednesday publication mask and without the added 50%
artificial masking. This screen masks targets as well as covariates and uses a
later B1 training recipe. That changes the forecasting task. We have not isolated
the cause of the difference from B0's 0.883–0.895 results. See the
[verified protocol comparison and wastewater example](b0-comparison.md).

A scored-support audit found that most scored target histories contain the latest
week: all admissions in 2025–26 and approximately 95–99% of ED cells. Broad
full-calendar missingness does not establish the cause of the B0 score difference.
The completed no-artificial-mask follow-up above isolates that change; it mostly worsened the selected models.

A previous Saturday's wastewater value absent from the Wednesday snapshot is
unavailable even if final truth now exists. Smoothing can use older reports,
never the withheld value. A separately trained nowcaster was not used here.

## Leading results

Lower relative WIS is better; 1 is Hub ensemble parity. “Paired improvement” is
the mean of `100 × (1 − model/control)` over matched seeds, using the control with
the same geography. It is not the percentage computed from the two displayed means.
SD describes variation across three seeds, not a confidence interval.

| Model | Relative WIS, mean ± seed SD | Paired improvement | Seeds improved |
| --- | ---: | ---: | ---: |
| Independent, no covariates | 1.124 ± 0.062 | — | — |
| **Independent, mixed summaries** | **1.036 ± 0.030** | **7.6%** | **3/3** |
| Independent, new flu sources smoothed | 1.049 ± 0.038 | 6.6% | 3/3 |
| Independent, Kinsa summaries | 1.063 ± 0.025 | 5.3% | 3/3 |
| Pooled geography, claims summaries | 1.067 ± 0.039 | 9.5% versus pooled control | 3/3 |
| Independent, claims smoothed | 1.071 ± 0.035 | 4.4% | 2/3 |

The pooled control is weaker (1.181), so its larger percentage improvement does
not make it the better final model. The winning mean remains **3.6% worse than
the Hub ensemble** under this score; all three winning-model seeds exceed 1.

![Leading models and individual seeds](leaders-and-seeds.png)

## Fan plots: the three leading models

Shown directly below: the Hub ensemble, mixed covariate summaries, smoothed new
flu covariates, and Kinsa summaries. The models were selected by three-seed mean
WIS; all fans use **seed 42**, not the best seed. Black curves are final truth;
fans show medians and 50%/90% intervals at four-week reference intervals.
The native US and North Carolina are shown; these are illustrative locations,
not substitutes for all-location scores. Hub fans appear only where benchmark
support exists.

### US admissions

![US admissions: Hub ensemble and three leading models](best-fans/fans-US-hosp.png)

### US ED proportions

![US ED proportions: Hub ensemble and three leading models](best-fans/fans-US-ed.png)

### North Carolina admissions

![North Carolina admissions: Hub ensemble and three leading models](best-fans/fans-NC-hosp.png)

### North Carolina ED proportions

![North Carolina ED proportions: Hub ensemble and three leading models](best-fans/fans-NC-ed.png)

## 1. The original “covariates do not help” conclusion was too broad

For the independent-location model, the raw mixed bundle scores 1.125, essentially
unchanged from the 1.124 control. Summarizing exactly that bundle gives 1.036.
The mixed bundle contains claims, new ILINet/clinical-lab/FluSurv inputs, wastewater
WVAL-like indices and Kinsa. Its parameter count falls from **152,259 to 114,243**
across the three pathogen models—a 25% reduction. The control has 101,571.

This is consistent with the concern that a long raw covariate history is a poor
representation for this dataset. It does **not** establish excess parameters as
the cause: the summaries also change the statistics and transform the values.
The learned bottleneck is not a general cure: mixed/shared scores 1.140 despite
being compact. Smoothing the new flu bundle is excellent without reducing its
parameter count. Useful inductive structure matters more here than compression
alone.

Across the 15 geography/bundle pairs, smoothing and summaries each beat raw inputs
in 9; the learned encoder does so in 7. None is uniformly superior. Claims
summaries, for example, are poor without pooling even though mixed summaries win.

![Covariate effects against matched controls](covariate-effects-overall.png)

## 2. The new Delphi sources and national Kinsa are worth retaining

The three new flu sources with smoothing improve overall WIS by 6.6%, with
improvement in every seed. They improve both states/DC (5.3%) and US (11.2%).
These are bundle results; the experiment does not identify whether ILINet,
clinical-lab positivity or FluSurv drives the gain.

Kinsa summaries improve overall WIS by 5.3%. Broadcasting the national series
allows a **5.1% state/DC improvement, in all three seeds**, despite the absence of
learned spatial exchange. Its US improvement is 6.0%, but only two of three seeds
improve there. Raw Kinsa alone (1.128) does not improve the independent control.

The winning mixed-summary model improves states/DC by 6.7% and US by 11.0%, with
all three seeds improving in each geography. National gains are therefore not
merely hiding worse state forecasts. We cannot attribute the mixed model's gain
to any single source without leave-one-source-out comparisons.

Wastewater alone is less compelling: raw inputs score 1.191, summaries 1.100;
the summary arm improves on the control in two seeds, but trails the other leaders.
There is not enough evidence here to call wastewater useless or to credit it for
the mixed model's improvement.

![State and DC effects](covariate-effects-states.png)

![US effects](covariate-effects-us.png)

## 3. More geographic sharing usually made things worse

The no-covariate means are 1.124 (independent), 1.181 (pooled) and 1.119 (attention).
Attention's tiny mean advantage comes with much larger seed SD: 0.146 versus
0.062 for the independent control.

Keeping bundle and representation matched, **attention wins only 5/21 comparisons**
and pooling only 4/21. The best attention arm, raw Kinsa, scores 1.079; only one
seed improves against its attention control. The mixed-summary winner deteriorates
from 1.036 to 1.210 with attention. Pooling plus claims summaries is a useful
exception, but does not surpass the best independent model.

This screen does not support the hypothesis that covariates generally need
nationwide attention to become useful. It tests equal-state context pooling and
all-location attention, not adjacency, distance-based structure, regional models
or population-weighted pooling. Kinsa is a national input even in the independent
architecture, so “independent” does not mean “without national information.”

## 4. There are remaining season and calibration weaknesses

| Model | 2023–24 | 2024–25 | 2025–26 |
| --- | ---: | ---: | ---: |
| Independent, no covariates | 1.077 | 0.966 | 1.329 |
| Mixed summaries | 1.019 | 0.878 | 1.210 |
| New flu sources smoothed | 0.956 | 0.921 | 1.271 |
| Kinsa summaries | 0.989 | 0.875 | 1.325 |

Mixed summaries improve the mean in all three seasons, although not every
seed-season pair improves. Kinsa contributes almost no mean improvement in
2025–26. The new flu bundle is particularly good in 2023–24, when the benchmark
supports only flu admissions.

**The seasons do not contain the same scored targets:** 2023–24 has flu admissions;
2024–25 has flu and COVID admissions; 2025–26 has all six targets. Therefore the
higher last-season score cannot be interpreted purely as a temporal failure.
COVID ED is a conspicuous weakness: the mixed-summary model's relative WIS is
2.157 on that target. Its flu admissions score is 0.967, COVID admissions 1.100,
RSV admissions 1.001, flu ED 1.187 and RSV ED 0.894. These target scores use their
available seasons and should not be averaged to reconstruct the combined score.

The best model's nominal 95% intervals cover only **84.0%** under the same
location/target/season weighting; the control covers 85.0% and the ensemble 90.7%.
Better WIS has not resolved undercoverage. Calibration and COVID ED diagnostics
are more justified next steps than increasing geographic complexity.

![Leading models across seasons](leading-seasons.png)

## What to take forward

1. Use **independent-location mixed summaries** as the primary candidate and
   **smoothed new flu sources** as the simpler alternative.
2. Keep **Kinsa summaries** as a compact national-signal comparator.
3. Confirm these shortlisted models on a genuinely new evaluation period before
   declaring a winner. Three seeds measure optimization variability, not three
   independent epidemiological replications; selecting among 63 configurations
   on these same folds introduces selection optimism.
4. Examine COVID ED errors and interval calibration. Any calibration parameters
   must be fitted outside the final evaluation data.
5. If further source attribution is needed, drop one group at a time from the
   winning mixed representation. These runs were not a factorial source-ablation
   study and cannot separate individual source effects.

No further training or calibration was launched for this report.

## Ensemble-support audit

A direct audit of the original leading model (seed 42) verified **56,662 scored
forecast cells across nine target/season cases**. Every scored cell has both a
model forecast and the named official Hub ensemble forecast, matched on reference
date, target-end date, location and horizon within its target/season. There are
**zero missing ensemble rows, zero missing model rows, zero duplicate ensemble
keys and zero truth mismatches**. Both forecasts use the same frozen observed value
and 23-quantile grid. The model's extra predictions outside ensemble support are
excluded. No ensemble forecast is imputed. Horizon 0 ends on the reference Saturday;
the model's context ends one week before that Saturday.

This audit checks the saved benchmark, not every historical upstream submission.
Within each saved case, frozen units exactly equal the available ensemble rows.
The shared scorer applies the same matching requirements to every fitted run and
raises on missing or duplicate forecast tasks. Earlier ranking checks also verified
identical support across all runs. This rules out scoring model-only weeks as an
explanation for the current comparison; it does not by itself prove equivalence
to the old B0 training/data protocol. Both forecasts are scored against **frozen
benchmark truth**, which need not equal the most recently rebuilt training panel.

[Per-case support counts](ensemble_support_audit.csv) ·
[Audit record](ensemble_support_audit.json). Reproduce with
`python analysis/forecast-geography-v2/ensemble_support_audit.py`.

## Protocol, evidence and reproduction

Ranking `ranking-e53fd92c4f32` includes every planned seed. The shared scorer
verified identical frozen target/season/location/horizon support across runs.
All 567 fold manifests use the same panel hash. Forecasts use **finalized values
masked by Wednesday reporting availability**, including national Kinsa broadcast
to all locations. Later revisions are allowed. This is retrospective development
CV, not a historical real-time nowcaster-to-forecaster evaluation.

Scoring first forms each location's model/ensemble WIS ratio. States/DC share
80% of the geography weight equally and native US receives 20%. Within a season,
available admissions targets receive weight 1 and ED targets 0.5; seasons then
receive equal weight. Missing historical target support is common to all models.
A ratio below one denotes improvement under these weights, not uniform dominance
on every task. All paired percentages above use the mean of the three seedwise
ratios. No significance tests or independence-based confidence intervals are claimed.

- [Full standard report, forecast fans and all 63 scenarios](index.md)
- [Named ranking](named_ranking.csv), [paired comparisons](matched_comparisons.csv),
  [paired seed scores](paired_seed_scores.csv), [season comparisons](paired_season_scores.csv)
- [Parameter counts](parameter_counts.csv), [95% coverage](coverage95.csv),
  [analysis provenance](analysis_provenance.json)

From `/proj/jlessler/projects/tapestry-all/tapestry` on Longleaf:

```bash
.venv/bin/python -m tapestry.experiment.planner status -e forecast-geography-v2
.venv/bin/python -m tapestry.experiment.planner rank -e forecast-geography-v2
```

Ranking/plots completed in patron-node Slurm job **2445207**. The report job recipe
is `analysis/forecast-geography-v2/report.sbatch`; launch it with
`sbatch analysis/forecast-geography-v2/report.sbatch` when regenerating on Longleaf.
The analysis is reproducible with:

```bash
.venv/bin/python analysis/forecast-geography-v2/analyze.py \
  data/experiments/forecast-geography-v2/ranking-e53fd92c4f32
```

The original manager plan and training launch commands are recorded below for
reproduction, not required to regenerate this report:

```bash
bash experiments/forecast-covariates.sh
GPUS=6 sbatch --job-name=forecast-geography-v2 --array=0-3 scripts/jlessler.sbatch forecast-geography-v2
GPUS=6 sbatch --job-name=forecast-geography-v2-h100 --array=0-1 --nodelist=g1803jles02 scripts/jlessler.sbatch forecast-geography-v2
```
