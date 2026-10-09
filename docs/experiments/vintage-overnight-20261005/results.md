# Artificial vintaging: completed node 1 research

Among fully replicated C1/C2/C3 comparisons, no artificial-input treatment has
a lower three-seed forward mean than
the C1 model trained on finalized inputs. Its 0.93984 score is nearly tied with
C3 trained using empirical admission errors in the first two seasons and actual
recent reports, 0.94058. These are development comparisons, not prospective validation.

The actual-recent-only controls are essential: they explain most apparent gains
from artificial early-season inputs on C2/C3. None of the tested learned
first-two-season generators improves the forward mean beyond actual recent
reports alone on those two architectures. A simpler C2 treatment does improve
its own finalized-input reference: half-strength empirical admission errors in
all three seasons, with ED and covariates finalized, reduces its mean score by
2.72%; all three paired seeds improve. That gain is mainly RSV admissions; all ED targets worsen.

## Protocol and selected completed comparisons

Every row **retrains** the named forecasting model and learns unchanged finalized
future labels at t+1 through t+4. Forward evaluation uses **2025–26**, after training
on **2022–23/2023–24/2024–25**; reporting errors use **only 2024–25**. Retrospective
evaluation uses **2024–25**, after training on **2022–23/2023–24/2025–26**; errors use
**only 2025–26**. “Early two” means 2022–23/2023–24; “recent” means the permitted
donor season in the relevant fold. “Actual” always includes the specified final
proxies where numerical archives are absent. Evaluation inputs are identical
Wednesday archived/proxy values with the original B2 source availability and lags.

C1 is a pathogen MLP with distance sharing and no covariates. C2 is a target MLP
without spatial sharing, using inpatient, wastewater, Kinsa, ILINet, clinical labs
and FluSurv. C3 is a target MLP with neighbor sharing and Kinsa. All use the same
B2 settings described in the protocol, with finality channels and the original
20% training missingness regularizer omitted. This choice of regularization is
separate from assumed input availability. All reported fits ran on node 1 L40 GPUs.

Scores below are mean±SD across seeds 42/43/44. Lower relative WIS is better; it compares each location with the frozen hub ensemble.
The forward score covers all six targets; the retrospective score covers only
flu/COVID admissions. Admission targets receive twice ED weight; US gets 20%,
states/DC share 80%. A half-strength error halves the signed log reporting error.
The learned half-strength admission treatment includes its conditional mean and
centered empirical residuals. Prediction labels are never perturbed.

| Model | Training inputs | Forward 2025–26, six targets | Retrospective 2024–25, flu/COVID admissions |
| --- | --- | ---: | ---: |
| C1 | Finalized inputs in all three training seasons | 0.93984 ± 0.01230 | 0.93467 ± 0.03536 |
| C1 | 50% clean episodes; 50% half-strength empirical errors in all six signals, all three seasons | 0.94765 ± 0.02118 | 0.86518 ± 0.05055 |
| C2 | Finalized inputs in all three training seasons | 0.98733 ± 0.01868 | 0.93397 ± 0.04085 |
| C2 | Finalized early two; actual recent reports | 0.96640 ± 0.02106 | 0.90910 ± 0.04759 |
| C2 | Empirical half-strength admissions all three; ED and covariates finalized | 0.96053 ± 0.03444 | 0.90603 ± 0.06245 |
| C2 | Learned half-strength admissions early two; actual recent reports | 0.97044 ± 0.01683 | 0.84737 ± 0.00826 |
| C3 | Finalized inputs in all three training seasons | 0.97501 ± 0.05111 | 0.97920 ± 0.06795 |
| C3 | Finalized early two; actual recent reports | 0.94373 ± 0.02378 | 0.86213 ± 0.04038 |
| C3 | Empirical half-strength admissions early two; actual recent reports | 0.94058 ± 0.03156 | 0.83249 ± 0.02578 |
| C3 | Learned half-strength admissions early two; actual recent reports | 0.94701 ± 0.01996 | 0.84071 ± 0.02983 |

C3's empirical admission errors in the first two seasons improve on actual-recent-only
training by just 0.33% (two of three seeds). Learned admission errors instead worsen
that reference by 0.35% (one of three seeds improves). C2's learned admission errors
worsen its actual-recent-only reference by 0.42%. C1's learned admission errors partly
repair its worse actual-recent training setup, but still score 1.99% worse than its
all-finalized-input reference. The final C1 clean/noisy episode mixture also fails
forward: 0.94765 versus 0.93984, a 0.83% worsening, with one of three seeds improving.

The first-two-only empirical treatment that keeps the recent season finalized
also lacks a consistent win: forward means are 0.94604 for C1, 0.97395 for C2 and
0.97765 for C3, versus their all-finalized-input references 0.93984/0.98733/0.97501.
Only C2 has a lower mean, and only one of its three paired seeds improves.

## What the reporting-model work established

A conditional ridge model of archived/final log error improves blocked temporal
validation RMSE for current-week admissions. For 2024–25 donor errors, flu/COVID/RSV
RMSE changes from 0.1373/0.1735/0.1899 using a shared mean to 0.1227/0.1660/0.1762
with ridge. Standalone gradient boosting and ridge plus boosted residuals improve
none of the six current-week admission comparisons across the two donor seasons.
Better admission error prediction therefore does not by itself establish a better
six-target forecasting pipeline.

Learned means can extrapolate into cells without archived pairs, whereas empirical
residuals there remain zero. In the first generator audit, empirical transport left
98% of 2022–23 RSV-ED input cells unchanged; learned means left only 5% unchanged.
This changes the density of perturbation as well as its conditional mean. Current-week
2024–25 ED evidence is concentrated in a single temporal validation block; withholding
it leaves no fitting evidence, so tied zero-fallback errors are not meaningful ED validation.

The first round used generator v1. Follow-up v2 sets growth features to zero when
adjacent values are unavailable. Learned follow-up comparisons also change strength
and perturbed signals, so differences from v1 are not isolated ED-only effects.
Admissions-only arms still use the inherited joint donor eligibility and phase matching.

## Material assumptions and interpretation

- Full scheduled availability is assumed only where historical final truth exists.
  RSV admission inputs are absent in 2022–23; no such observations are invented.
- Frozen-final numerical proxies are substantial. At age zero,2024–25 scheduled
  inputs use proxies in 33–37% of available admission cells and 88–89% of ED cells;
  in 2025–26 the ranges are 12–14% and 20–23%. Proxies never train empirical reporting
  errors, but are used in the actual-recent training and evaluation inputs.
- Error transport assumes the permitted donor reporting process is informative about
  earlier seasons after conditioning on phase, level, calendar and location. Only 34
  donor origins are eligible in 2024–25, versus 48 in 2025–26. Missing archive pairs
  are not evidence of zero true reporting error.
- The secondary shared flu/COVID-admission score uses the same forecasts without
  retraining. It confirms that target support explains part, but not all, of the
  retrospective/forward difference. The six-target forward objective remains primary.
- Three seeds quantify optimization variability, not all reporting or epidemiological
  uncertainty. Seasons were reused for development; no H100 results are pooled here.

## Artifacts and completion

All 117 research seed runs completed both folds:78 first-round,36 admissions follow-up
and 3 diluted C1 runs. This gives 37 fully replicated conditions and 6 single-seed pilots;
the pilots are excluded from three-seed claims. One timing-driven early restart is
recorded; it completed on its second attempt. No research GPU allocation remains.

- [Full three-seed comparisons, exact reference IDs and paired differences](principal-three-seed-comparisons.csv)
- [Contribution beyond actual recent reports](actual-recent-attribution.md), [figure](actual-recent-attribution.png)
- [Shared-target seasonal comparison](shared-target-season-comparison.png)
- [Archive/proxy support](scheduled-input-proxy-audit.png), [numerical counts](scheduled-input-proxy-audit.csv)
- [Learned reporting-model validation](reporting-generator-blocked-validation.png), [nonlinear diagnostic](nonlinear-reporting-blocked-validation.png)
- [Manager plan/launch/status/rank commands](manager-commands.md)
- [Complete protocol and decision log](index.md), [final job state](final-job-state.json)
- [Integration patch](implementation.patch), [source-copy checksums](reproduction-source-checksums.json), [metadata clarification](reporting-metadata-clarification.json)

The shared source tree was not modified. Private source is in
`/tmp/chromantis-vintage-overnight-20261005` locally and
`/proj/jlessler/projects/tapestry-all/tapestry-vintage-overnight-20261005` remotely.
The patch contains the learned generator, scenario options and dated research scripts;
pinned experiment snapshots preserve the exact fitted versions. Focused scientific
checks passed for donor-season isolation, unchanged labels/availability and input scope.
Graphs were generated with Matplotlib and were not visually inspected, per instructions.
