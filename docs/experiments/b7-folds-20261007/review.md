# Review of the completed model comparison

Eight configurations completed two fitting seeds and three held-out seasons each: 48 model fits. The initial allocations reached their 28-minute limit; a user-authorized 20-minute continuation finished in 11 minutes 39 seconds. It reused saved model weights and completed forecasts, restarting only four unfinished training folds with their original settings and seeds. The final report completed at 10:26 pm Eastern on 7 October 2026.

Every comparison below evaluates October–May reference dates at horizons 0–3, with three shared artificial reporting draws per issuance. Evaluation on 2023–24 trains on 2022–23, 2024–25 and 2025–26; evaluation on 2024–25 trains on 2022–23, 2023–24 and 2025–26; evaluation on 2025–26 trains on 2022–23, 2023–24 and 2024–25. These are exchangeable-season experiments, not chronological prospective backtests. Future prediction labels remain the September 2026 latest flu-admission counts and ED proportions. Artificial revisions change the admission and ED input histories; Kinsa, where used, and prediction labels remain unchanged.

## Model choices

The **256-wide neighbor-sharing quantile MLP, without Kinsa**, trained on artificially preliminary admission/ED histories and latest future labels plus recent-history reconstruction, has the lowest state/DC log-admission WIS: **0.308620**. Its evaluation inputs receive two-week tree correction. This is the strongest tested configuration for the prespecified primary score.

The **192-wide sampled-path MLP with Kinsa**, trained on cross-fitted tree-corrected artificial admission/ED histories and latest future labels, has state/DC log-admission WIS **0.317961**, native admission WIS **49.2624**, and ED WIS **0.004403**. Its evaluation inputs also receive two-week tree correction. The quantile model's corresponding native scores are **53.6017** and **0.004555**. Lower WIS is better. The sampled model is better on native admission counts and ED in each of the three held-out seasons; it is the stronger sampled-path option and the more balanced choice across those native scores. The quantile model wins log-admission WIS in 2024–25 and 2025–26, while the sampled model wins in 2023–24.

These are complete-configuration comparisons. They do not isolate an architecture or Kinsa effect. The input view was selected using these evaluation results, so the results are an exploratory screen rather than an independent final accuracy estimate. Two fitting seeds do not establish small differences precisely.

All eight configurations score best on the primary metric when their existing fitted models receive corrected evaluation histories. Raw and corrected inputs are replays of each fitted model; the half view mixes their predictive distributions 50/50. None of these view comparisons retrains the forecaster.

Retain the calendar in the tested 192-wide quantile configuration: removing it during retraining raises log-admission WIS from **0.321873 to 0.342296**, with the same preliminary training inputs, reconstruction labels and corrected evaluation inputs. This does not establish the effect in sampled models; that separate calendar experiment was stopped.

The 96-wide, three-residual-block sampled MLP with Kinsa does not show a primary-score improvement from replacing the two-week tree corrector with the residual neural corrector in both training and evaluation. After retraining, log-admission WIS is **0.322554**, versus **0.320964** with tree correction, and ED WIS is worse. The US score improves, so this is not a claim that the neural correction loses on every metric. The ILI-pretrained sampled configuration also fails to beat the leading configurations; that package additionally changes history length, signal features and training realizations, so its outcome does not isolate pretraining.

## What still needs work

The leading two models' uncalibrated 95% admission intervals cover only **88.6%** of state/DC observations under this artificial reporting process. Their 80% admission intervals cover about **73%**. These averages give the three seasons equal weight. Good mean WIS does not establish adequate interval calibration. No interval recalibration was introduced in this experiment; see [leading-model-coverage.csv](leading-model-coverage.csv).

Direct nowcaster diagnostics improve the newest admission week but worsen the preceding week for the tested tree and neural corrections. For example, the three-block sampled model's tree corrector changes newest-week admission MAE from **15.57 to 11.72** counts, but preceding-week MAE from **5.18 to 6.35**. The neural corrector gives **11.52** and **8.01**, respectively. These diagnostics use reporting draw zero and all available held-out episodes, rather than the headline October–May window. Compare before/after within a model; different model families need not have identical diagnostic support. A one-week correction is a justified next experiment, but was not tested here.

The sampled-model calendar-removal and four-week-correction comparisons were deliberately stopped for runtime before inspecting their comparative accuracy. Their four failed statuses are intentional; they cannot support model-choice conclusions.

## Audit corrections and verification

The previous donor exclusion removed winter reporting examples disproportionately when evaluating 2025–26. The new rule excludes the closest seasonal donor release in every recipient season and retains 47 of 48 donors. Adjacent error windows can overlap: the 2025–26 reporting-error bank is prescribed calibration information, not an independently held-out reporting dataset. Only reporting residuals are transferred; donor epidemic paths and future labels are not transferred. Other assumptions are the prescribed source-availability masks, non-revising Kinsa values, exchangeable epidemic seasons, and unchanged latest labels, including the existing 2024 reporting-coverage limitations.

The scorer's missing Scenario import was repaired. The previously omitted wide quantile candidates were included. Training and epoch selection both use the native predictive loss plus the added log-admission term; evaluation separates states/DC from US and reports native admission, native ED and log-admission WIS.

All **2,592** headline entries have matching truth/task support, all three reporting draws, and no duplicated score cells. Task-weighted month, horizon and location tables each reproduce the headline scores to floating-point precision. Three targeted scientific checks passed for ED units and excluded-pathogen leakage. No graph screenshots were inspected. Full evidence is in [score-validation.json](score-validation.json), [scientific-check.json](scientific-check.json), and the [report](index.md).

The exact manager plan, launch, status and rank commands, including the continuation, are in [commands.sh](commands.sh). Manager status still lists the four deliberately stopped optional fits; its generic retry command would restart them.
