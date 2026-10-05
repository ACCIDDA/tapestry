# Contribution of artificial errors after recent-season actual inputs

Every row retrains the named model with unchanged finalized t+1 through t+4 labels. Forward evaluation uses 2025–26 Wednesday reports after training on 2022–23/2023–24/2024–25. The reference uses finalized 2022–23/2023–24 inputs and actual 2024–25 reports. Treatments add artificial errors to those first two seasons, learned or sampled only from 2024–25. Retrospective evaluation swaps 2024–25 and 2025–26 in these roles. All use identical B2 source delays and frozen-final proxies where archives are absent. Forward scores cover six targets; retrospective scores cover flu/COVID admissions only. Lower relative WIS is better.

C1: pathogen MLP, distance sharing, no covariates. C2: target MLP, no spatial sharing, inpatient/wastewater/Kinsa/ILINet/clinical-lab/FluSurv covariates. C3: target MLP, neighbor sharing, Kinsa.

Only comparisons with all three treatment and reference seeds are shown. Percent changes are changes of three-seed means. The CSV also gives paired differences and seed variation.

| Model | Additional training-input treatment in 2022–23/2023–24 | Forward change vs actual-recent-only | Improving seeds | Retrospective change vs actual-recent-only |
| --- | --- | ---: | ---: | ---: |
| C1 | Empirical half-strength early two; recent actual reports | -1.81% | 2/3 | -0.82% |
| C1 | Empirical full-strength early two; recent actual reports | +5.87% | 0/3 | -3.02% |
| C1 | Learned conditional mean and residual early two; recent actual reports | +0.46% | 1/3 | -2.98% |
| C1 | Learned conditional mean only early two; recent actual reports | +1.23% | 1/3 | +2.16% |
| C2 | Empirical half-strength early two; recent actual reports | +1.28% | 2/3 | -6.96% |
| C2 | Empirical full-strength early two; recent actual reports | +7.62% | 0/3 | +0.79% |
| C2 | Learned conditional mean and residual early two; recent actual reports | +11.46% | 0/3 | +2.10% |
| C2 | Learned conditional mean only early two; recent actual reports | +2.75% | 1/3 | -3.71% |
| C3 | Empirical half-strength early two; recent actual reports | +5.32% | 1/3 | -6.78% |
| C3 | Empirical full-strength early two; recent actual reports | +1.39% | 2/3 | -3.72% |
| C3 | Learned conditional mean and residual early two; recent actual reports | +1.59% | 1/3 | -4.24% |
| C3 | Learned conditional mean only early two; recent actual reports | +0.62% | 2/3 | -3.08% |
| C1 | Empirical half-strength admissions early two; actual recent reports | -1.56% | 1/3 | -5.42% |
| C1 | Learned half-strength admissions early two; actual recent reports | -3.14% | 2/3 | -0.17% |
| C2 | Empirical half-strength admissions early two; actual recent reports | +3.65% | 1/3 | -4.94% |
| C2 | Learned half-strength admissions early two; actual recent reports | +0.42% | 1/3 | -6.79% |
| C3 | Empirical half-strength admissions early two; actual recent reports | -0.33% | 2/3 | -3.44% |
| C3 | Learned half-strength admissions early two; actual recent reports | +0.35% | 1/3 | -2.48% |
