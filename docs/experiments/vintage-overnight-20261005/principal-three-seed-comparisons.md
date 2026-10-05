# Principal three-seed comparisons on node1 L40

Every row retrains the named model and learns unchanged finalized future labels. Forward evaluation uses 2025–26 Wednesday reports after training on 2022–23/2023–24/2024–25; artificial errors come only from 2024–25. Retrospective evaluation uses 2024–25 after training on 2022–23/2023–24/2025–26; errors come only from 2025–26. Source delays and frozen-final proxies for absent archives are identical across evaluations.

Forward scores cover six targets; retrospective scores cover flu/COVID admissions only. Lower relative WIS is better. Admission weight is twice ED; US receives 20%, states/DC share 80%. C1 is a pathogen MLP with distance sharing and no covariates. C2 is a target MLP without spatial sharing using inpatient, wastewater, Kinsa, ILINet, clinical labs and FluSurv. C3 is a target MLP with neighbor sharing and Kinsa.

Early two means 2022–23/2023–24; recent means the permitted error-donor season above. Percent changes compare with the same architecture trained on finalized inputs in all three seasons. CSV contains SDs, paired differences, seed counts, full definitions and exact clean/actual-recent reference IDs. Actual-recent attribution is withheld until its three seeds complete.

| Model | Training-input treatment | Forward mean | Change vs finalized training | Retrospective mean | Change vs finalized training |
| --- | --- | ---: | ---: | ---: | ---: |
| C1 | Finalized training inputs | 0.93984 | +0.00% | 0.93467 | +0.00% |
| C1 | Empirical half-strength, all three training seasons | 0.95561 | +1.68% | 0.85018 | -9.04% |
| C1 | Empirical half-strength early two; recent finalized | 0.94604 | +0.66% | 0.89547 | -4.19% |
| C1 | Empirical half-strength early two; recent actual reports | 0.97181 | +3.40% | 0.85541 | -8.48% |
| C1 | Empirical full-strength early two; recent actual reports | 1.04780 | +11.49% | 0.83643 | -10.51% |
| C1 | Learned conditional mean and residual early two; recent actual reports | 0.99428 | +5.79% | 0.83671 | -10.48% |
| C1 | Learned conditional mean only early two; recent actual reports | 1.00186 | +6.60% | 0.88107 | -5.73% |
| C1 | Learned conditional mean and residual all three | 0.99093 | +5.44% | 0.88178 | -5.66% |
| C2 | Finalized training inputs | 0.98733 | +0.00% | 0.93397 | +0.00% |
| C2 | Empirical half-strength, all three training seasons | 0.98778 | +0.05% | 1.00346 | +7.44% |
| C2 | Empirical half-strength early two; recent finalized | 0.97395 | -1.36% | 0.99514 | +6.55% |
| C2 | Empirical half-strength early two; recent actual reports | 0.97874 | -0.87% | 0.84579 | -9.44% |
| C2 | Empirical full-strength early two; recent actual reports | 1.03999 | +5.33% | 0.91627 | -1.89% |
| C2 | Learned conditional mean and residual early two; recent actual reports | 1.07712 | +9.09% | 0.92821 | -0.62% |
| C2 | Learned conditional mean only early two; recent actual reports | 0.99293 | +0.57% | 0.87536 | -6.27% |
| C2 | Learned conditional mean and residual all three | 0.98212 | -0.53% | 0.91591 | -1.93% |
| C3 | Finalized training inputs | 0.97501 | +0.00% | 0.97920 | +0.00% |
| C3 | Empirical half-strength, all three training seasons | 0.99536 | +2.09% | 0.88429 | -9.69% |
| C3 | Empirical half-strength early two; recent finalized | 0.97765 | +0.27% | 0.96151 | -1.81% |
| C3 | Empirical half-strength early two; recent actual reports | 0.99396 | +1.94% | 0.80365 | -17.93% |
| C3 | Empirical full-strength early two; recent actual reports | 0.95681 | -1.87% | 0.83005 | -15.23% |
| C3 | Learned conditional mean and residual early two; recent actual reports | 0.95873 | -1.67% | 0.82556 | -15.69% |
| C3 | Learned conditional mean only early two; recent actual reports | 0.94960 | -2.61% | 0.83555 | -14.67% |
| C3 | Learned conditional mean and residual all three | 0.98640 | +1.17% | 0.87981 | -10.15% |
| C1 | Finalized early two; actual recent reports | 0.98968 | +5.30% | 0.86245 | -7.73% |
| C1 | Empirical half-strength admissions early two; actual recent reports | 0.97425 | +3.66% | 0.81572 | -12.73% |
| C1 | Learned half-strength admissions early two; actual recent reports | 0.95857 | +1.99% | 0.86100 | -7.88% |
| C1 | Empirical half-strength admissions all three; ED and covariates finalized | 0.94444 | +0.49% | 0.87064 | -6.85% |
| C2 | Finalized early two; actual recent reports | 0.96640 | -2.12% | 0.90910 | -2.66% |
| C2 | Empirical half-strength admissions early two; actual recent reports | 1.00168 | +1.45% | 0.86422 | -7.47% |
| C2 | Learned half-strength admissions early two; actual recent reports | 0.97044 | -1.71% | 0.84737 | -9.27% |
| C2 | Empirical half-strength admissions all three; ED and covariates finalized | 0.96053 | -2.72% | 0.90603 | -2.99% |
| C3 | Finalized early two; actual recent reports | 0.94373 | -3.21% | 0.86213 | -11.96% |
| C3 | Empirical half-strength admissions early two; actual recent reports | 0.94058 | -3.53% | 0.83249 | -14.98% |
| C3 | Learned half-strength admissions early two; actual recent reports | 0.94701 | -2.87% | 0.84071 | -14.14% |
| C3 | Empirical half-strength admissions all three; ED and covariates finalized | 0.96745 | -0.77% | 0.92468 | -5.57% |
| C1 | Half-clean episodes; half-strength empirical errors all three | 0.94765 | +0.83% | 0.86518 | -7.43% |
