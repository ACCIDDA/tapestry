# Completed covariate and correction-window refinement

All 24 configurations × two seeds completed successfully at 09:18 EDT, about 20 minutes after launch. Common-manager ranking: `ranking-8717617b218d`. [Full protocol and manager commands](../index.md). No configuration improved overall on the original two leaders.

All models are independently retrained joint flu MLPs, learning finalized four-week flu admissions/ED labels. Recipe A trains on artificial reports corrected by cross-fitted real-pair trees; recipe B trains with artificial reports and recent finalized reconstruction. Forward training uses 2022–23, 2023–24, 2024–25 and evaluates 2025–26. Retrospective training uses 2022–23, 2023–24, 2025–26 and evaluates 2024–25. Evaluation uses Wednesday/holiday reported histories with finalized fills for missing archives and tree-corrected recent flu admissions; ED stays reported. The experiment crosses each recipe's unchanged architecture/hyperparameters with covariate choices and 2/4/8-week correction windows. For A, correction-window changes affect training and evaluation corrections; for B, they affect evaluation correction. Contemporary ILINet uses its existing publication lag and is distinct from historical ILI pretraining.

## Two-week correction results

Scores are two-seed means. Admissions native/log average both seasons equally. ED ensemble-relative WIS has only the forward season. Lower is better; 1 is the frozen ensemble. Composite has no retrospective ED support.

| Recipe / covariates | Native admissions | Log admissions | Forward ED | Composite |
|---|---:|---:|---:|---:|
| A / Kinsa (original) | 0.7724 | 0.8480 | 0.7871 | 0.7799 |
| A / Kinsa + ILINet | 0.7897 | 0.8424 | 0.8902 | 0.7980 |
| A / ILINet | 0.8893 | 0.8941 | 0.8770 | 0.8879 |
| A / none | 0.9126 | 0.9441 | 0.9538 | 0.9178 |
| B / none (original) | 0.7918 | 0.7953 | 0.8941 | 0.7956 |
| B / ILINet | 0.9037 | 0.8946 | 0.8970 | 0.8981 |
| B / Kinsa + ILINet | 0.9646 | 0.9556 | 0.8696 | 0.9566 |
| B / Kinsa | 1.0223 | 1.0104 | 0.8869 | 1.0051 |

Adding ILINet to A with Kinsa slightly improves mean log admissions (~0.7%), but harms native admissions (~2.2%) and forward ED (~13.1%). It helps retrospective admissions but hurts forward admissions: native 2024–25 0.8026 → 0.7389, 2025–26 0.7422 → 0.8405; log 0.8089 → 0.7721 and 0.8871 → 0.9127. This is a season tradeoff, not an overall improvement. On B, additional covariates substantially worsen admissions. Covariate additions change the learned context dimensions and initialization, so two seeds do not isolate a universal causal effect of a data source.

Two-week correction is the best choice for each original recipe. With A+Kinsa, composite worsens from 0.7799 to 0.8610 (four weeks) and 0.9495 (eight weeks). With B and no covariates, it changes only slightly: 0.7956 → 0.7969 → 0.7972. Thus correcting older weeks brings no benefit here; A's larger changes include retraining on different corrected histories, not merely replaying the same model with a new evaluation window.

Raw ED was checked in both seasons. Adding ILINet to A+Kinsa slightly improves retrospective ED (states/DC 0.006035 → 0.005989, US 0.005162 → 0.005127) but worsens forward ED (0.005070 → 0.005740 and 0.004069 → 0.004798). For B, Kinsa+ILINet modestly improves forward ED but worsens retrospective ED and both admission scores. These comparisons preserve finalized prediction labels, seasonal folds and the same two-week corrected evaluation rule.

Recommendation: retain A+Kinsa and B without extra covariates, both with two-week correction. No additional jobs launched. Exact per-view, season and seed results are in labeled-summary.csv, labeled-seeds.csv and pilot-raw-wis.csv.
