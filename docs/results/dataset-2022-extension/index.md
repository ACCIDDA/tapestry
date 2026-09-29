# Dataset extended with the 2022–23 season

The working base dataset, `data/processed/panel.npz`, now covers **2022-05-14 through 2026-09-19** (228 weekly observations; 228 nominal Wednesday rounds). The first 12 weeks precede the first Saturday of 2022–23 (August 6, 2022). They are stored as context; the existing cross-validation policy excludes seasons outside the declared training seasons from fitted histories.

**2022–23 is added to training**, while the evaluation seasons remain 2023–24, 2024–25 and 2025–26. Missing outcomes are masked by the existing loss policy. A fourth scored fold and its Hub benchmark were not created. No models were trained or rescored by this dataset update.

## Target and covariate coverage in 2022–23

Counts are retrospective finite values in the rebuilt dataset. “Weeks with values” means at least one native location is present, not complete state coverage. Missing covariates and outcomes remain NaN. National Kinsa is stored once.

| series | weeks_with_values | season_weeks | finite_cells | total_cells | first_week | last_week |
| --- | --- | --- | --- | --- | --- | --- |
| nhsn_flu_admissions | 52 | 52 | 2704 | 2704 | 2022-08-06 | 2023-07-29 |
| nhsn_covid_admissions | 52 | 52 | 2704 | 2704 | 2022-08-06 | 2023-07-29 |
| nhsn_rsv_admissions | 0 | 52 | 0 | 2704 | — | — |
| nssp_flu_proportion | 44 | 52 | 2288 | 2704 | 2022-10-01 | 2023-07-29 |
| nssp_covid_proportion | 44 | 52 | 2288 | 2704 | 2022-10-01 | 2023-07-29 |
| nssp_rsv_proportion | 44 | 52 | 2288 | 2704 | 2022-10-01 | 2023-07-29 |
| inpatient_flu | 52 | 52 | 2609 | 2704 | 2022-08-06 | 2023-07-29 |
| inpatient_covid | 52 | 52 | 2616 | 2704 | 2022-08-06 | 2023-07-29 |
| outpatient_flu | 52 | 52 | 2704 | 2704 | 2022-08-06 | 2023-07-29 |
| outpatient_covid | 52 | 52 | 2704 | 2704 | 2022-08-06 | 2023-07-29 |
| nwss_flu_wval_like | 52 | 52 | 845 | 2704 | 2022-08-06 | 2023-07-29 |
| nwss_covid_wval_like | 52 | 52 | 2426 | 2704 | 2022-08-06 | 2023-07-29 |
| nwss_rsv_wval_like | 52 | 52 | 759 | 2704 | 2022-08-06 | 2023-07-29 |
| nwss_flu_pct_rank | 52 | 52 | 845 | 2704 | 2022-08-06 | 2023-07-29 |
| nwss_covid_pct_rank | 52 | 52 | 2426 | 2704 | 2022-08-06 | 2023-07-29 |
| nwss_rsv_pct_rank | 52 | 52 | 759 | 2704 | 2022-08-06 | 2023-07-29 |
| ilinet_ili | 52 | 52 | 2704 | 2704 | 2022-08-06 | 2023-07-29 |
| clinical_lab_flu_pct_positive | 52 | 52 | 2060 | 2704 | 2022-08-06 | 2023-07-29 |
| flusurv_flu_rate | 30 | 52 | 390 | 2704 | 2022-10-08 | 2023-04-29 |
| kinsa_ili | 52 | 52 | 52 | 52 | 2022-08-06 | 2023-07-29 |

## Finalized NHSN policy

Finite nonnegative CDC finalized admission counts take precedence for retrospective target values; native `USA` maps to `US`. Where the finalized snapshot has no finite value, the previous archive-resolved value remains. The override is applied to final targets only, never historical as-of arrays. This preserves the distinction between complete retrospective training information and evidence of historical availability.

Changes to NHSN values within the previously existing calendar are recorded below; these are consequences of explicit finalized-source precedence, not of adding the season alone.

| target | changed_existing_cells |
| --- | --- |
| nhsn_flu_admissions | 0 |
| nhsn_covid_admissions | 0 |
| nhsn_rsv_admissions | 0 |

All overlapping covariates, ED final values and historical as-of arrays were verified unchanged. Future observation weeks remain empty in every historical as-of array. Training-role checks confirm the added season is available for fitting and each evaluation season remains excluded from its own fit. The source snapshot IDs and policy are recorded in dataset metadata; truth resolution remains pinned to September 22, 2026 to avoid mixing this extension with a source refresh.

The seven covariate groups retain the existing extraction rules. In particular claims conflict rejection has **not** been resolved by this extension, so finalized covariate histories can still have those known gaps. No assumed availability is inserted into the base archive panel. The separate `panel-b2-deadline.npz` and `panel-b2-operational.npz` used by the completed experiment are unchanged and must be rebuilt against this expanded base before a new operational experiment uses 2022–23. Saved experiment pins and reported scores are not modified.

SHA256: `54d0e3306a76ded067c6d1b65493d50b1c19bde2e66ab80a438674ec37e6c80f`.

[Coverage CSV](coverage.csv) · [Verification record](verification.json) · [Finalized NHSN changes](finalized-nhsn-changes.csv).

Rebuild the base panel with the same truth cutoff:

```bash
.venv/bin/python -m tapestry.dataset.build build --start 2022-05-14 --truth-day 2026-09-22 --workers 3 --output data/processed/panel.npz
```
