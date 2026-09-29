# Every modeled covariate: availability and missing states/weeks

**[See the season timeline heatmaps](timeline.md)** — observation week across, submission date upward; hover to identify missing states in the interactive explorer.


All **14 covariate series** in the frozen panel are included: four claims signals, six wastewater indices (WVAL-like and percentile for each pathogen), ILI, clinical-lab positivity, FluSurv, and national Kinsa. These are the panel's full covariate set, not every signal in the broader data catalog. The B2 grid used WVAL-like wastewater, not its percentile alternatives.

Availability means at least one finite value in the latest eligible source statement, including finite claims conflicts discarded by the model pipeline. No finalized proxies are used. The 53 deadlines and holiday adjustments are unchanged. Counts use each source's fixed native geography. Unsupported states are listed on each source's detail page, excluded from the denominator, and **not silently counted as complete**.

- **—** means the source has no native support for that geography, not zero missing weeks.
- **Missing states:** unique members of the 50 states with at least one missing value at this lag. DC and the native US series are separate columns.
- **Distinct weeks, states:** unique submission dates with at least one missing state. The same week missing in 20 states counts once here but 20 times in missing location-weeks.
- **Per-state missing weeks:** each source link opens a state-by-lag table. Each count is the number of distinct missing submission dates for that state at that lag. Exact submission and observation dates are downloadable. At a fixed lag these date counts are identical; **do not sum across lags to obtain unique weeks**.
- **Location-weeks checked:** native locations × 53 deadlines; this is not the number of finite revision pairs. All-season archive gaps remain in the denominator.

[Full summary CSV](full-availability-summary.csv) · [Every state's counts and exact dates](missing-by-state-lag.csv) · [All missing cells](missing-cells.csv.gz) · [All-lag conditional revisions](full-revisions.md) · [Conflict diagnosis](conflict-audit.md) · [Main availability report](index.md).

## Lag 0: latest observation week

| Covariate | Frequency / representation | Reported / checked | Missing states (of 50) | Distinct weeks, states | DC missing weeks | US missing weeks | Missing location-weeks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [inpatient_flu](inpatient_flu.md) | Daily; Saturday trailing-7-day value | 2109/2703 (78.0%) | 49 | 53 | 42 | 4 | 594 |
| [inpatient_covid](inpatient_covid.md) | Daily; Saturday trailing-7-day value | 2109/2703 (78.0%) | 49 | 53 | 42 | 4 | 594 |
| [outpatient_flu](outpatient_flu.md) | Daily; Saturday trailing-7-day value | 2520/2756 (91.4%) | 50 | 13 | 4 | 4 | 236 |
| [outpatient_covid](outpatient_covid.md) | Daily; Saturday trailing-7-day value | 2520/2756 (91.4%) | 50 | 13 | 4 | 4 | 236 |
| [nwss_flu_wval_like](nwss_flu_wval_like.md) | Weekly derived index; irregular samples | 339/2597 (13.1%) | 48 | 53 | — | 38 | 2258 |
| [nwss_covid_wval_like](nwss_covid_wval_like.md) | Weekly derived index; irregular samples | 326/2703 (12.1%) | 49 | 53 | 53 | 40 | 2377 |
| [nwss_rsv_wval_like](nwss_rsv_wval_like.md) | Weekly derived index; irregular samples | 443/2597 (17.1%) | 48 | 53 | — | 31 | 2154 |
| [nwss_flu_pct_rank](nwss_flu_pct_rank.md) | Weekly derived index; irregular samples | 339/2597 (13.1%) | 48 | 53 | — | 38 | 2258 |
| [nwss_covid_pct_rank](nwss_covid_pct_rank.md) | Weekly derived index; irregular samples | 326/2703 (12.1%) | 49 | 53 | 53 | 40 | 2377 |
| [nwss_rsv_pct_rank](nwss_rsv_pct_rank.md) | Weekly derived index; irregular samples | 443/2597 (17.1%) | 48 | 53 | — | 31 | 2154 |
| [ilinet_ili](ilinet_ili.md) | Weekly | 52/2756 (1.9%) | 50 | 52 | 52 | 52 | 2704 |
| [clinical_lab_flu_pct_positive](clinical_lab_flu_pct_positive.md) | Weekly | 38/2385 (1.6%) | 44 | 53 | — | 52 | 2347 |
| [flusurv_flu_rate](flusurv_flu_rate.md) | Weekly | 26/742 (3.5%) | 13 | 53 | — | 51 | 716 |
| [kinsa_ili](kinsa_ili.md) | Daily; complete weekly mean | 14/53 (26.4%) | — | — | — | 39 | 39 |

## Lag 1: 1 observation weeks earlier

| Covariate | Frequency / representation | Reported / checked | Missing states (of 50) | Distinct weeks, states | DC missing weeks | US missing weeks | Missing location-weeks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [inpatient_flu](inpatient_flu.md) | Daily; Saturday trailing-7-day value | 2329/2703 (86.2%) | 49 | 52 | 13 | 3 | 374 |
| [inpatient_covid](inpatient_covid.md) | Daily; Saturday trailing-7-day value | 2329/2703 (86.2%) | 49 | 52 | 13 | 3 | 374 |
| [outpatient_flu](outpatient_flu.md) | Daily; Saturday trailing-7-day value | 2597/2756 (94.2%) | 50 | 5 | 3 | 3 | 159 |
| [outpatient_covid](outpatient_covid.md) | Daily; Saturday trailing-7-day value | 2597/2756 (94.2%) | 50 | 5 | 3 | 3 | 159 |
| [nwss_flu_wval_like](nwss_flu_wval_like.md) | Weekly derived index; irregular samples | 805/2597 (31.0%) | 48 | 53 | — | 29 | 1792 |
| [nwss_covid_wval_like](nwss_covid_wval_like.md) | Weekly derived index; irregular samples | 774/2703 (28.6%) | 49 | 53 | 53 | 29 | 1929 |
| [nwss_rsv_wval_like](nwss_rsv_wval_like.md) | Weekly derived index; irregular samples | 948/2597 (36.5%) | 48 | 53 | — | 29 | 1649 |
| [nwss_flu_pct_rank](nwss_flu_pct_rank.md) | Weekly derived index; irregular samples | 805/2597 (31.0%) | 48 | 53 | — | 29 | 1792 |
| [nwss_covid_pct_rank](nwss_covid_pct_rank.md) | Weekly derived index; irregular samples | 774/2703 (28.6%) | 49 | 53 | 53 | 29 | 1929 |
| [nwss_rsv_pct_rank](nwss_rsv_pct_rank.md) | Weekly derived index; irregular samples | 948/2597 (36.5%) | 48 | 53 | — | 29 | 1649 |
| [ilinet_ili](ilinet_ili.md) | Weekly | 2001/2756 (72.6%) | 50 | 44 | 6 | 6 | 755 |
| [clinical_lab_flu_pct_positive](clinical_lab_flu_pct_positive.md) | Weekly | 1476/2385 (61.9%) | 44 | 53 | — | 6 | 909 |
| [flusurv_flu_rate](flusurv_flu_rate.md) | Weekly | 351/742 (47.3%) | 13 | 53 | — | 26 | 391 |
| [kinsa_ili](kinsa_ili.md) | Daily; complete weekly mean | 15/53 (28.3%) | — | — | — | 38 | 38 |

## Lag 2: 2 observation weeks earlier

| Covariate | Frequency / representation | Reported / checked | Missing states (of 50) | Distinct weeks, states | DC missing weeks | US missing weeks | Missing location-weeks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [inpatient_flu](inpatient_flu.md) | Daily; Saturday trailing-7-day value | 2410/2703 (89.2%) | 49 | 50 | 7 | 2 | 293 |
| [inpatient_covid](inpatient_covid.md) | Daily; Saturday trailing-7-day value | 2410/2703 (89.2%) | 49 | 50 | 7 | 2 | 293 |
| [outpatient_flu](outpatient_flu.md) | Daily; Saturday trailing-7-day value | 2649/2756 (96.1%) | 50 | 4 | 2 | 2 | 107 |
| [outpatient_covid](outpatient_covid.md) | Daily; Saturday trailing-7-day value | 2649/2756 (96.1%) | 50 | 4 | 2 | 2 | 107 |
| [nwss_flu_wval_like](nwss_flu_wval_like.md) | Weekly derived index; irregular samples | 1012/2597 (39.0%) | 48 | 53 | — | 29 | 1585 |
| [nwss_covid_wval_like](nwss_covid_wval_like.md) | Weekly derived index; irregular samples | 1059/2703 (39.2%) | 49 | 53 | 53 | 29 | 1644 |
| [nwss_rsv_wval_like](nwss_rsv_wval_like.md) | Weekly derived index; irregular samples | 1021/2597 (39.3%) | 48 | 53 | — | 29 | 1576 |
| [nwss_flu_pct_rank](nwss_flu_pct_rank.md) | Weekly derived index; irregular samples | 1012/2597 (39.0%) | 48 | 53 | — | 29 | 1585 |
| [nwss_covid_pct_rank](nwss_covid_pct_rank.md) | Weekly derived index; irregular samples | 1059/2703 (39.2%) | 49 | 53 | 53 | 29 | 1644 |
| [nwss_rsv_pct_rank](nwss_rsv_pct_rank.md) | Weekly derived index; irregular samples | 1021/2597 (39.3%) | 48 | 53 | — | 29 | 1576 |
| [ilinet_ili](ilinet_ili.md) | Weekly | 2054/2756 (74.5%) | 50 | 43 | 5 | 5 | 702 |
| [clinical_lab_flu_pct_positive](clinical_lab_flu_pct_positive.md) | Weekly | 1607/2385 (67.4%) | 44 | 53 | — | 5 | 778 |
| [flusurv_flu_rate](flusurv_flu_rate.md) | Weekly | 351/742 (47.3%) | 13 | 53 | — | 26 | 391 |
| [kinsa_ili](kinsa_ili.md) | Daily; complete weekly mean | 16/53 (30.2%) | — | — | — | 37 | 37 |

## Lag 3: 3 observation weeks earlier

| Covariate | Frequency / representation | Reported / checked | Missing states (of 50) | Distinct weeks, states | DC missing weeks | US missing weeks | Missing location-weeks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [inpatient_flu](inpatient_flu.md) | Daily; Saturday trailing-7-day value | 2477/2703 (91.6%) | 49 | 49 | 2 | 1 | 226 |
| [inpatient_covid](inpatient_covid.md) | Daily; Saturday trailing-7-day value | 2477/2703 (91.6%) | 49 | 49 | 2 | 1 | 226 |
| [outpatient_flu](outpatient_flu.md) | Daily; Saturday trailing-7-day value | 2701/2756 (98.0%) | 50 | 3 | 1 | 1 | 55 |
| [outpatient_covid](outpatient_covid.md) | Daily; Saturday trailing-7-day value | 2701/2756 (98.0%) | 50 | 3 | 1 | 1 | 55 |
| [nwss_flu_wval_like](nwss_flu_wval_like.md) | Weekly derived index; irregular samples | 1039/2597 (40.0%) | 48 | 53 | — | 29 | 1558 |
| [nwss_covid_wval_like](nwss_covid_wval_like.md) | Weekly derived index; irregular samples | 1093/2703 (40.4%) | 49 | 53 | 53 | 29 | 1610 |
| [nwss_rsv_wval_like](nwss_rsv_wval_like.md) | Weekly derived index; irregular samples | 1037/2597 (39.9%) | 48 | 53 | — | 29 | 1560 |
| [nwss_flu_pct_rank](nwss_flu_pct_rank.md) | Weekly derived index; irregular samples | 1039/2597 (40.0%) | 48 | 53 | — | 29 | 1558 |
| [nwss_covid_pct_rank](nwss_covid_pct_rank.md) | Weekly derived index; irregular samples | 1093/2703 (40.4%) | 49 | 53 | 53 | 29 | 1610 |
| [nwss_rsv_pct_rank](nwss_rsv_pct_rank.md) | Weekly derived index; irregular samples | 1037/2597 (39.9%) | 48 | 53 | — | 29 | 1560 |
| [ilinet_ili](ilinet_ili.md) | Weekly | 2107/2756 (76.5%) | 50 | 42 | 4 | 4 | 649 |
| [clinical_lab_flu_pct_positive](clinical_lab_flu_pct_positive.md) | Weekly | 1669/2385 (70.0%) | 44 | 53 | — | 4 | 716 |
| [flusurv_flu_rate](flusurv_flu_rate.md) | Weekly | 351/742 (47.3%) | 13 | 53 | — | 26 | 391 |
| [kinsa_ili](kinsa_ili.md) | Daily; complete weekly mean | 17/53 (32.1%) | — | — | — | 36 | 36 |

## Lag 4: 4 observation weeks earlier

| Covariate | Frequency / representation | Reported / checked | Missing states (of 50) | Distinct weeks, states | DC missing weeks | US missing weeks | Missing location-weeks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [inpatient_flu](inpatient_flu.md) | Daily; Saturday trailing-7-day value | 2534/2703 (93.7%) | 45 | 49 | 1 | 0 | 169 |
| [inpatient_covid](inpatient_covid.md) | Daily; Saturday trailing-7-day value | 2534/2703 (93.7%) | 45 | 49 | 1 | 0 | 169 |
| [outpatient_flu](outpatient_flu.md) | Daily; Saturday trailing-7-day value | 2753/2756 (99.9%) | 2 | 2 | 0 | 0 | 3 |
| [outpatient_covid](outpatient_covid.md) | Daily; Saturday trailing-7-day value | 2753/2756 (99.9%) | 2 | 2 | 0 | 0 | 3 |
| [nwss_flu_wval_like](nwss_flu_wval_like.md) | Weekly derived index; irregular samples | 1056/2597 (40.7%) | 48 | 53 | — | 29 | 1541 |
| [nwss_covid_wval_like](nwss_covid_wval_like.md) | Weekly derived index; irregular samples | 1107/2703 (41.0%) | 49 | 53 | 53 | 29 | 1596 |
| [nwss_rsv_wval_like](nwss_rsv_wval_like.md) | Weekly derived index; irregular samples | 1046/2597 (40.3%) | 48 | 53 | — | 29 | 1551 |
| [nwss_flu_pct_rank](nwss_flu_pct_rank.md) | Weekly derived index; irregular samples | 1056/2597 (40.7%) | 48 | 53 | — | 29 | 1541 |
| [nwss_covid_pct_rank](nwss_covid_pct_rank.md) | Weekly derived index; irregular samples | 1107/2703 (41.0%) | 49 | 53 | 53 | 29 | 1596 |
| [nwss_rsv_pct_rank](nwss_rsv_pct_rank.md) | Weekly derived index; irregular samples | 1046/2597 (40.3%) | 48 | 53 | — | 29 | 1551 |
| [ilinet_ili](ilinet_ili.md) | Weekly | 2160/2756 (78.4%) | 50 | 41 | 3 | 3 | 596 |
| [clinical_lab_flu_pct_positive](clinical_lab_flu_pct_positive.md) | Weekly | 1714/2385 (71.9%) | 44 | 53 | — | 3 | 671 |
| [flusurv_flu_rate](flusurv_flu_rate.md) | Weekly | 351/742 (47.3%) | 13 | 53 | — | 26 | 391 |
| [kinsa_ili](kinsa_ili.md) | Daily; complete weekly mean | 18/53 (34.0%) | — | — | — | 35 | 35 |

## Lag 5: 5 observation weeks earlier

| Covariate | Frequency / representation | Reported / checked | Missing states (of 50) | Distinct weeks, states | DC missing weeks | US missing weeks | Missing location-weeks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [inpatient_flu](inpatient_flu.md) | Daily; Saturday trailing-7-day value | 2583/2703 (95.6%) | 6 | 48 | 0 | 0 | 120 |
| [inpatient_covid](inpatient_covid.md) | Daily; Saturday trailing-7-day value | 2583/2703 (95.6%) | 6 | 48 | 0 | 0 | 120 |
| [outpatient_flu](outpatient_flu.md) | Daily; Saturday trailing-7-day value | 2755/2756 (100.0%) | 1 | 1 | 0 | 0 | 1 |
| [outpatient_covid](outpatient_covid.md) | Daily; Saturday trailing-7-day value | 2755/2756 (100.0%) | 1 | 1 | 0 | 0 | 1 |
| [nwss_flu_wval_like](nwss_flu_wval_like.md) | Weekly derived index; irregular samples | 1070/2597 (41.2%) | 48 | 53 | — | 29 | 1527 |
| [nwss_covid_wval_like](nwss_covid_wval_like.md) | Weekly derived index; irregular samples | 1120/2703 (41.4%) | 49 | 53 | 53 | 29 | 1583 |
| [nwss_rsv_wval_like](nwss_rsv_wval_like.md) | Weekly derived index; irregular samples | 1054/2597 (40.6%) | 48 | 53 | — | 29 | 1543 |
| [nwss_flu_pct_rank](nwss_flu_pct_rank.md) | Weekly derived index; irregular samples | 1070/2597 (41.2%) | 48 | 53 | — | 29 | 1527 |
| [nwss_covid_pct_rank](nwss_covid_pct_rank.md) | Weekly derived index; irregular samples | 1120/2703 (41.4%) | 49 | 53 | 53 | 29 | 1583 |
| [nwss_rsv_pct_rank](nwss_rsv_pct_rank.md) | Weekly derived index; irregular samples | 1054/2597 (40.6%) | 48 | 53 | — | 29 | 1543 |
| [ilinet_ili](ilinet_ili.md) | Weekly | 2213/2756 (80.3%) | 50 | 40 | 2 | 2 | 543 |
| [clinical_lab_flu_pct_positive](clinical_lab_flu_pct_positive.md) | Weekly | 1759/2385 (73.8%) | 44 | 53 | — | 2 | 626 |
| [flusurv_flu_rate](flusurv_flu_rate.md) | Weekly | 351/742 (47.3%) | 13 | 53 | — | 26 | 391 |
| [kinsa_ili](kinsa_ili.md) | Daily; complete weekly mean | 18/53 (34.0%) | — | — | — | 35 | 35 |

## Lag 6: 6 observation weeks earlier

| Covariate | Frequency / representation | Reported / checked | Missing states (of 50) | Distinct weeks, states | DC missing weeks | US missing weeks | Missing location-weeks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [inpatient_flu](inpatient_flu.md) | Daily; Saturday trailing-7-day value | 2588/2703 (95.7%) | 5 | 48 | 0 | 0 | 115 |
| [inpatient_covid](inpatient_covid.md) | Daily; Saturday trailing-7-day value | 2588/2703 (95.7%) | 5 | 48 | 0 | 0 | 115 |
| [outpatient_flu](outpatient_flu.md) | Daily; Saturday trailing-7-day value | 2755/2756 (100.0%) | 1 | 1 | 0 | 0 | 1 |
| [outpatient_covid](outpatient_covid.md) | Daily; Saturday trailing-7-day value | 2755/2756 (100.0%) | 1 | 1 | 0 | 0 | 1 |
| [nwss_flu_wval_like](nwss_flu_wval_like.md) | Weekly derived index; irregular samples | 1074/2597 (41.4%) | 48 | 53 | — | 29 | 1523 |
| [nwss_covid_wval_like](nwss_covid_wval_like.md) | Weekly derived index; irregular samples | 1125/2703 (41.6%) | 49 | 53 | 53 | 29 | 1578 |
| [nwss_rsv_wval_like](nwss_rsv_wval_like.md) | Weekly derived index; irregular samples | 1058/2597 (40.7%) | 48 | 53 | — | 29 | 1539 |
| [nwss_flu_pct_rank](nwss_flu_pct_rank.md) | Weekly derived index; irregular samples | 1074/2597 (41.4%) | 48 | 53 | — | 29 | 1523 |
| [nwss_covid_pct_rank](nwss_covid_pct_rank.md) | Weekly derived index; irregular samples | 1125/2703 (41.6%) | 49 | 53 | 53 | 29 | 1578 |
| [nwss_rsv_pct_rank](nwss_rsv_pct_rank.md) | Weekly derived index; irregular samples | 1058/2597 (40.7%) | 48 | 53 | — | 29 | 1539 |
| [ilinet_ili](ilinet_ili.md) | Weekly | 2266/2756 (82.2%) | 50 | 39 | 1 | 1 | 490 |
| [clinical_lab_flu_pct_positive](clinical_lab_flu_pct_positive.md) | Weekly | 1800/2385 (75.5%) | 44 | 53 | — | 1 | 585 |
| [flusurv_flu_rate](flusurv_flu_rate.md) | Weekly | 351/742 (47.3%) | 13 | 53 | — | 26 | 391 |
| [kinsa_ili](kinsa_ili.md) | Daily; complete weekly mean | 18/53 (34.0%) | — | — | — | 35 | 35 |

## Lag 7: 7 observation weeks earlier

| Covariate | Frequency / representation | Reported / checked | Missing states (of 50) | Distinct weeks, states | DC missing weeks | US missing weeks | Missing location-weeks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [inpatient_flu](inpatient_flu.md) | Daily; Saturday trailing-7-day value | 2596/2703 (96.0%) | 5 | 48 | 0 | 0 | 107 |
| [inpatient_covid](inpatient_covid.md) | Daily; Saturday trailing-7-day value | 2596/2703 (96.0%) | 5 | 48 | 0 | 0 | 107 |
| [outpatient_flu](outpatient_flu.md) | Daily; Saturday trailing-7-day value | 2755/2756 (100.0%) | 1 | 1 | 0 | 0 | 1 |
| [outpatient_covid](outpatient_covid.md) | Daily; Saturday trailing-7-day value | 2755/2756 (100.0%) | 1 | 1 | 0 | 0 | 1 |
| [nwss_flu_wval_like](nwss_flu_wval_like.md) | Weekly derived index; irregular samples | 1078/2597 (41.5%) | 48 | 53 | — | 29 | 1519 |
| [nwss_covid_wval_like](nwss_covid_wval_like.md) | Weekly derived index; irregular samples | 1129/2703 (41.8%) | 49 | 53 | 53 | 29 | 1574 |
| [nwss_rsv_wval_like](nwss_rsv_wval_like.md) | Weekly derived index; irregular samples | 1063/2597 (40.9%) | 48 | 53 | — | 29 | 1534 |
| [nwss_flu_pct_rank](nwss_flu_pct_rank.md) | Weekly derived index; irregular samples | 1078/2597 (41.5%) | 48 | 53 | — | 29 | 1519 |
| [nwss_covid_pct_rank](nwss_covid_pct_rank.md) | Weekly derived index; irregular samples | 1129/2703 (41.8%) | 49 | 53 | 53 | 29 | 1574 |
| [nwss_rsv_pct_rank](nwss_rsv_pct_rank.md) | Weekly derived index; irregular samples | 1063/2597 (40.9%) | 48 | 53 | — | 29 | 1534 |
| [ilinet_ili](ilinet_ili.md) | Weekly | 2341/2756 (84.9%) | 22 | 37 | 0 | 0 | 415 |
| [clinical_lab_flu_pct_positive](clinical_lab_flu_pct_positive.md) | Weekly | 1858/2385 (77.9%) | 25 | 53 | — | 0 | 527 |
| [flusurv_flu_rate](flusurv_flu_rate.md) | Weekly | 351/742 (47.3%) | 13 | 53 | — | 26 | 391 |
| [kinsa_ili](kinsa_ili.md) | Daily; complete weekly mean | 18/53 (34.0%) | — | — | — | 35 | 35 |

## Lag 8: 8 observation weeks earlier

| Covariate | Frequency / representation | Reported / checked | Missing states (of 50) | Distinct weeks, states | DC missing weeks | US missing weeks | Missing location-weeks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [inpatient_flu](inpatient_flu.md) | Daily; Saturday trailing-7-day value | 2603/2703 (96.3%) | 5 | 47 | 0 | 0 | 100 |
| [inpatient_covid](inpatient_covid.md) | Daily; Saturday trailing-7-day value | 2603/2703 (96.3%) | 5 | 47 | 0 | 0 | 100 |
| [outpatient_flu](outpatient_flu.md) | Daily; Saturday trailing-7-day value | 2755/2756 (100.0%) | 1 | 1 | 0 | 0 | 1 |
| [outpatient_covid](outpatient_covid.md) | Daily; Saturday trailing-7-day value | 2755/2756 (100.0%) | 1 | 1 | 0 | 0 | 1 |
| [nwss_flu_wval_like](nwss_flu_wval_like.md) | Weekly derived index; irregular samples | 1081/2597 (41.6%) | 48 | 53 | — | 29 | 1516 |
| [nwss_covid_wval_like](nwss_covid_wval_like.md) | Weekly derived index; irregular samples | 1131/2703 (41.8%) | 49 | 53 | 53 | 29 | 1572 |
| [nwss_rsv_wval_like](nwss_rsv_wval_like.md) | Weekly derived index; irregular samples | 1068/2597 (41.1%) | 48 | 53 | — | 29 | 1529 |
| [nwss_flu_pct_rank](nwss_flu_pct_rank.md) | Weekly derived index; irregular samples | 1081/2597 (41.6%) | 48 | 53 | — | 29 | 1516 |
| [nwss_covid_pct_rank](nwss_covid_pct_rank.md) | Weekly derived index; irregular samples | 1131/2703 (41.8%) | 49 | 53 | 53 | 29 | 1572 |
| [nwss_rsv_pct_rank](nwss_rsv_pct_rank.md) | Weekly derived index; irregular samples | 1068/2597 (41.1%) | 48 | 53 | — | 29 | 1529 |
| [ilinet_ili](ilinet_ili.md) | Weekly | 2363/2756 (85.7%) | 22 | 36 | 0 | 0 | 393 |
| [clinical_lab_flu_pct_positive](clinical_lab_flu_pct_positive.md) | Weekly | 1880/2385 (78.8%) | 24 | 53 | — | 0 | 505 |
| [flusurv_flu_rate](flusurv_flu_rate.md) | Weekly | 351/742 (47.3%) | 13 | 53 | — | 26 | 391 |
| [kinsa_ili](kinsa_ili.md) | Daily; complete weekly mean | 18/53 (34.0%) | — | — | — | 35 | 35 |

## Lag 9: 9 observation weeks earlier

| Covariate | Frequency / representation | Reported / checked | Missing states (of 50) | Distinct weeks, states | DC missing weeks | US missing weeks | Missing location-weeks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [inpatient_flu](inpatient_flu.md) | Daily; Saturday trailing-7-day value | 2607/2703 (96.4%) | 5 | 45 | 0 | 0 | 96 |
| [inpatient_covid](inpatient_covid.md) | Daily; Saturday trailing-7-day value | 2607/2703 (96.4%) | 5 | 45 | 0 | 0 | 96 |
| [outpatient_flu](outpatient_flu.md) | Daily; Saturday trailing-7-day value | 2755/2756 (100.0%) | 1 | 1 | 0 | 0 | 1 |
| [outpatient_covid](outpatient_covid.md) | Daily; Saturday trailing-7-day value | 2755/2756 (100.0%) | 1 | 1 | 0 | 0 | 1 |
| [nwss_flu_wval_like](nwss_flu_wval_like.md) | Weekly derived index; irregular samples | 1082/2597 (41.7%) | 48 | 53 | — | 29 | 1515 |
| [nwss_covid_wval_like](nwss_covid_wval_like.md) | Weekly derived index; irregular samples | 1131/2703 (41.8%) | 49 | 53 | 53 | 29 | 1572 |
| [nwss_rsv_wval_like](nwss_rsv_wval_like.md) | Weekly derived index; irregular samples | 1070/2597 (41.2%) | 48 | 53 | — | 29 | 1527 |
| [nwss_flu_pct_rank](nwss_flu_pct_rank.md) | Weekly derived index; irregular samples | 1082/2597 (41.7%) | 48 | 53 | — | 29 | 1515 |
| [nwss_covid_pct_rank](nwss_covid_pct_rank.md) | Weekly derived index; irregular samples | 1131/2703 (41.8%) | 49 | 53 | 53 | 29 | 1572 |
| [nwss_rsv_pct_rank](nwss_rsv_pct_rank.md) | Weekly derived index; irregular samples | 1070/2597 (41.2%) | 48 | 53 | — | 29 | 1527 |
| [ilinet_ili](ilinet_ili.md) | Weekly | 2385/2756 (86.5%) | 21 | 35 | 0 | 0 | 371 |
| [clinical_lab_flu_pct_positive](clinical_lab_flu_pct_positive.md) | Weekly | 1898/2385 (79.6%) | 21 | 53 | — | 0 | 487 |
| [flusurv_flu_rate](flusurv_flu_rate.md) | Weekly | 351/742 (47.3%) | 13 | 53 | — | 26 | 391 |
| [kinsa_ili](kinsa_ili.md) | Daily; complete weekly mean | 18/53 (34.0%) | — | — | — | 35 | 35 |

## Lag 10: 10 observation weeks earlier

| Covariate | Frequency / representation | Reported / checked | Missing states (of 50) | Distinct weeks, states | DC missing weeks | US missing weeks | Missing location-weeks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [inpatient_flu](inpatient_flu.md) | Daily; Saturday trailing-7-day value | 2612/2703 (96.6%) | 5 | 45 | 0 | 0 | 91 |
| [inpatient_covid](inpatient_covid.md) | Daily; Saturday trailing-7-day value | 2612/2703 (96.6%) | 5 | 45 | 0 | 0 | 91 |
| [outpatient_flu](outpatient_flu.md) | Daily; Saturday trailing-7-day value | 2755/2756 (100.0%) | 1 | 1 | 0 | 0 | 1 |
| [outpatient_covid](outpatient_covid.md) | Daily; Saturday trailing-7-day value | 2755/2756 (100.0%) | 1 | 1 | 0 | 0 | 1 |
| [nwss_flu_wval_like](nwss_flu_wval_like.md) | Weekly derived index; irregular samples | 1082/2597 (41.7%) | 48 | 53 | — | 29 | 1515 |
| [nwss_covid_wval_like](nwss_covid_wval_like.md) | Weekly derived index; irregular samples | 1131/2703 (41.8%) | 49 | 53 | 53 | 29 | 1572 |
| [nwss_rsv_wval_like](nwss_rsv_wval_like.md) | Weekly derived index; irregular samples | 1072/2597 (41.3%) | 48 | 53 | — | 29 | 1525 |
| [nwss_flu_pct_rank](nwss_flu_pct_rank.md) | Weekly derived index; irregular samples | 1082/2597 (41.7%) | 48 | 53 | — | 29 | 1515 |
| [nwss_covid_pct_rank](nwss_covid_pct_rank.md) | Weekly derived index; irregular samples | 1131/2703 (41.8%) | 49 | 53 | 53 | 29 | 1572 |
| [nwss_rsv_pct_rank](nwss_rsv_pct_rank.md) | Weekly derived index; irregular samples | 1072/2597 (41.3%) | 48 | 53 | — | 29 | 1525 |
| [ilinet_ili](ilinet_ili.md) | Weekly | 2406/2756 (87.3%) | 21 | 34 | 0 | 0 | 350 |
| [clinical_lab_flu_pct_positive](clinical_lab_flu_pct_positive.md) | Weekly | 1914/2385 (80.3%) | 21 | 53 | — | 0 | 471 |
| [flusurv_flu_rate](flusurv_flu_rate.md) | Weekly | 351/742 (47.3%) | 13 | 53 | — | 26 | 391 |
| [kinsa_ili](kinsa_ili.md) | Daily; complete weekly mean | 18/53 (34.0%) | — | — | — | 35 | 35 |

## Lag 11: 11 observation weeks earlier

| Covariate | Frequency / representation | Reported / checked | Missing states (of 50) | Distinct weeks, states | DC missing weeks | US missing weeks | Missing location-weeks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [inpatient_flu](inpatient_flu.md) | Daily; Saturday trailing-7-day value | 2617/2703 (96.8%) | 5 | 44 | 0 | 0 | 86 |
| [inpatient_covid](inpatient_covid.md) | Daily; Saturday trailing-7-day value | 2617/2703 (96.8%) | 5 | 44 | 0 | 0 | 86 |
| [outpatient_flu](outpatient_flu.md) | Daily; Saturday trailing-7-day value | 2755/2756 (100.0%) | 1 | 1 | 0 | 0 | 1 |
| [outpatient_covid](outpatient_covid.md) | Daily; Saturday trailing-7-day value | 2755/2756 (100.0%) | 1 | 1 | 0 | 0 | 1 |
| [nwss_flu_wval_like](nwss_flu_wval_like.md) | Weekly derived index; irregular samples | 1082/2597 (41.7%) | 48 | 53 | — | 29 | 1515 |
| [nwss_covid_wval_like](nwss_covid_wval_like.md) | Weekly derived index; irregular samples | 1131/2703 (41.8%) | 49 | 53 | 53 | 29 | 1572 |
| [nwss_rsv_wval_like](nwss_rsv_wval_like.md) | Weekly derived index; irregular samples | 1073/2597 (41.3%) | 48 | 53 | — | 29 | 1524 |
| [nwss_flu_pct_rank](nwss_flu_pct_rank.md) | Weekly derived index; irregular samples | 1082/2597 (41.7%) | 48 | 53 | — | 29 | 1515 |
| [nwss_covid_pct_rank](nwss_covid_pct_rank.md) | Weekly derived index; irregular samples | 1131/2703 (41.8%) | 49 | 53 | 53 | 29 | 1572 |
| [nwss_rsv_pct_rank](nwss_rsv_pct_rank.md) | Weekly derived index; irregular samples | 1073/2597 (41.3%) | 48 | 53 | — | 29 | 1524 |
| [ilinet_ili](ilinet_ili.md) | Weekly | 2427/2756 (88.1%) | 21 | 33 | 0 | 0 | 329 |
| [clinical_lab_flu_pct_positive](clinical_lab_flu_pct_positive.md) | Weekly | 1930/2385 (80.9%) | 21 | 53 | — | 0 | 455 |
| [flusurv_flu_rate](flusurv_flu_rate.md) | Weekly | 351/742 (47.3%) | 13 | 53 | — | 26 | 391 |
| [kinsa_ili](kinsa_ili.md) | Daily; complete weekly mean | 18/53 (34.0%) | — | — | — | 35 | 35 |

