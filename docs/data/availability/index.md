# Covariate availability and revisions, 2025–26

**[New: distinguish unknown timing, later observed values and explicit missing reports](evidence.md)** — includes crosses and a state selector.


**[See the season timeline heatmaps](timeline.md)** — observation week across, submission date upward; hover to identify missing states in the interactive explorer.


This checks **actual archived values at the Hub deadline**, not the assumed-available finalized proxies used to fill gaps in the B2 operational panel. The B0 reproduction audited targets; the existing B2 covariate coverage table combined 12 history weeks and did not show this lag/revision breakdown.



<!-- raw-audit:start -->
## Full tables: all 14 covariates and 12 lags

[Availability, unique missing states and distinct missing weeks](full-availability-summary.csv) · [All-lag revisions](summary.csv) · [Per-state counts and exact missing dates](missing-by-state-lag.csv).

Every covariate link in the full table opens its state-by-lag missing-week counts. DC and national gaps are separate; unsupported geographies are listed explicitly.

## Correction: reported values versus retained values

**The low claims availability was partly a processing problem.** Multiple different finite values on the same report date were being converted to missing by the pipeline. The corrected source-availability counts below include those published values; they do not pick an arbitrary value for modeling. [Detailed diagnosis and evidence](conflict-audit.md).

“Location-weeks checked” = **native locations × 53 deadlines**: 52 × 53 = 2,756 for a source covering all states, DC and US; 1 × 53 = 53 for national Kinsa. It is not a count of revision pairs. The denominator also includes archive gaps and off-season weeks.

| Covariate | Frequency / model input | Location-weeks checked | Latest | One week earlier | Two earlier | Three earlier |
| --- | --- | --- | --- | --- | --- | --- |
| inpatient_flu | Daily → Saturday trailing-7-day value | 2703 | 2109 (78.0%) | 2329 (86.2%) | 2410 (89.2%) | 2477 (91.6%) |
| inpatient_covid | Daily → Saturday trailing-7-day value | 2703 | 2109 (78.0%) | 2329 (86.2%) | 2410 (89.2%) | 2477 (91.6%) |
| outpatient_flu | Daily → Saturday trailing-7-day value | 2756 | 2520 (91.4%) | 2597 (94.2%) | 2649 (96.1%) | 2701 (98.0%) |
| outpatient_covid | Daily → Saturday trailing-7-day value | 2756 | 2520 (91.4%) | 2597 (94.2%) | 2649 (96.1%) | 2701 (98.0%) |
| nwss_flu_wval_like | Weekly index; irregular raw samples | 2597 | 339 (13.1%) | 805 (31.0%) | 1012 (39.0%) | 1039 (40.0%) |
| nwss_covid_wval_like | Weekly index; irregular raw samples | 2703 | 326 (12.1%) | 774 (28.6%) | 1059 (39.2%) | 1093 (40.4%) |
| nwss_rsv_wval_like | Weekly index; irregular raw samples | 2597 | 443 (17.1%) | 948 (36.5%) | 1021 (39.3%) | 1037 (39.9%) |
| nwss_flu_pct_rank | Weekly index; irregular raw samples | 2597 | 339 (13.1%) | 805 (31.0%) | 1012 (39.0%) | 1039 (40.0%) |
| nwss_covid_pct_rank | Weekly index; irregular raw samples | 2703 | 326 (12.1%) | 774 (28.6%) | 1059 (39.2%) | 1093 (40.4%) |
| nwss_rsv_pct_rank | Weekly index; irregular raw samples | 2597 | 443 (17.1%) | 948 (36.5%) | 1021 (39.3%) | 1037 (39.9%) |
| ilinet_ili | Weekly | 2756 | 52 (1.9%) | 2001 (72.6%) | 2054 (74.5%) | 2107 (76.5%) |
| clinical_lab_flu_pct_positive | Weekly | 2385 | 38 (1.6%) | 1476 (61.9%) | 1607 (67.4%) | 1669 (70.0%) |
| flusurv_flu_rate | Weekly | 742 | 26 (3.5%) | 351 (47.3%) | 351 (47.3%) | 351 (47.3%) |
| kinsa_ili | Daily → complete 7-day mean | 53 | 14 (26.4%) | 15 (28.3%) | 16 (30.2%) | 17 (32.1%) |

Claims are daily but the model selects Saturday's trailing-seven-day percentage. ILI/labs/FluSurv are weekly. Kinsa daily values form a complete weekly mean. Wastewater indices are weekly aggregates of irregular measurements. “Latest” refers to that weekly input, not the newest arbitrary daily report.

**ILI's 72.6% is a different issue:** six zero-report weeks and reduced state coverage are present in the raw archive. Kinsa/wastewater also lack early-season vintage coverage. These season-wide percentages therefore do not measure reporting delay alone. The old revision figures below remain conditional on the unambiguous retained subset and need a conflict-resolution policy before they can describe all claims reports.
<!-- raw-audit:end -->

## What the counts mean

The season is defined by the latest observation week: **2025-08-02 through 2026-08-01**, with **53 weekly issuance dates** (2025-08-06 through 2026-08-05). This is the complete modeled season, not only weeks with a scored Hub submission. **Lag 0** is the Saturday four days before the nominal Wednesday; lag 1 is the preceding Saturday, and so on. Each count is a **location × issuance** opportunity. The old label “possible pairs” meant the number of location-weeks checked, **not** the number of finite reported/final pairs. For a source covering 50 states, DC and US this is 52 locations × 53 deadlines = **2,756**. For national Kinsa it is 1 × 53 = **53**. A revision pair exists only when both reported and final values are finite. Kinsa is counted once nationally, not broadcast and counted 52 times. Geography is the fixed native support observed anywhere in this season's final data or eligible archived histories; structurally unsupported states are excluded. Full location lists are downloadable.

Cutoffs are Wednesday **23:00 America/New_York**, except the study's documented joint-Hub holiday deadlines: the nominal December 24 and December 31, 2025 rounds use December 29 and January 4. The context Saturday stays fixed. Thus this is availability **when submission is due**, including extensions, rather than a literal-Wednesday-only audit. Date-only release labels are conservatively placed at the end of their UTC day, following the frozen panel policy. No observations, lags or availability have been imputed here.

## Source frequency and what “latest” means

Claims are daily trailing-seven-day percentages; the current model takes the Saturday value, rather than whichever daily observation was most recently published. Kinsa is daily and is converted to a complete Sunday–Saturday mean. ILINet, clinical labs and FluSurv are weekly. Wastewater measurements have irregular sampling dates and are aggregated into weekly indices. This audit checks those **model input representations**; it does not count a Friday daily value as an available Saturday value.

## Availability by age of observation

These are the frozen pipeline counts, **after conflict rejection**. See the [raw archive conflict audit](conflict-audit.md) for finite reported values discarded by that rule. Each cell is **available count (percent of opportunities)**. The denominator is the same for every lag in a row. Wastewater percentiles are included for completeness but were not used by the seven-source B2 grid, which used WVAL-like indices.

| Covariate | Native frequency / model representation | Locations | Location-weeks checked | Latest (0) | 1 week earlier | 2 earlier | 3 earlier | 4 earlier | 5 earlier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Inpatient claims — flu | Daily; Saturday trailing-7-day value | 51 | 2703 | 1819 (67.3%) | 1967 (72.8%) | 2030 (75.1%) | 2088 (77.2%) | 2137 (79.1%) | 2197 (81.3%) |
| Inpatient claims — COVID | Daily; Saturday trailing-7-day value | 51 | 2703 | 1675 (62.0%) | 1791 (66.3%) | 1850 (68.4%) | 1922 (71.1%) | 1971 (72.9%) | 2046 (75.7%) |
| Outpatient claims — flu | Daily; Saturday trailing-7-day value | 52 | 2756 | 1860 (67.5%) | 1930 (70.0%) | 2002 (72.6%) | 2061 (74.8%) | 2107 (76.5%) | 2125 (77.1%) |
| Outpatient claims — COVID | Daily; Saturday trailing-7-day value | 52 | 2756 | 1822 (66.1%) | 1894 (68.7%) | 1967 (71.4%) | 2024 (73.4%) | 2066 (75.0%) | 2078 (75.4%) |
| ILINet ILI | Weekly | 52 | 2756 | 52 (1.9%) | 2001 (72.6%) | 2054 (74.5%) | 2107 (76.5%) | 2160 (78.4%) | 2213 (80.3%) |
| Clinical lab flu positivity | Weekly | 45 | 2385 | 38 (1.6%) | 1476 (61.9%) | 1607 (67.4%) | 1669 (70.0%) | 1714 (71.9%) | 1759 (73.8%) |
| FluSurv flu | Weekly | 14 | 742 | 26 (3.5%) | 351 (47.3%) | 351 (47.3%) | 351 (47.3%) | 351 (47.3%) | 351 (47.3%) |
| Kinsa national | Daily; complete 7-day mean | 1 | 53 | 14 (26.4%) | 15 (28.3%) | 16 (30.2%) | 17 (32.1%) | 18 (34.0%) | 18 (34.0%) |
| Wastewater FLU — WVAL-like | Weekly index from irregular samples | 49 | 2597 | 339 (13.1%) | 805 (31.0%) | 1012 (39.0%) | 1039 (40.0%) | 1056 (40.7%) | 1070 (41.2%) |
| Wastewater FLU — percentile | Weekly index from irregular samples | 49 | 2597 | 339 (13.1%) | 805 (31.0%) | 1012 (39.0%) | 1039 (40.0%) | 1056 (40.7%) | 1070 (41.2%) |
| Wastewater COVID — WVAL-like | Weekly index from irregular samples | 51 | 2703 | 326 (12.1%) | 774 (28.6%) | 1059 (39.2%) | 1093 (40.4%) | 1107 (41.0%) | 1120 (41.4%) |
| Wastewater COVID — percentile | Weekly index from irregular samples | 51 | 2703 | 326 (12.1%) | 774 (28.6%) | 1059 (39.2%) | 1093 (40.4%) | 1107 (41.0%) | 1120 (41.4%) |
| Wastewater RSV — WVAL-like | Weekly index from irregular samples | 49 | 2597 | 443 (17.1%) | 948 (36.5%) | 1021 (39.3%) | 1037 (39.9%) | 1046 (40.3%) | 1054 (40.6%) |
| Wastewater RSV — percentile | Weekly index from irregular samples | 49 | 2597 | 443 (17.1%) | 948 (36.5%) | 1021 (39.3%) | 1037 (39.9%) | 1046 (40.3%) | 1054 (40.6%) |

All **12 context lags**, weeks with any/all native locations available, and every location's counts are in the [summary](summary.csv) and [location table](by-location.csv).

## How far the deadline value is from final

**These are conditional on the old pipeline retaining an unambiguous deadline value and final value.** Same-report-date conflicts in claims were discarded; the [conflict audit](conflict-audit.md) explains why these revision estimates are incomplete. They must not be interpreted as revision estimates for all reported claims values.

For the **same observation week and location**, revision = frozen final value minus deadline value. Only finite pairs enter. The table reports **100 × sum(abs(final − reported)) / sum(abs(final))**, with the paired sample count. This is absolute revision relative to aggregate final magnitude, not an average of cellwise percentages; it remains meaningful when some final values are zero. Missing deadline values do not count as zero revision. Samples differ by lag, so this is not a fixed cohort estimate of revision decay.

| Covariate | Latest (0) | 1 week earlier | 2 earlier | 3 earlier | 4 earlier | 5 earlier |
| --- | --- | --- | --- | --- | --- | --- |
| Inpatient claims — flu | 59.1% (n=1610) | 39.4% (n=1771) | 29.8% (n=1829) | 23.5% (n=1881) | 18.1% (n=1934) | 14.7% (n=1990) |
| Inpatient claims — COVID | 67.1% (n=1407) | 43.1% (n=1552) | 31.6% (n=1623) | 24.6% (n=1690) | 20.0% (n=1740) | 17.3% (n=1807) |
| Outpatient claims — flu | 26.7% (n=768) | 17.3% (n=804) | 13.8% (n=854) | 12.2% (n=884) | 11.0% (n=926) | 8.8% (n=972) |
| Outpatient claims — COVID | 38.6% (n=704) | 29.4% (n=744) | 25.4% (n=800) | 24.4% (n=822) | 22.6% (n=864) | 19.2% (n=909) |
| ILINet ILI | 6.7% (n=52) | 2.5% (n=2001) | 1.2% (n=2054) | 0.9% (n=2107) | 0.8% (n=2160) | 0.7% (n=2213) |
| Clinical lab flu positivity | 13.8% (n=38) | 8.2% (n=1476) | 2.7% (n=1607) | 1.7% (n=1669) | 1.0% (n=1714) | 0.8% (n=1759) |
| FluSurv flu | 30.4% (n=26) | 29.2% (n=351) | 10.1% (n=351) | 6.6% (n=351) | 5.4% (n=351) | 4.7% (n=351) |
| Kinsa national | 0.0% (n=14) | 0.0% (n=15) | 0.0% (n=16) | 0.0% (n=17) | 0.0% (n=18) | 0.0% (n=18) |
| Wastewater FLU — WVAL-like | 31.6% (n=339) | 14.0% (n=805) | 11.6% (n=1012) | 10.7% (n=1039) | 10.6% (n=1056) | 10.4% (n=1070) |
| Wastewater FLU — percentile | 17.7% (n=339) | 8.8% (n=805) | 6.5% (n=1012) | 5.6% (n=1039) | 5.2% (n=1056) | 4.8% (n=1070) |
| Wastewater COVID — WVAL-like | 38.7% (n=326) | 26.1% (n=774) | 23.0% (n=1059) | 21.4% (n=1093) | 20.7% (n=1107) | 20.5% (n=1120) |
| Wastewater COVID — percentile | 38.6% (n=326) | 26.4% (n=774) | 21.1% (n=1059) | 19.1% (n=1093) | 17.7% (n=1107) | 16.8% (n=1120) |
| Wastewater RSV — WVAL-like | 50.7% (n=443) | 15.8% (n=948) | 12.2% (n=1021) | 10.4% (n=1037) | 10.2% (n=1046) | 10.0% (n=1054) |
| Wastewater RSV — percentile | 19.9% (n=443) | 6.1% (n=948) | 4.4% (n=1021) | 4.0% (n=1037) | 3.9% (n=1046) | 3.8% (n=1054) |

For an original-unit view, the following uses **the earliest lag with any paired observations for each source**, not a common lag. Positive bias means the final value was higher. “Changed” ignores float noise with rtol=1e-5 and atol=1e-6.

| Covariate | Lag | Pairs | Changed | Mean revision | Mean absolute revision | 90th percentile absolute | Units |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Inpatient claims — flu | 0 | 1610 | 39.9% | -0.007393 | 0.06911 | 0.1817 | percentage points |
| Inpatient claims — COVID | 0 | 1407 | 68.1% | +0.005785 | 0.1102 | 0.28 | percentage points |
| Outpatient claims — flu | 0 | 768 | 70.8% | -0.00382 | 0.03979 | 0.1146 | percentage points |
| Outpatient claims — COVID | 0 | 704 | 88.6% | +0.03936 | 0.05805 | 0.1375 | percentage points |
| ILINet ILI | 0 | 52 | 65.4% | +0.0282 | 0.09546 | 0.18 | percentage points |
| Clinical lab flu positivity | 0 | 38 | 36.8% | +0.02474 | 0.05526 | 0.163 | percentage points |
| FluSurv flu | 0 | 26 | 23.1% | +0.02692 | 0.02692 | 0.1 | per 100,000 catchment residents |
| Kinsa national | 0 | 14 | 0.0% | +0 | 0 | 0 | percentage points |
| Wastewater FLU — WVAL-like | 0 | 339 | 35.1% | -0.1879 | 0.4587 | 1.583 | index units |
| Wastewater FLU — percentile | 0 | 339 | 100.0% | +0.02138 | 0.07573 | 0.1655 | index units |
| Wastewater COVID — WVAL-like | 0 | 326 | 92.0% | +0.5927 | 0.8771 | 2.338 | index units |
| Wastewater COVID — percentile | 0 | 326 | 99.7% | +0.08646 | 0.1109 | 0.2355 | index units |
| Wastewater RSV — WVAL-like | 0 | 443 | 60.0% | -0.8407 | 1.386 | 4.233 | index units |
| Wastewater RSV — percentile | 0 | 443 | 99.8% | -0.03059 | 0.1025 | 0.2434 | index units |

## Limitations and provenance

**Absent from our archive does not prove absent upstream.** Kinsa's archived releases start in April 2026; wastewater vintages also start in 2026, and FluSurv has archive gaps. Whole-season percentages therefore combine reporting timeliness and archive coverage. The [first archived releases](archive-starts.csv) identify the boundary for each source; they are not inferred historical launch dates. Revision estimates cover only observed pairs, not the missing months.

**Claims have an additional comparison gap:** some finite deadline values have no finite counterpart in the frozen final training panel. The table below counts these at lag 0. They contribute to availability but are excluded from revision calculations. This can make the revision subset unrepresentative; these figures do not establish the cause of the final-panel gaps or prove that upstream final values were unavailable.

| Covariate | Reported at lag 0 | Finite final pairs | Reported but final missing |
| --- | --- | --- | --- |
| inpatient_flu | 1819 | 1610 | 209 |
| inpatient_covid | 1675 | 1407 | 268 |
| outpatient_flu | 1860 | 768 | 1092 |
| outpatient_covid | 1822 | 704 | 1118 |

The [weekly counts](by-week.csv) separate temporal gaps from geographic coverage. All-season counts should not be interpreted as a source's operating-period reliability.

“Final” means the frozen finalized arrays used by this experiment, not an assertion that the source can never revise again. The deadline and final values are aligned in the panel's native units. Wastewater WVAL-like/percentile changes may include changes to index construction or historical baseline as well as revisions to underlying measurements; they should not be interpreted as raw-concentration revisions. Kinsa values are complete seven-day averages of its national daily signal. Claims are the source's trailing-seven-day percentage sampled at Saturday.

Source: `data/processed/panel-b2-deadline.npz`, SHA256 **27c36c76c96cbd1a6705fe9dd9d848a4e734861d61fd300495868bbe78fd6d9a**. Original [deadline policy and pinned source snapshots](provenance/deadline-policy.json). The operational proxy-filled panel is deliberately not used.

[All paired and missing cells](cells.csv.gz) · [Native geographic support](native-support.csv) · [Full summary](summary.csv) · Main experiment report (retired).

Reproduce from the repository root:

```bash
# Retired audit command: covariate_availability.py
```

[First observation dates and first archived reports, before model filters](../index.md#first-observation-and-first-archived-report-dates).
