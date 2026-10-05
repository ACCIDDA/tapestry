# Data

## Human-made assumptions on availability

| Data source | Availability |
| --- | --- |
| Kinsa ILI | T-0 |
| FluSurv-NET | T-1 |
| All six targets (NHSN / NSSP) | T-0 |
| Clinical lab flu percent positive | T-1 |
| ILINet ILI | T-1 |
| Inpatient flu / COVID | T-0 |
| Outpatient flu / COVID | T-0 |
| NWSS | T-0 |

T is the latest completed Saturday in the forecast context. These are human
assumptions for the [B-2 training protocol](../experiments/b-2-t0/index.md#protocol), not measured release dates.
The same schedule governs the [standard evaluation](../workflow.md#standard-evaluation).
ED (NSSP) is assumed T-0 in **every** season because CDC publishes it in time for
the Wednesday deadline since 2025–26 and will next season. In 2024–25 the archived
NSSP data first appeared on the Friday after each week (one week late), so 2024–25
newest-week ED inputs are mostly finalized values, starred in reports.

## Deadlines

Decided 5 October 2026: **each Hub's own deadline**, never a common earliest one. The
panel reads archived reports at Wednesday 23:00 America/New_York (before: 23:59 UTC,
i.e. 18:59/19:59 Eastern). A release labelled only by a date counts at the end of that
UTC day; Git commits keep their exact time. Holiday extensions, from each Hub's
`hub-config/tasks.json` / README history (`dataset/build.py` `HOLIDAY_DEADLINES`):

| Hub reference date (Saturday) | Nominal Wednesday | FluSight | COVID Hub | RSV Hub |
|---|---|---|---|---|
| 2024-12-28 | 2024-12-25 | Thu 2024-12-26 | Thu 2024-12-26 | Wed 2024-12-25 (no RSV Hub before 2025-08) |
| 2025-01-04 | 2025-01-01 | Thu 2025-01-02 | Thu 2025-01-02 | Wed 2025-01-01 (no RSV Hub) |
| 2025-12-27 | 2025-12-24 | Tue 2025-12-30 | Mon 2025-12-29 | Mon 2025-12-29 |
| 2026-01-03 | 2025-12-31 | Mon 2026-01-05 | Sun 2026-01-04 | Sun 2026-01-04 |

The Wednesday and its context Saturday stay fixed. The panel's main as-of arrays use
the FluSight deadline (flu forecasts and all training episodes); the rows where the
COVID or RSV deadline differs are stored separately and used for those pathogens'
forecasts. No extension was found for the 2025-01-22 round (newest week 2025-01-18),
whose admissions were not archived by the deadline: those inputs are finalized values
and starred.

Start here for **what is in the dataset, when it was available, and how much it changed**. Coverage of finalized history and evidence of historical availability are different questions; the pages below keep them separate.

## Explore the data

| Question | Where to go |
| --- | --- |
| What could we see at each forecast deadline? | **[Interactive availability staircase](staircase.md)** — choose a target/covariate and state, then hover for evidence |
| What are the sources, units and geographic definitions? | [Source catalog](sources.md) and the source-specific pages below |

## Current dataset

The default dataset is `data/processed/panel.npz`: **May 14, 2022–September 19, 2026**, 228 observation weeks, 52 locations (50 states, DC and native US), six targets and 14 covariate series. It includes the 12 weeks preceding the August 6, 2022 season start. **2022–23 is available for training**; B-2 evaluates 2025–26 and 2024–25; the general scenario default also supports 2023–24.

The added season has finalized NHSN flu/COVID admissions for all 52 weeks and locations. The three ED outcomes begin October 1, 2022; RSV admissions are absent. Covariates retain their native gaps. Finite CDC finalized NHSN counts take precedence in retrospective targets; dated inputs do not receive that override.

The availability study remains a dated audit. B-2 uses the assumptions above
and the current finalized panel, retaining native gaps; it does not use the audit's
historical release masks.

<!-- source-history:start -->
## First observation and first archived report dates

These dates scan all downloaded native geographies for each modeled signal, with **no state selection, Saturday-only selection, season/deadline cutoff, fill-method filter, age-group filter, or conflicting-value rejection**. Earliest finite values are reported below; the CSV also records the first raw report including null rows. Acquisition scope still limits what exists in our downloaded archives.

**Observation date** is the day/week the measurement describes. **Report date** is the earliest release timestamp carried by our archive for a finite value. Neither is our download date. An archive beginning later than its observations contains retrospective history; its first report date is not proof the source was unavailable before that date.

| Covariate | Earliest observation date | Earliest archived report date |
| --- | --- | --- |
| inpatient_flu | 2019-10-08 | 2020-05-29 |
| inpatient_covid | 2019-10-08 | 2020-05-29 |
| outpatient_flu | 2019-10-08 | 2020-05-29 |
| outpatient_covid | 2019-10-08 | 2020-05-29 |
| ilinet_ili | 1997-10-04 | 2003-11-28 |
| clinical_lab_flu_pct_positive | 2016-10-08 | 2018-10-05 |
| flusurv_flu_rate | 2003-10-04 | 2012-11-02 |
| kinsa_ili | 2019-01-01 | 2026-04-06 |
| NWSS raw covid_avg_conc_lin | 2020-01-14 | 2026-02-25 |
| NWSS raw rsv_avg_conc_lin | 2022-02-27 | 2026-02-25 |
| NWSS raw flu_avg_conc_lin | 2021-09-15 | 2026-02-25 |

Kinsa here is the raw daily PopHIVE/Kinsa series, before requiring complete seven-day weeks. NWSS rows are raw sewershed concentrations, before deriving weekly state indices or imposing site-count/history requirements. The modeled wastewater indices therefore have different earliest observation weeks; see the raw-date CSV for their separate entries.

A long archive history does not imply continuous coverage. In particular FluSurv has a major gap from November 2020 to November 2025, then to February 2026. First-date summaries cannot establish availability in each intervening season. ILINet's earliest archive entry can also be national rather than state-level.

[All signal dates and raw row counts](availability/raw-archive-starts.csv) · [Raw NWSS dates](availability/raw-nwss-starts.csv) · [Availability evidence](staircase.md).
<!-- source-history:end -->

## Sources and reporting frequency

| Source | Native frequency and model representation | Details |
| --- | --- | --- |
| NHSN admissions and NSSP ED | Weekly target histories; six pathogen/outcome combinations | [Sources](sources.md) |
| Inpatient/outpatient claims | Daily trailing-seven-day percentages, sampled on Saturday | [Claims](sources.md#claims), [same-date conflict audit](methods.md#same-date-conflicts) |
| Kinsa ILI | Daily national signal, averaged over complete seven-day weeks | [Kinsa](sources.md#kinsa) |
| ILINet ILI, clinical lab positivity, FluSurv | Weekly; geographic coverage differs by source | [FluView and FluSurv](sources.md#fluview-and-flusurv) |
| Wastewater | Irregular raw sampling, converted to weekly WVAL-like and percentile indices | [Wastewater](sources.md#wastewater) |


**Claims caveat:** conflicting finite values sharing a report date are rejected by the model pipeline. Availability evidence counts those reports, but revision estimates remain conditional on the unambiguous subset. [Evidence and implications](methods.md#same-date-conflicts).

## Acquisition, definitions and provenance

- [Source catalog](sources.md) and [selection and provenance](methods.md).
- [Vintages and geography](methods.md#vintages-and-geography).
- [Storage and acquisition](methods.md#storage-and-provenance).
- [Deadline audit policy](availability/provenance/deadline-policy.json) and [operational availability assumptions](availability/provenance/operational-policy.json).
- [Panel diagnostics](panel.md), generated from the saved panel with its snapshot date and hash.
- [General data explorer](../explorer/overview.md), [filters](../explorer/filters.md) and [local API](../explorer/api.md). This is separate from the availability staircase explorer above.

Model comparisons are collected under [Experiments](../experiments/index.md). Data definitions, coverage, availability and revisions are collected here.
