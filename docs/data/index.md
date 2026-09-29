# Data

Start here for **what is in the dataset, when it was available, and how much it changed**. Coverage of finalized history and evidence of historical availability are different questions; the pages below keep them separate.

## Explore the data

| Question | Where to go |
| --- | --- |
| What could we see at each forecast deadline? | **[Interactive availability staircase](availability/timeline/evidence.html)** — choose a target/covariate and state, then hover for evidence |
| Show the graphs together | [All 20 availability heatmaps](availability/timeline.md), with downloadable figures |
| What are the sources, units and geographic definitions? | [Source catalog](sources.md) and the source-specific pages below |

## Current dataset

The default dataset is `data/processed/panel.npz`: **May 14, 2022–September 19, 2026**, 228 observation weeks, 52 locations (50 states, DC and native US), six targets and 14 covariate series. It includes the 12 weeks preceding the August 6, 2022 season start. **2022–23 is available for training**; the scored evaluation seasons remain 2023–24, 2024–25 and 2025–26.

The added season has finalized NHSN flu/COVID admissions for all 52 weeks and locations. The three ED outcomes begin October 1, 2022; RSV admissions are absent. Covariates retain their native gaps. Finite CDC finalized NHSN counts take precedence in retrospective targets; dated inputs do not receive that override.

**The availability study and its plots still describe the frozen 2025–26 audit underlying the completed B2 experiment.** They have not been silently recomputed from the expanded base dataset. That experiment's deadline and operational panels remain separate, and must be rebuilt before a new run uses the added season.

## Read the staircase

The horizontal axis is the **Saturday observation week**. The vertical axis is the **nominal Wednesday forecast round**, with later rounds higher. **T−0 lies on the diagonal boundary:** the Saturday four days before Wednesday. One cell left is T−1; two left is T−2. Follow a column upward to see the evidence for that same observation change over time.

For Wednesday January 14, 2026, T−0 is January 10 and T−1 is January 3. Holiday extensions change the actual cutoff, but retain the nominal Wednesday row and its original Saturday. Hover shows the actual cutoff.

| Appearance | What it establishes |
| --- | --- |
| Green | A finite value was reported by the deadline |
| Gray | Arrival timing is unknown from the archive; it may need to be inferred from documented periods |
| Blue | An explicit missing report was followed by a later finite report |
| Red × | Explicitly missing, with no later finite value observed |
| Beige × | No finite value anywhere in the inspected history for that location/week |
| Yellow | Locations have different evidence; hover or select a state |
| White | A future observation week |

A cross does **not** prove permanent absence. A later archive entry does **not** by itself prove late publication. A dashboard can display historical values now without preserving evidence of when they first appeared. [Classification details and state explorer](availability/evidence.md).

[![Example: flu admissions availability evidence](availability/timeline/nhsn_flu_admissions-evidence.png)](availability/timeline.md)

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

[All signal dates and raw row counts](availability/raw-archive-starts.csv) · [Raw NWSS dates](availability/raw-nwss-starts.csv) · [Availability evidence](availability/evidence.md).
<!-- source-history:end -->

## Sources and reporting frequency

| Source | Native frequency and model representation | Details |
| --- | --- | --- |
| NHSN admissions and NSSP ED | Weekly target histories; six pathogen/outcome combinations | [Sources](sources.md) |
| Inpatient/outpatient claims | Daily trailing-seven-day percentages, sampled on Saturday | [Claims](covariates.md), [same-date conflict audit](availability/conflict-audit.md) |
| Kinsa ILI | Daily national signal, averaged over complete seven-day weeks | [Kinsa](kinsa.md) |
| ILINet ILI, clinical lab positivity, FluSurv | Weekly; geographic coverage differs by source | [FluView and FluSurv](fluview-flusurv.md) |
| Wastewater | Irregular raw sampling, converted to weekly WVAL-like and percentile indices | [Wastewater](wastewater.md) |


**Claims caveat:** conflicting finite values sharing a report date were rejected by the model pipeline. The corrected availability tables count those reports, but revision estimates remain conditional on the unambiguous subset. Dataset expansion did not resolve those conflicts. [Evidence and implications](availability/conflict-audit.md).

## Acquisition, definitions and provenance

- [Source catalog](sources.md) and [shared selection rules](selection.md).
- [Vintages and geography](vintages-and-geography.md).
- [Storage and provenance](storage.md) and [acquisition implementation](acquisition.md).
- [Deadline audit policy](availability/provenance/deadline-policy.json) and [operational availability assumptions](availability/provenance/operational-policy.json).
- [Earlier panel analysis](panel.md), explicitly dated to its preceding build.
- [General data explorer](../explorer/overview.md), [filters](../explorer/filters.md) and [local API](../explorer/api.md). This is separate from the availability staircase explorer above.

Model comparisons remain under **Results**. Data definitions, coverage, availability and revisions are collected here.
