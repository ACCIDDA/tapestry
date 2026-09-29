# Where the raw covariate histories begin

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

[All signal dates and raw row counts](raw-archive-starts.csv) · [Raw NWSS dates](raw-nwss-starts.csv) · [Availability evidence](evidence.md).
