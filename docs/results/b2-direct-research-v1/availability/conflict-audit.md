# Why the previous availability table was low

**A processing rule was counting conflicting reports as unavailable.** The table was numerically faithful to the model panel, but it understated source availability for claims. We should have labeled it “values retained by the pipeline,” not simply “reported values available.”

## Corrected source-availability interpretation

At each deadline, find the latest archived statement date for that observation week and location. Count it as reported if **at least one finite native value exists on that date**, including dates with multiple conflicting finite values. This does not choose one conflicting number to feed a model. Explicit all-null latest reports stay unavailable. For the seven Delphi covariates, this audit rereads raw archives before the conflict rule; wastewater/Kinsa retain their previously reconstructed archive counts.

Counts are location-weeks, not finite revision pairs. Each row's denominator is **53 deadlines × fixed native locations**. For example 52 locations × 53 = 2,756; Kinsa is 53 national weeks. Latest means the preceding Saturday, not the newest arbitrary daily observation. Cutoffs retain holiday extensions. Archive gaps remain in the denominator.

| Covariate | Frequency / model input | Location-weeks | Latest | One week earlier | Two earlier | Three earlier |
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

## How much the conflict rule removed

The following is **one week earlier**. “Finite raw report” is not a claim that the exact latest value can be uniquely recovered from the date-only archive.

| Covariate | Location-weeks | Retained in pipeline | Finite raw report | Discarded conflicts |
| --- | --- | --- | --- | --- |
| clinical_lab_flu_pct_positive | 2385 | 1476 (61.9%) | 1476 (61.9%) | 0 |
| flusurv_flu_rate | 742 | 351 (47.3%) | 351 (47.3%) | 0 |
| ilinet_ili | 2756 | 2001 (72.6%) | 2001 (72.6%) | 0 |
| inpatient_covid | 2703 | 1791 (66.3%) | 2329 (86.2%) | 538 |
| inpatient_flu | 2703 | 1967 (72.8%) | 2329 (86.2%) | 362 |
| outpatient_covid | 2756 | 1894 (68.7%) | 2597 (94.2%) | 703 |
| outpatient_flu | 2756 | 1930 (70.0%) | 2597 (94.2%) | 667 |

The affected extraction function is `tapestry.dataset.extract._without_conflicts`: different values sharing (source tier, report time, observation day, location) become NaN. It runs before both historical resolution and final-panel resolution. In the audited deadline cells, the counted conflicts contain only finite values (no mixed finite/null statements). Thus this is value ambiguity, not evidence that no observation was published.

Concrete raw example: **North Carolina outpatient flu**, observation day **2026-01-10**, report date **2026-01-20**, has values **0.832603 and 0.834298**. Both are available before the January 21 deadline, but our pipeline discards the cell. The raw file preserves only a date for these reports; file ordering has not been verified as revision ordering. Choosing the last CSV row would therefore introduce an unsupported assumption.

## ILI's remaining low percentage is visible in the archive

For lag 1, the 53 weekly rounds break down as follows: **9 rounds × 52 locations; 6 × 0; 9 × 30; 1 × 31; 28 × 44 = 2,001**, or 72.6% of 2,756. The six zero rounds are October 8–November 12, 2025. The 28 rounds from January 28 onward have 44/52, or 84.6%, native locations.

In a direct raw-file check for the observation week January 24, 2026, AK, CT, HI, NY, OK, OR, SD and UT lack eligible statewide values in the panel. The raw state ILI archive contains no AK observation for that week. A fresh read-only Delphi archive request for report times after January 29 likewise returned no AK rows. This supports an archive/source-coverage gap, not an arithmetic denominator error. It does not establish whether another upstream CDC resource could supply those values. No attribution to a shutdown or genuine nonreporting is asserted from these checks alone.

Kinsa and wastewater additionally lack early-season vintage archives, and FluSurv has a substantial archival gap. The season-wide table is not a pure reporting-delay estimate. It also includes off-season dates and fixed native geographies rather than conditioning on only dates/locations that happened to report.

## Consequences for revision estimates and experiments

The earlier revision table uses the **unambiguous subset retained by the old pipeline**. Because the same conflict rule also removes final values, claims revision sample sizes are much smaller than raw availability. Those conditional numbers remain reproducible but do not measure revision magnitude over all published values. Recovering a defensible within-date ordering, or an explicitly defined aggregation with a sensitivity analysis, is necessary before replacing them with all-report revision estimates.

This audit **does not change the frozen training panels, resolve ambiguous values arbitrarily, or rerun models**. The old B2 scores are still scores of the saved fits; claims-source attribution and the comparison to B0 now have an additional identified data-processing limitation. Its performance effect has not been measured.

[All 12-lag conflict counts](conflict-audit.csv) · [Original panel availability and conditional revisions](index.md) · [Weekly original counts](by-week.csv). Per-source compressed statement tables alongside this report contain deadline/final values, raw min/max/count, and the conflict flags.

Reproduce with:

```bash
.venv/bin/python analysis/b2-research/covariate_availability.py
.venv/bin/python analysis/b2-research/audit_covariate_conflicts.py
.venv/bin/python analysis/b2-research/write_availability_audit.py
```
