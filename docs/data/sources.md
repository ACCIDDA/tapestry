# Source catalog

The checked-in catalog defines 25 independently retrievable datasets. The table
below summarizes acquisition and modeling metadata; run
`python scripts/pull_covariates.py catalog` for the complete machine-readable
specifications, signals, natural keys, and source URLs.

| Dataset key | Provider | Fetcher | Cadence | Versioned | Native geography |
|---|---|---|---|---:|---|
| `cdc_nhsn_final` | CDC | `socrata` | `weekly` | no | state, territory, hhs, nation |
| `cdc_nhsn_initial_release` | CDC | `socrata` | `weekly` | yes | state, territory, hhs, nation |
| `cdc_nhsn_preliminary` | CDC | `socrata` | `weekly` | no | state, territory, hhs, nation |
| `cdc_nrevss_comprehensive` | CDC | `socrata` | `weekly` | no | state, hhs, nation |
| `cdc_nrevss_covid_vintages` | CDC | `socrata` | `weekly` | yes | hhs, nation |
| `cdc_nrevss_national` | CDC | `socrata` | `weekly` | no | nation |
| `cdc_nrevss_rsv_vintages` | CDC | `socrata` | `weekly` | yes | hhs, nation |
| `cdc_nssp_daily` | CDC | `socrata` | `daily` | no | state, nation |
| `cdc_nssp_demographics` | CDC | `socrata` | `weekly` | no | nation |
| `cdc_nssp_trajectories` | CDC | `socrata` | `weekly` | no | state, hsa |
| `cdc_nwss_covid_raw` | CDC | `socrata` | `sample` | no | sewershed, county, state |
| `cdc_nwss_influenza_raw` | CDC | `socrata` | `sample` | no | sewershed, county, state |
| `cdc_nwss_rsv_raw` | CDC | `socrata` | `sample` | no | sewershed, county, state |
| `cdc_nwss_wval` | CDC | `socrata` | `weekly` | no | sewershed, state |
| `delphi_claims_inpatient` | CMU Delphi | `delphi_v5` | `daily` | yes | county, hrr, msa, state, hhs, census region/division, nation |
| `delphi_claims_outpatient` | CMU Delphi | `delphi_v5` | `daily` | yes | county, hrr, msa, state, hhs, census region/division, nation |
| `delphi_nhsn` | CMU Delphi | `delphi_v5` | `weekly` | yes | state, nation, hhs, census region/division |
| `delphi_nssp` | CMU Delphi | `delphi_v5` | `weekly` | yes | state, nation, hhs, county, hsa, hrr, msa, census region/division |
| `delphi_nwss` | CMU Delphi | `delphi_v5` | `sample` | yes | sewershed |
| `hub_covid_current` | Hubverse community | `hubverse` | `weekly` | yes | state, nation |
| `hub_covid_legacy` | Hubverse community | `hubverse` | `weekly` | yes | county, state, nation |
| `hub_flusight_current` | Hubverse community | `hubverse` | `weekly` | yes | state, nation |
| `hub_flusight_legacy` | Hubverse community | `hubverse` | `weekly` | yes | state, nation |
| `hub_rsv_current` | Hubverse community | `hubverse` | `weekly` | yes | state, nation |
| `hub_rsvnet` | Hubverse community | `hubverse` | `weekly` | yes | catchment, nation |

The `Versioned` column includes both preserved initial releases and archives
with multiple release dates. A `yes` does not necessarily mean every revision
is available; see the NHSN distinction below.

For the consumer-facing organization and retained measures, see
[Shared selection](selection.md): 25 acquisitions map to 15 logical groups.

## Source families

### CDC Socrata

Downloads use a source pre-count, deterministic ordering, complete offset
pagination, and a post-download count check. Payloads are streamed rather than
held in memory.

#### NHSN initial releases and revision history

`cdc_nhsn_initial_release` is the local catalog key for CDC's
[Weekly Hospital Respiratory Data (HRD) Metrics by Jurisdiction (Historical)](https://data.cdc.gov/Public-Health-Surveillance/Weekly-Hospital-Respiratory-Data-HRD-Metrics-by-Ju/rhwp-grxi)
(Socrata ID `rhwp-grxi`). It contains weekly hospital admissions,
hospitalizations, bed capacity and occupancy, and reporting coverage for
COVID-19, influenza, and RSV, beginning November 2024. CDC states that values
are not updated or adjusted after initial publication for a reporting week.

CDC therefore preserves first-release NHSN values in this product, but it is
not a complete archive of every subsequent revision.

| Catalog key | Release semantics |
|---|---|
| `cdc_nhsn_initial_release` | Frozen first published values for each reporting week; `revision_mode=initial_release`. |
| `cdc_nhsn_final` | Current finalized weekly values, which can incorporate later corrections. |
| `cdc_nhsn_preliminary` | Preliminary weekly release, available before the finalized release; not a full revision archive. |
| `delphi_nhsn` | Report-time vintages for available NHSN signals; `reference_time` identifies the observation week and `report_time` identifies the vintage. |

For example, if a week's admissions are first published as 100, then revised
to 110 and later 115, the initial-release dataset retains 100. Comparing it
with the current finalized dataset can reveal the change from 100 to 115,
but does not recover the intermediate value of 110 or its publication date.

Use initial releases to study first-publication error against current values.
For backtests requiring what was known at a particular forecast date, use the
available `delphi_nhsn` report-time archive or historical snapshots captured
at the relevant cutoff. Check vintage coverage first: an early observation
date does not imply that all releases since that date are archived. The
catalog's `versioned=True` for `cdc_nhsn_initial_release` means first-release
preservation, not a complete sequence of vintages.

### Delphi Epidata

Only the five Delphi V5 sources listed above are enabled. Direct CDC and Hubverse
sources are separate.

Following Delphi's [Python migration guide](https://cmu-delphi.github.io/epidatpy/migration_guide.html),
the downloader uses `EpiDataContext.epidata_meta()` to discover and validate live
signals and geographic resolutions, then `epidata_archive()` or
`epidata_snapshot()` to construct each query. Its public
[`request_arguments()` method](https://cmu-delphi.github.io/epidatpy/epidatpy.html#epidatpy.request.EpiDataCall.request_arguments)
provides the canonical HTTPS endpoint and wire parameters. These requests select
all `reference_time` dates and geographies within each signal/geography partition;
`--report-time` and `--snapshot-date` select the revision view.

The raw repository downloads the resulting CSV with `format=csv` instead of
converting through `.df()`, so large archives can be streamed without loading
them into memory. This preserves
all publisher columns, timestamp precision, leading-zero identifiers, zeros,
and missing values. Queries keep every published `fill_method` by default, as
the official client does; no local imputation is performed. Keep `fill_method`
in observation keys when selecting or aggregating series. An explicit
`--fill-method` can restrict the query. Authentication uses the V5 `token` header.

Large pulls use bounded concurrency, complete gzip validation, whole-partition
retries, and explicit resume directories. A saved `request.json` prevents reuse
under different source, signal, geography, fill, or version selectors. A resumed
unbounded archive or latest snapshot can include new upstream revisions in its
newly downloaded partitions; use an explicit historical version selector when
a fixed information cutoff is required. Staging directories without
`request.json` cannot resume.

Source details:

- [NHSN](https://cmu-delphi.github.io/delphi-epidata/api/v5-signals/nhsn.html):
  weekly admissions, reporting coverage, and bed indicators.
- [NSSP](https://cmu-delphi.github.io/delphi-epidata/api/v5-signals/nssp.html):
  weekly ED indicators. The configured pull excludes the MSA partition; it can
  be requested explicitly with `--geo-type msa`.
- [NWSS](https://cmu-delphi.github.io/delphi-epidata/api/v5-signals/nwss.html):
  COVID-19, influenza A, and RSV concentrations at native sewershed support.
  `reference_time` is sample collection date. Keep `nwss_source` and
  `sample_index` in the natural key and retain `pcr_target`. Location and
  population metadata are in Delphi's separate `aux_data` table; this downloader
  does not yet acquire that table or construct state wastewater aggregates.
- [Inpatient claims](https://cmu-delphi.github.io/delphi-epidata/api/v5-signals/claims_inpatient.html)
  and [outpatient claims](https://cmu-delphi.github.io/delphi-epidata/api/v5-signals/claims_outpatient.html):
  daily seven-day percentages for influenza, COVID-19, and other ARI. These use
  the `claims_inpatient_adm_pct_*` and `claims_outpatient_ov_pct_*` signal
  names. Suppressed windows are absent rows; these feeds need revision-aware
  evaluation because claims backfill substantially. Claims rows carry blank fill
  labels even though the source docs describe `source`; imposing that filter
  would discard these rows.

Full reference-date history is not necessarily full historical release coverage.
Inspect the saved `reference_time_range` and `report_time_range` before selecting
backtest periods.

### Hubverse

Repositories are maintained as read-only, blob-filtered Git mirrors. Exports are
pinned to an exact commit; `--hub-as-of` resolves the last first-parent commit at
or before a historical UTC cutoff.
