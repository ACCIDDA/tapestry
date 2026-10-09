# Source catalog

The checked-in catalog defines 23 independently retrievable datasets. The table
below summarizes acquisition and modeling metadata; run
`python -m chromantis.data catalog` for the complete machine-readable
specifications, signals, natural keys, and source URLs.

| Dataset key | Provider | Fetcher | Cadence | Versioned | Native geography |
|---|---|---|---|---:|---|
| `cdc_nhsn_final` | CDC | `socrata` | `weekly` | no | state, territory, hhs, nation |
| `cdc_nhsn_initial_release` | CDC | `socrata` | `weekly` | yes | state, territory, hhs, nation |
| `cdc_nhsn_preliminary` | CDC | `socrata` | `weekly` | no | state, territory, hhs, nation |
| `cdc_nssp_daily` | CDC | `socrata` | `daily` | no | state, nation |
| `cdc_nssp_demographics` | CDC | `socrata` | `weekly` | no | nation |
| `cdc_nssp_trajectories` | CDC | `socrata` | `weekly` | no | state, hsa |
| `delphi_claims_inpatient` | CMU Delphi | `delphi_v5` | `daily` | yes | county, hrr, msa, state, hhs, census region/division, nation |
| `delphi_claims_outpatient` | CMU Delphi | `delphi_v5` | `daily` | yes | county, hrr, msa, state, hhs, census region/division, nation |
| `delphi_flusurv` | CMU Delphi | `delphi_v5` | `weekly` | yes | state, nation, msa, misc |
| `delphi_fluview_clinical` | CMU Delphi | `delphi_v5` | `weekly` | yes | state, nation, hhs, census division |
| `delphi_fluview_ilinet` | CMU Delphi | `delphi_v5` | `weekly` | yes | state, nation, hhs, census division |
| `delphi_nhsn` | CMU Delphi | `delphi_v5` | `weekly` | yes | state, nation, hhs, census region/division |
| `delphi_nssp` | CMU Delphi | `delphi_v5` | `weekly` | yes | state, nation, hhs, county, hsa, hrr, msa, census region/division |
| `delphi_nwss` | CMU Delphi | `delphi_v5` | `sample` | yes | sewershed |
| `delphi_nwss_aux` | CMU Delphi | `delphi_v5_aux` | `sample` | yes | sewershed |
| `derived_nwss_state_indices` | Chromantis, derived from CMU Delphi NWSS | `derived` | `weekly` | yes | state, nation |
| `hub_covid_current` | Hubverse community | `hubverse` | `weekly` | yes | state, nation |
| `hub_covid_legacy` | Hubverse community | `hubverse` | `weekly` | yes | county, state, nation |
| `hub_flusight_current` | Hubverse community | `hubverse` | `weekly` | yes | state, nation |
| `hub_flusight_legacy` | Hubverse community | `hubverse` | `weekly` | yes | state, nation |
| `hub_rsv_current` | Hubverse community | `hubverse` | `weekly` | yes | state, nation |
| `hub_rsvnet` | Hubverse community | `hubverse` | `weekly` | yes | catchment, nation |
| `pophive_kinsa_ili` | PopHIVE (Kinsa Insights) | `pophive_git` | `daily` | yes | nation |

The `Versioned` column includes both preserved initial releases and archives
with multiple release dates. A `yes` does not necessarily mean every revision
is available; see the NHSN distinction below.

For the consumer-facing organization and retained measures, see
[Shared selection](methods.md): 23 acquisitions map to 11 logical groups.

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

The Delphi V5 sources listed above are enabled (NHSN, NSSP, NWSS with its
auxiliary table, both claims feeds, FluView ILINet, FluView clinical and public
health laboratories, and FluSurv-NET). Direct CDC and Hubverse sources are separate.

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

- [FluView ILINet](https://cmu-delphi.github.io/delphi-epidata/api/v5-signals/fluview_ilinet.html),
  [FluView clinical labs](https://cmu-delphi.github.io/delphi-epidata/api/v5-signals/fluview_resp_lab_clinical.html)
  and [FluSurv-NET](https://cmu-delphi.github.io/delphi-epidata/api/v5-signals/flusurv.html):
  the V5 replacements (September 2026) for the legacy V3 `fluview`,
  `fluview_clinical` and `flusurv` endpoints. FluView rows carry an extra
  `age_group` key column. The public health labs (`fluview_resp_lab_ph`) are
  not acquired (state rows are season totals). Release timing, archive gaps and the New York pool are
  in [FluView and FluSurv-NET](sources.md).

Full reference-date history is not necessarily full historical release coverage.
Inspect the saved `reference_time_range` and `report_time_range` before selecting
backtest periods.

### PopHIVE Git history

`pophive_git` reads one PopHIVE Ingest `standard/data.csv.gz` table and turns its
commit history into a report-time archive: a row's report time is the first
commit that contains it. It lists the file's commits through the GitHub API and
downloads each version from raw.githubusercontent.com rather than mirroring the
10 GB repository. Commits can be backfills, so the earliest report times
are when a value entered PopHIVE, not when its publisher released it. Details,
terms and the Kinsa commit history are in [Kinsa (PopHIVE)](sources.md).

### Hubverse

Repositories are maintained as read-only, blob-filtered Git mirrors. Exports are
pinned to an exact commit; `--hub-as-of` resolves the last first-parent commit at
or before a historical UTC cutoff.

Canonical target files without row release dates are also exported across their
first-parent Git history. `chromantis.data hub-history DATASET...` adds that history
to a new acquisition at the previously saved commit without refreshing the branch.
The canonical selector and explorer expose it as Git commit history, with the
availability assumptions documented in [Vintages and geography](methods.md).

## Claims

Inpatient and outpatient claims provide daily trailing-seven-day percentages.
The panel selects Saturday values for influenza and COVID-19. These are covariates,
not admission counts. A finite conflicting same-date report establishes source
availability but cannot supply an unambiguous model value; see
[conflict handling](methods.md#same-date-conflicts).

## Kinsa

Kinsa is the national PopHIVE ILI series, recovered from pinned Git snapshots.
Weekly values require all seven daily observations. Episode construction broadcasts
the national value and reporting mask to every location; it does not create
state-specific measurements. Commit time is evidence of repository availability,
not an assumed provider release time. As-of coverage begins with the retained
archive, even when the final series extends farther back.

## FluView and FluSurv

`delphi_fluview_ilinet`, `delphi_fluview_clinical` and `delphi_flusurv` use Delphi V5
archives. ILINet provides ILI percentages, clinical laboratories provide influenza
positivity, and FluSurv provides surveillance hospitalization rates. Geographic
support and gaps remain source-specific; a region or surveillance catchment must
not be relabeled as statewide support. Observation history can be much longer
than usable release history. The scheduled-final protocol assumes a one-week lag
for these three groups; this is an experimental assumption, not measured archive timing.

## Wastewater

The panel uses `derived_nwss_state_indices`, rebuilt from versioned Delphi NWSS
concentrations and auxiliary metadata by `python -m chromantis.dataset.build nwss-indices`.
These WVAL-like indices are derived predictors, not the published CDC WVAL series.

At each report time, use only positive finite samples and auxiliary metadata
available by that time. Group by sewershed, NWSS source, PCR target and laboratory
method. For groups with at least 26 distinct weeks and positive log-scale standard
deviation, compute `exp((log(value) - log_value_p10) / log_value_sd)` and the
within-group log-value percentile. Average within a Saturday-ending week, take
medians across groups at each site, then across sites per state and nationally.
Each state or national week requires at least three sites. Same-release conflicts
become missing; disappearing state-weeks receive explicit null statements.
The formulas and release-boundary handling live in `chromantis.dataset.nwss`.

The earliest retained Delphi NWSS archive vintage is February 25, 2026.
Older observation dates do not establish historical as-of availability.
Run index construction after changing NWSS inputs and before rebuilding the panel.
