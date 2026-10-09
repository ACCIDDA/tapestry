# Selection and provenance

## Shared post-intake selection

The selection layer lives in `chromantis.data.selection`, between immutable raw
intake and consumers. The explorer uses it directly. Downstream analysis uses the
same table decisions, measure allowlists, and canonical signal descriptions.
The first selection step admits only native **state and national** observations.
Unsupported partitions are skipped before opening them. Mixed tables are filtered
row by row before measure selection and Hub conflict validation: HHS, HSA, county,
catchment, and wastewater-site observations are discarded from the selected stream.
A state field on a finer observation is context, not statewide support. National-only
sources may omit geographic columns. Raw snapshots are retained on disk; this filter
does not reclaim their disk space or select the model's final channel set.

```text
Publisher → intake → immutable raw snapshots
                         ↓
                SelectedData (policy v6)
                   ↙              ↘
          ExplorerIndex       native SelectedRecord stream
          display variants    downstream materialization
```

### Source counts

| Family | Acquisition datasets | Logical source groups |
|---|---:|---:|
| NHSN | 4 | 1 |
| NSSP | 4 | 1 |
| NWSS (Delphi archive, auxiliary table, derived state indices) | 3 | 1 |
| Inpatient and outpatient claims | 2 | 2 |
| FluView (ILINet, clinical labs) | 2 | 1 |
| FluSurv-NET | 1 | 1 |
| Kinsa (via PopHIVE) | 1 | 1 |
| Current and historical forecast hubs | 6 | 3 additional (current hubs join NHSN/NSSP) |
| **Total** | **23** | **11** |

These are catalog definitions, not claims that every source has been downloaded
or every geography is displayable. The selection summary reports missing
acquisitions. 

A **signal** is a meaningful measure within a logical group. A **variant** retains
its acquisition source, cadence, release product, geography, units, smoothing,
and demographic/fill facets. The explorer displays one checkbox per variant
under each signal, so multiple releases or providers can be selected together.
Grouping does not average providers or replace a missing
variant with another provider's values. Finalized native-state CDC NHSN is the
preferred display variant when available. Native-state support is the default
UI filter; change it to see national context.

Choice counts are reported as `indexed_signal_choices` and `indexed_variants`
from `/api/catalog` after a rebuild. Counts vary with local acquisitions and
publisher schema changes.

### NHSN: CDC measures and Delphi mirrors

The current research selection is **all ages only**. For each of COVID-19,
influenza, and RSV, it retains total admissions, total hospitalized patients,
total ICU patients, and hospitals reporting admissions. Two shared fields retain
inpatient beds and occupied inpatient beds: **14 measures total**.
Adult, pediatric, individual age-band, and unknown-age fields are excluded from
selection, but remain in immutable raw downloads. Rates, percentages, changes,
and seasonal cumulative sums remain excluded from NHSN outcomes.

Each CDC pull saves the publisher's `fieldName`, `name`, `description`, and
`dataTypeName` from `/api/views/<dataset-id>.json` in `columns.json`, alongside
complete `metadata.json`. Raw rows keep API field identifiers. Selected signal
labels use the publisher's `name`, with descriptions available as explorer
hover text. Older downloads use their saved metadata directly.

Delphi and current Hub outcomes join their documented originating CDC column.
For example, `confirmed_admissions_flu_ew` and FluSight's `wk inc flu hosp`
join NHSN `totalconfflunewadm` (CDC label: “Total Influenza Admissions”).
Delphi's `signal` values are its measures; `value` is the storage field.
Hub `target` values likewise identify measures stored in `observation`.
Unmapped historical Hub sources retain their distinct origins.

The three CDC products remain distinguishable as **Finalized**, **Preliminary**,
and **First publication**. Delphi contributes its configured NHSN signals as
archive variants of their matching CDC columns; it does not supply every CDC
measure (for example, adult-only hospitalized and ICU counts). The
first-publication product is not a full revision archive. See
[Vintages and geography](methods.md).

### NSSP: one emergency department surveillance group

Signals are ED-visit percentages for COVID-19, influenza, RSV, ARI, and combined
respiratory pathogens where published. Daily, weekly, and national demographic products remain variants under their
corresponding CDC measure. Reported and smoothed measures use distinct CDC
columns; Delphi joins the matching column. Current Hub ED targets join the
reported column, with their 0–1 proportions explicitly distinguished from
CDC/Delphi percentages. Values are not rescaled by grouping.

- Daily and demographic CDC tables retain `percent_visits`.
- CDC trajectories retain the four reported and four smoothed percentage fields.
  Socrata's `percent_visits_smoothed` maps to combined and
  `percent_visits_smoothed_1` maps to influenza, as defined by its metadata.
- Delphi retains its nine configured weekly reported/smoothed signals.
- Trend classifications, location identifiers, and build metadata are not outcomes.

Both downstream selection and the explorer retain only state/national support.
CDC trajectories require `county=All` and no specific HSA; county copies of an HSA trajectory must
not be averaged into an apparent state observation. Demographic and smoothed
variants remain separate; percentages and hub ED decimal proportions are not
silently treated as equal units.

### Wastewater and finer geography

The currently cataloged NWSS products contain site/sewershed observations, even when their
rows carry a state label. They remain in the raw acquisition catalog but are
excluded from both `SelectedData` and the explorer. No site-to-state averaging
or geographic crosswalk is performed. CDC also publishes a separate weekly
[state/territory WVAL product](https://www.cdc.gov/wastewater/respiratory-viruses/state.html)
for influenza A, COVID-19, and RSV; it is not yet in this acquisition catalog. HHS and catchment observations are excluded
as well; a genuine national row from a mixed catchment/national source can remain.

### FluView and FluSurv-NET

Policy v6 (2026-09-22) adds the Delphi V5 FluView and FluSurv-NET archives. Each
retains `value` for its configured catalog signals only.

- **FluView** is one group with two components. Signal keys carry a component
  prefix (`fluview:ilinet_ili`, `fluview:clinical_pct_positive`). ILINet signals
  are filed under *Influenza-like illness*, laboratory signals under
  *Influenza*. `age_group` and `fill_method` stay variant facets: ILINet visit
  counts are age-stratified, and statewide New York is Delphi's
  `nyc_plus_ny_minus_nyc` pool of CDC's two New York jurisdictions (NYC and
  NY-minus-NYC are not states and are excluded).
- **FluSurv-NET** signals are catchment hospitalization rates per 100,000,
  labelled by stratum (`rate_age_0` is ages 0–4, following Delphi's legacy V3
  numbering). The state variants cover only FluSurv-NET catchment states; the
  national row is the network rate. `misc` network rows (EIP, IHSP) and MSA
  rows are not state or national support and are excluded.

Delphi's public health laboratory source is not acquired (its state rows are
season totals). See [FluView and FluSurv-NET](sources.md).

### Hubs: canonical observations grouped under their origins

| Hub | Canonical file selection | Observed target choices |
|---|---|---:|
| Current FluSight | `target-data/time-series.csv` | 2 |
| Current COVID | `target-data/time-series.parquet` | 2 |
| Current RSV | `target-data/time-series.parquet` | 2 |
| Legacy FluSight | `data-truth/truth-Incident Hospitalizations.csv` | 1 |
| Legacy COVID | Six top-level incident/cumulative case/death/hospitalization truth files | 6 |
| RSV-NET | Latest top-level dated `rsvnet_hospitalization.csv` | 2 |
| **Expected definitions** | | **15** |

Current hubs use `observation`; legacy hubs and RSV-NET use `value`. The target
field, or the legacy truth filename, defines the signal. Age strata are variants.
Population, location codes, quantiles, oracle output, redundant exports,
auxiliary tables, and alternative truth providers do not create outcome series.
Derived peak/rate-change evaluation definitions are not independent observation
streams. The rate exclusion for NHSN does not remove the RSV-NET rate target.

Current hub files retain their `as_of` releases. Intake additionally materializes
configured target files without native release dates into `git-history.ndjson.gz`,
with a release/commit/path/blob index in `git-history.json`. These are canonical
Git-history variants of the same observations, not new outcome definitions.
Preferred filenames replace earlier aliases; dated backups and oracle outputs
remain excluded. The selected stream and explorer read these saved releases.
Complete-snapshot resolution respects omitted rows and whole-file deletions.
See [the Git vintage policy and reproduction commands](methods.md).

Canonical hub tables are checked across the entire file before yielding rows.
Identical duplicate keys are emitted once. Conflicting location/target/event/
release/age keys are quarantined as missing (`_selection_conflict=True` in native
metadata), preserving the release timestamp so an earlier value cannot silently
replace the conflict. Other observations remain available. Conflicts appear in
the source audit as `quarantined`.

Missing canonical files and Git LFS pointers are explicitly unavailable. There
is no fallback to USAFacts, NYTimes, or another truth provider. In the local
legacy COVID export, all six primary files are LFS pointers. Intake must
retrieve their payloads before those choices can be plotted.

### Downstream API

```python
from chromantis.data.selection import SelectedData, describe

selected = SelectedData("data")
print(selected.summary())

for record in selected.iter_records(
    group="nhsn",
    dataset_key="cdc_nhsn_final",
):
    # Only record.values contains selected outcomes.
    # metadata carries native geography, dates, QC, and other raw context.
    for column, value in record.values.items():
        signal = describe(record.dataset_key, column, record.source_path,
                          record.metadata)
        consume(signal["signal_key"], value, record.metadata,
                record.available_at)
```

Specify `dataset_key` when a downstream task needs one particular provider or
release product. Omitting it exposes all selected variants with provenance; it
does not combine them. Raw suppression markers and nulls remain unchanged.

`available_by="2026-01-09T12:00:00Z"` filters out later releases using the publisher
vintage column, or conservatively the saved commit/retrieval time. Unknown
availability is excluded when a cutoff is supplied. It **emits all eligible
revisions**, not a resolved training panel. Downstream materialization must
choose the latest eligible observation or full release, handle retractions,
validate native support and units, and apply an event-date cutoff where required.
A recent raw download does not establish historical availability.

`selected_tables()` is the lower-level native table interface used by the
explorer. Apply `measure_columns()` to choose outcome columns. Consume each table
before advancing the iterator: temporary extraction of a tar member lasts only
for that iterator step. No training data should be built by reading the
explorer's aggregated state-point cache.

### Commands, audit, and maintenance

```bash
## Policy counts, grouping, and missing downloads; no raw or index writes.
PYTHONPATH=src python -m chromantis.data.selection --data-root data

## Rebuild selected explorer data; reads raw snapshots without changing them.
python -m chromantis.explorer --data-root data index --force
```

The installed summary command is `chromantis-select --data-root data`.
`/api/catalog` includes the selection version, logical and indexed source counts,
canonical signal count, variant count, and unavailable source warnings.
`SelectedData.audit` records excluded hub files and missing canonical files after
iteration. The index's `sources` table also records excluded paths and read errors;
**Index details** emphasizes errors and unavailable inputs rather than listing
hundreds of deliberately excluded files.

To change selection, update the explicit mappings and `POLICY_VERSION` in
`data/selection.py`, update its regression tests, and rebuild. The policy version
participates in index invalidation. New NHSN/NSSP/NWSS numeric fields are not
silently admitted as outcomes. Other source families retain their existing
measure discovery and source grouping.


## Vintages and geography

### Revision semantics

`versioned` means the source can recover a historical information state; it does
not merely mean that local downloads are timestamped.

| Revision mode | Interpretation | Versioned |
|---|---|---:|
| `snapshot_only` | Current mutable publisher view; local pulls remain immutable | no |
| `initial_release` | Frozen first-publication product | yes |
| `as_of_column` | Rows carry publisher release dates | yes |
| `report_time` | Delphi archive carries retained report-time revisions | yes |
| `git_history` | Files are recoverable at an exact Git commit | yes |

The raw repository preserves all available vintage fields. The explorer supports an explicit as-of cutoff and overlays of multiple versions.
Report-time archives select the latest eligible revision per event date; full
`as_of` tables select a complete release. Unversioned sources apply only an
event-date cutoff and cannot reconstruct past revisions.

**No change does not mean no version.** Delphi is a change history: for each
observation, the value advertised at `report_time` applies until another advertised
change replaces it. No new row on a Wednesday is required. A Hub commit gives the
actual repository state, including unchanged observations; that state remains in
force between target-file changes. Skipping duplicate blobs or unchanged rows is
storage compression, not a break in coverage. Explicit nulls, removals, and dates
before the first advertised version remain distinct from unchanged valid values.
This persistence is along publication time for the same observation; it never
fills a different event week or carries information backward before publication.

Hub acquisitions now include `git-history.ndjson.gz` and `git-history.json` for
configured canonical target CSVs without native release dates. Intake walks the
pinned main branch's first-parent history and saves complete target-file states,
including empty releases after deletion. The selected stream and explorer consume
that saved history; they do not run Git on each query. In the explorer these are
**Git commit history** variants beside the native `as_of` variants of the same
measure. The date cutoff selects a complete eligible Git snapshot, preserving
omissions and retractions rather than carrying removed values forward.

**Repository ground truth:** a commit fixes the complete target-file contents.
**Timing assumption:** its main-branch committer timestamp places that state on
the availability timeline; it does not establish the provider release time. Author time, observation dates,
filename dates, and retrieval time are not substituted. Backdated target changes
cannot predate the preceding target state; equal-time changes resolve to the last
first-parent state. Only history reachable from the pinned commit is exported.
Original commit hashes, paths and blob hashes are retained. CSV `date` and
`target_end_date` are explicit event-date aliases, never release dates. Current
Hub ED CSV values are already proportions; no percentage conversion is applied.
Once a preferred filename has appeared, an obsolete alias cannot replace it
after deletion, even if that older file remains in the Git tree.

`hub-history` augments the latest acquisition at its existing pinned commit;
ordinary Hub pulls also create these histories. No user's worktree is checked out.

```bash
.venv/bin/python -m chromantis.data --data-root data hub-history \
  hub_flusight_current hub_covid_current hub_rsv_current hub_flusight_legacy
.venv/bin/python -m chromantis.explorer --data-root data index --force
.venv/bin/python -m chromantis.explorer --data-root data export
```

The path policy covers historical current-FluSight admissions and ED files,
current-COVID admissions, and legacy FluSight/COVID primary truth files. Current
RSV has native `as_of` history and no configured unversioned predecessor. Legacy
COVID LFS pointers still require their actual payloads; a pointer is not a usable
historical observation. RSV-NET catchment files are a separate source outside the
six scored NHSN/NSSP targets and are not folded into statewide Hub outcomes.

The September 17 backfill, pinned to the existing September 16 acquisitions:

| Source | Complete Git releases | Native state/DC/US rows | Commit-time range |
|---|---:|---:|---|
| Current FluSight | 132 | 1,577,922 | 2023-10-03–2026-07-09 |
| Current COVID | 93 | 241,072 | 2024-11-18–2026-09-09 |
| Current RSV | 0 | 0 | Native `as_of` history already present |
| Legacy FluSight | 527 | 3,736,470 | 2021-12-07–2023-11-22 |

Repeated event weeks appear in multiple complete releases; these counts are not
independent observations. Local acquisition IDs and exact pinned commit hashes
are in `data/processed/hub-git-history-summary.json`.

### Geographic support

Native support is never silently relabeled:

- State observations are valid state-level inputs.
- National observations are stored once and shown with explicit national-context labels.
- HHS, HSA, and county observations are excluded at the shared post-intake filter.
- Catchment/network data are not treated as statewide values.
- Wastewater site and sewershed rows remain in raw storage but are excluded from
  the selected stream and explorer. No site-to-state average is computed.

This distinction is essential for the later mask: native state availability,
parent context, and unavailable state data need not share the same mask policy.


## Storage and provenance

### On-disk layout

```text
data/
├── catalog.json
├── mirrors/
│   └── hub_flusight_current.git/
├── raw/
│   └── cdc_nhsn_final/
│       ├── latest.json
│       └── snapshots/<retrieval-id>/
│           ├── data.ndjson.gz
│           ├── metadata.json
│           └── manifest.json
├── .staging/
└── .explorer/
    ├── series.sqlite3
    └── revisions.parquet
```

`raw/` and `mirrors/` are durable source material. `.explorer/` is disposable
and can always be reconstructed. `series.sqlite3` contains searchable series
metadata plus the latest resolved point cache; `revisions.parquet` contains the
complete revision ledger used for as-of reads. `.staging/` contains in-progress work and may
also contain validated Delphi partitions retained after an interrupted pull.

### Atomic snapshots

A fetcher obtains a `SnapshotWriter`, streams its files into an isolated staging
directory, records row counts and metadata, and commits only after every file is
complete. Commit writes the manifest and atomically advances `latest.json`.

If a pull fails, an incomplete snapshot is not visible through `latest()`. Large
Delphi V5 pulls can deliberately preserve gzip CRC-valid partitions and reuse
them with `--resume-from`.

### Manifest contents

Each snapshot manifest records:

- dataset key and immutable retrieval identifier;
- retrieval time, source URL, and revision mode;
- whether the source exposes historical versions;
- each payload path, byte count, row count, and SHA-256 digest;
- fetcher-specific provenance such as source metadata or a Hub Git commit.

Verify the latest snapshot with:

```bash
python -m chromantis.data --data-root data verify cdc_nhsn_final
```

### Data lifecycle

Raw snapshots are append-only. Cleaning the disposable explorer index does not
remove source data. Before deleting staging or dated backup directories, inspect
them for resumable partitions or recovery material.

## Same-date conflicts

Different finite values for the same signal, location, observation date and
report date have no release ordering. The model extractor leaves the ambiguous
cell unavailable rather than choosing an arbitrary value. Availability evidence
counts a finite published statement even when it conflicts; source publication
and model usability are different measurements. Later unambiguous reports can
restore usability. Missing values never become observed zeroes.
