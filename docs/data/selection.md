# Shared post-intake selection

The selection layer lives in `influpaintx.data.selection`, between immutable raw
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
                SelectedData (policy v4)
                   ↙              ↘
          ExplorerIndex       native SelectedRecord stream
          display variants    downstream materialization
```

## Source counts

| Family | Acquisition datasets | Logical source groups |
|---|---:|---:|
| NHSN | 4 | 1 |
| NSSP | 4 | 1 |
| NWSS | 5 | 1 |
| NREVSS | 4 | 4 |
| Inpatient and outpatient claims | 2 | 2 |
| Current and historical forecast hubs | 6 | 3 additional (current hubs join NHSN/NSSP) |
| **Total** | **25** | **12** |

These are catalog definitions, not claims that every source has been downloaded
or every geography is displayable. The selection summary reports missing
acquisitions. Migrated, retired Delphi FluView/clinical/FluSurv snapshots may
remain on disk; they are excluded from the selected inventory.

A **signal** is a meaningful measure within a logical group. A **variant** retains
its acquisition source, cadence, release product, geography, units, smoothing,
and demographic/fill facets. The explorer displays one checkbox per variant
under each signal, so multiple releases or providers can be selected together.
Grouping does not average providers or replace a missing
variant with another provider's values. Finalized native-state CDC NHSN is the
preferred display variant when available. Native-state support is the default
UI filter; change it to see national context.

The earlier planning estimate of 685 choices was based on an old, unfiltered
explorer index and preceded NSSP/NWSS grouping. It is superseded by the actual
`indexed_signal_choices` and `indexed_variants` counts from `/api/catalog` after
a rebuild. Counts vary with local acquisitions and publisher schema changes.

## NHSN: CDC measures and Delphi mirrors

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
[Vintages and geography](vintages-and-geography.md).

## NSSP: one emergency department surveillance group

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

## Wastewater and finer geography

The currently cataloged NWSS products contain site/sewershed observations, even when their
rows carry a state label. They remain in the raw acquisition catalog but are
excluded from both `SelectedData` and the explorer. No site-to-state averaging
or geographic crosswalk is performed. CDC also publishes a separate weekly
[state/territory WVAL product](https://www.cdc.gov/wastewater/respiratory-viruses/state.html)
for influenza A, COVID-19, and RSV; it is not yet in this acquisition catalog. HHS and catchment observations are excluded
as well; a genuine national row from a mixed catchment/national source can remain.

## Hubs: canonical observations grouped under their origins

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

Current hub files retain their `as_of` releases. Full-snapshot resolution must
respect omissions/retractions. Legacy histories remain in raw archives and Git;
selection chooses a canonical file, rather than making every dated copy a new
series. `SelectedData.iter_records` reads the selected local snapshot; it does
not traverse Git history. For a historical legacy export, acquire the desired
commit using `--hub-as-of` first. The explorer uses only the selected snapshot and its recorded releases; it does
not read Git history.

Canonical hub tables are checked across the entire file before yielding rows.
Identical duplicate keys are emitted once. Conflicting location/target/event/
release/age keys are quarantined as missing (`_selection_conflict=True` in native
metadata), preserving the release timestamp so an earlier value cannot silently
replace the conflict. Other observations remain available. Conflicts appear in
the source audit as `quarantined`.

Missing canonical files and Git LFS pointers are explicitly unavailable. There
is no fallback to USAFacts, NYTimes, or another truth provider. In the audited
local legacy COVID export, all six primary files were LFS pointers. Intake must
retrieve their payloads before those choices can be plotted.

## Downstream API

```python
from influpaintx.data.selection import SelectedData, describe

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

## Commands, audit, and maintenance

```bash
# Policy counts, grouping, and missing downloads; no raw or index writes.
PYTHONPATH=src python -m influpaintx.data.selection --data-root data

# Rebuild selected explorer data; reads raw snapshots without changing them.
python scripts/explore_covariates.py --data-root data index --force
```

The installed summary command is `influpaintx-select --data-root data`.
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
