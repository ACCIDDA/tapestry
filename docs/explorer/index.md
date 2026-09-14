# Explorer

The explorer consumes the shared [post-intake selection](../data/selection.md):
**native state and national observations only**. HHS, county, HSA, catchment,
and wastewater-site data are excluded before profiling and indexing. A state
label on a site or county row does not make it a state observation.

There is one streaming Python indexer. SQLite holds metadata and latest points;
Parquet holds publisher revisions. National observations are stored once under
`US` and can be shown as explicitly labeled national context in state plots.
Raw snapshots remain unchanged. The index is disposable: outdated or missing
indexes are rebuilt, with no migration path.

## Commands

Install the Parquet dependency and start the explorer from the repository root:

```bash
python -m pip install -e '.[explorer]'
python scripts/explore_covariates.py --data-root data serve --no-browser
```

The server defaults to `http://127.0.0.1:8765/`. Omit `--no-browser` to open it
automatically, or pass `--port 8877` for another port.

```bash
# Build without starting the server; reuse a current index.
python scripts/explore_covariates.py --data-root data index

# Force a rebuild, optionally adjusting buffer and SQLite cache sizes.
python scripts/explore_covariates.py --data-root data index --force \
  --batch-rows 100000 --cache-mb 256

# Read saved metadata, source errors, and freshness without rebuilding.
python scripts/explore_covariates.py --data-root data status

# Check SQLite integrity and the Parquet footer.
python scripts/explore_covariates.py --data-root data validate

# Refuse builds or reuse with source errors.
python scripts/explore_covariates.py --data-root data index --strict

# Serve an existing index without checking the raw inventory.
python scripts/explore_covariates.py --data-root data serve --no-index --no-browser
```

`--batch-rows` bounds the revision buffer; `--cache-mb` sets the SQLite page-cache
budget. These are not a total process-memory limit. Every format uses the same
streaming path; there is no optional Polars indexer. For an isolated experiment,
put `--index-path` in a separate directory before the command, or specify both
`--index-path` and `--ledger-path`.

Builds report time and revision rows per artifact. Failed partial Parquet writes
abort the build and preserve the previous pair. Local file locks reject competing
builders. Stop the server before rebuilding: replacing the two files is not
atomic as a pair. Matching build IDs detect interrupted publication; rebuild to
recover. Schema 13 and selection policy 4 require rebuilding older indexes.

`status` and `validate` emit JSON. Storage errors cause a nonzero exit; stale
indexes and source errors are reported separately. Validation checks SQLite
`quick_check`/`foreign_key_check`, Parquet schema and row count, and matching build
IDs. It does not compare every value to raw data. Strict mode rejects source
errors; intentional exclusions and quarantined conflicts remain audit entries.

## Using the data

Choose a location, search measures, and select acquisition/release, demographic,
smoothing, or fill variants. Native-state signals are the default filter; choose
national context to see national observations. These are candidate measurements,
not an automatically chosen model channel set.

CDC API display names label the measures, with publisher descriptions on hover.
The API field and Delphi signal/Hub target identifiers remain visible in each
variant. Current Hub hospitalizations join NHSN, and ED targets join NSSP, at
their corresponding CDC columns. Reported and smoothed NSSP columns remain
separate. Historical Hub measures with different origins stay separate.

Delphi variants in the signal list are orange and Hub variants green. Every
plotted line, including additional dated versions, gets its own color. Colors
stay consistent across the chart, legend, overview, and hover values, and remain
stable when other lines are added or removed. Clear resets the color allocation.
Provider names remain in labels. The plot status counts **displayed
versions**, not every available publisher release. NHSN selection currently
retains 14 all-age measures and excludes adult, pediatric, age-band, and
unknown-age fields. Raw snapshots keep those fields.

Signal selections persist when switching states, including signals unavailable in
the new state (shown with no points). **United States (US)** appears when the index
contains native national observations; it reads those published US records only,
never a sum of states. Its spatial support is fixed to national. When entering US,
a selected state variant switches to a national counterpart only when source,
measure, and non-geographic dimensions match exactly. Returning to a state restores
the original state variant. Explicitly selected national-context signals remain
national context when viewing a state.

On desktop, the signal browser fills the viewport height and scrolls independently
beside the plot. Narrow screens retain the stacked layout.

**View dates** selects the observation-date window using From/To date inputs or
the overview below the chart. Drag its handles to resize, drag the selected window
to pan, or drag outside it to select a new range. **All dates** restores the full
history. The minimum window is one day; both endpoint dates are included. The
overview scales each signal independently so mixed units remain visible. The main
plot rescales its y-axis to observations within the window; series-maximum scaling
still uses the full displayed version, independent of zoom. View dates do not
change the publication cutoff. A chosen range persists across signal/version
updates while overlapping the available history; it resets when no longer
applicable or when selections are cleared. These controls run locally against
already loaded points, without rebuilding the index or adding chart dependencies.

Selecting a **Data available as of** date automatically overlays two curves per
signal: latest available values as a solid line, and values available at the
selected date as a dashed line in its own color. "Latest" is the reference for
finalized values here; the publisher may still revise it. **Keep for comparison**
pins additional dated curves, while the latest reference remains included.

The **Data available as of** control uses publisher revisions already present
in the selected raw snapshot. Full `as_of` releases preserve omissions and
retractions. Unversioned data uses an event-date cutoff. Inputs without row
publication dates are bounded by their saved snapshot/commit timestamp.
The explorer never traverses or fetches Git history; acquire a historical Hub
snapshot with the intake command's `--hub-as-of` option when needed.

Repeated rows for a state, measure, date, and release are shown as their
unweighted mean, with the contributing sample count. Null latest revisions do
not resurrect earlier values. Scaling divides each displayed version by its
maximum. **Index details** lists unavailable, excluded, and malformed sources.

See the [local API](api.md) for catalog, series, versions, and data endpoints.
