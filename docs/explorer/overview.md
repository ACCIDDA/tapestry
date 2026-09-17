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

Hub targets without an `as_of` column can still have release history in Git.
The **Git commit history** variants show canonical target-file states saved by
intake from the pinned branch's first-parent commits. Select an as-of date to
resolve the latest eligible complete state, including removed observations.
Native `as_of` variants remain separate. A commit fixes repository contents; its
committer timestamp dates that state, not the provider release clock; see [the vintage policy](../data/vintages-and-geography.md).

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
recover. Indexes built with a different schema or selection policy are rebuilt.

`status` and `validate` emit JSON. Storage errors cause a nonzero exit; stale
indexes and source errors are reported separately. Validation checks SQLite
`quick_check`/`foreign_key_check`, Parquet schema and row count, and matching build
IDs. It does not compare every value to raw data. Strict mode rejects source
errors; intentional exclusions and quarantined conflicts remain audit entries.

## Using the data

A banner under the title notes that the data come from CMU Delphi, the CDC, and the
respiratory forecast hubs, that this special-purpose explorer comes with no
guarantee, and points to Delphi [EpiVis](https://delphi.cmu.edu/epivis/) and
ACCIDDA [RespiLens](https://respilens.org), which inspired it, for a useful dashboard.
Dismissing it with × lasts only for the current page load; it returns on every visit.

Choose a location, search measures, and select acquisition/release, demographic,
smoothing, or fill variants. Native-state signals are the default filter; choose
national context to see national observations. These are candidate measurements,
not an automatically chosen model channel set.

CDC API display names label the measures, with publisher descriptions on hover.
The API field and Delphi signal/Hub target identifiers remain visible in each
variant. Current Hub hospitalizations join NHSN, and ED targets join NSSP, at
their corresponding CDC columns. Reported and smoothed NSSP columns remain
separate. Historical Hub measures with different origins stay separate.

Delphi variants in the signal list are orange and Hub variants blue. Each variant is one line named as a path (e.g. `Finalized/weekly/native state`); checking it opens a card with its point count, date range, and provenance. Each
plotted series gets one color, shared by all its dated versions; line style
(solid latest, dashed as-of) marks the version. The legend sits beside
the location title with one column per series: its real (latest) data on top and
the as-of revision directly below, reading *data / source* (plus the as-of date).
Revisions are also drawn underneath the real data. Hover an entry for the full
variant, maximum, and lineage. Hub target data is fixed: black for admissions (light gray in dark mode)
and red for ED visits. Lines are drawn at 70% opacity, and each selected series has its own line width
(widest drawn first) and marker shape (circle, square, triangle, cross, diamond;
hollow for revisions) at staggered positions, so identical series stay visible
as a thin line inside a thicker one with alternating markers.
Colors stay consistent across the chart, legend, overview, and hover values, and
remain stable when other lines are added or removed. Clear resets the color allocation.
NHSN selection currently
retains 14 all-age measures and excludes adult, pediatric, age-band, and
unknown-age fields. Raw snapshots keep those fields.

Signal selections persist when switching states, including signals unavailable in
the new state (shown with no points). **United States (US)** appears when the index
contains native national observations; it reads those published US records only,
never a sum of states. Its spatial support is fixed to national. Selections follow
the location: switching between US and states (or between states) swaps each selected
variant for its counterpart in the new location when source, measure, and
non-geographic dimensions match (Delphi `geo_type` files and NSSP `trend_source` are
treated as geography). National-only products stay as labeled national context.
National-context signals explicitly selected while viewing a state remain national.

**Hub presets** (COVID-19 Hub, FluSight, RSV Hub) replace the selection with the
Hub's target data and its Delphi ground truth for the current location: NHSN
weekly admissions and reported NSSP ED visit percentage, four series in all.
They also turn on **Divide by mean**, since counts,
percentages, and proportions only overlay once scaled.

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

Selecting an **As of** date automatically overlays two curves per signal: latest
available values as a solid line, and values available at the selected date as a
dashed line in the same color. "Latest" is the reference for finalized values
here; the publisher may still revise it. The selected date's weekday is shown
beside the picker and in the legend, so publisher release days (e.g. Wednesday)
are easy to spot. A dashed crimson line marks the as-of date on the chart;
clicking anywhere on the chart sets the as-of date to that day. **← Wed**
and **Wed →** step the date to the previous or next calendar Wednesday (from
today when showing latest values); the next step stops at today.

The **As of** control uses publisher revisions already present
in the selected raw snapshot. Delphi's last eligible advertised value remains
available when there is no new change. Hub states likewise persist until a later
complete state replaces them: unchanged rows still belong to the version.
Full `as_of` releases preserve omissions and retractions. Unversioned data uses an event-date cutoff. Inputs without row
publication dates are bounded by their saved snapshot/commit timestamp.
The explorer resolves the Git history materialized during intake; it does not
fetch or traverse Git during a query. `hub-history` backfills a pinned acquisition,
and `--hub-as-of` can acquire a different historical tip when needed.

Repeated rows for a state, measure, date, and release are shown as their
unweighted mean, with the contributing sample count. Null latest revisions do
not resurrect earlier values. Divide by mean divides every displayed version by
the same series' latest mean over the common window, so dated versions with a
shorter archived history (e.g. COVID-19 Hub snapshots before 2026-09-09 start at
2024-11-09) stay comparable. A bar under the header and a spinner on the chart
show while data is loading. **Index details** lists unavailable, excluded, and malformed sources.

See the [local API](api.md) for catalog, series, versions, and data endpoints.

## Published explorer

A static copy runs on GitHub Pages at
[accidda.github.io/tapestry/explorer/](https://accidda.github.io/tapestry/explorer/).
It uses the same page as the local server; when `data/catalog.json` sits next to
the page, the browser answers the catalog, series, versions, and data requests
itself and reads revisions with [hyparquet](https://github.com/hyparam/hyparquet)
from one Parquet file per series, downloaded whole when that series is plotted.
GitHub Pages gzips responses and applies byte ranges to the compressed stream, so
ranged reads of a single large Parquet file do not work there.

The copy is a thinned export of the local index, committed to
`docs/explorer/data/`. It is not updated live or by CI:

- Releases collapse to one per Wednesday week (the last release on or before
  each Wednesday), and rows that repeat the previous kept value are dropped.
  Those rows still have an as-of version: the retained value persists until its
  next change. This is change-log compression, not missing historical coverage.
- Delphi inpatient/outpatient claims keep Wednesday snapshots only for the first
  8 weeks after each date, plus each date's latest value.
- Full-snapshot Hub target files become a change log that keeps removals, so an
  as-of date resolves as it does locally.
- **As of** dates are Wednesdays only: chart clicks and typed dates snap to the
  nearest Wednesday (never after today), and ← / → step through Wednesday weeks
  with releases. These match the local explorer's values for that Wednesday.

The published banner adds that the online version is not updated (with its export
date) and that the local explorer is the ground-truth source. The docs header links
to it as **Live explorer**.

The September 2026 export keeps 6.0 million of 165.5 million ledger rows
(28 MB: 373 per-series Parquet files, the largest about 5 MB for daily claims,
plus series metadata). In a sampled check against
the local server, latest values and Wednesday as-of dates matched exactly for
non-claims series; claims matched for dates within 8 weeks of the as-of date.

To refresh it after pulling new raw data:

```bash
scripts/update_published_explorer.sh
git add docs/explorer/data
git commit -m "Update published explorer data"
git push
```

The script runs `index` (a no-op when raw data is unchanged) and then
`explore_covariates.py export --out docs/explorer/data`. On push to `main`,
the Documentation workflow builds MkDocs, which copies the committed data, adds
`index.html`, `app.js`, and `style.css` from `src/tapestry/explorer/static/`, and
deploys the site.
