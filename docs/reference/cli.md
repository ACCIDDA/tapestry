# Command-line reference

The installed commands are `tapestry-data`, `tapestry-dataset`,
`tapestry-experiment`, `tapestry-select`, and `tapestry-explore`. The
checkout scripts shown below are equivalent.

## Data repository

```bash
# Print the complete catalog.
python scripts/pull_covariates.py catalog

# Initialize data/catalog.json and repository directories.
python scripts/pull_covariates.py --data-root data init

# Preview or execute groups.
python scripts/pull_covariates.py --data-root data pull --group core --dry-run
python scripts/pull_covariates.py --data-root data pull --group all

# Inspect and verify immutable snapshots.
python scripts/pull_covariates.py --data-root data show delphi_nhsn
python scripts/pull_covariates.py --data-root data verify delphi_nhsn
```

Groups are `all`, `core`, `cdc`, `delphi`, `hubverse`, and `pophive`. Dataset keys can be
given instead of a group.

### Delphi selection and resume

```bash
python scripts/pull_covariates.py --data-root data pull delphi_nssp \
  --mode archive \
  --signal pct_ed_visits_influenza \
  --geo-type state \
  --workers 4

python scripts/pull_covariates.py --data-root data pull delphi_nssp \
  --signal pct_ed_visits_influenza --geo-type state \
  --resume-from data/.staging/delphi_nssp/<staging-id>
```

Relevant options include `--mode`, `--snapshot-date`, `--report-time`,
repeatable `--signal` and `--geo-type`, optional `--fill-method`, and bounded
`--workers`.
`--report-time` accepts a date or comparison such as `'<2026-09-01'` and is
valid only in archive mode. `--snapshot-date` is valid only in snapshot mode.
Both modes retain `reference_time` observation dates and full `report_time`
timestamps. Requests select the full observation history and all published fill
variants by default. Use `--fill-method source` only for a source that publishes
that variant; claims return blank labels.

The `delphi` group contains `delphi_nhsn`, `delphi_nssp`, `delphi_nwss`,
`delphi_claims_inpatient`, `delphi_claims_outpatient`, `delphi_fluview_ilinet`,
`delphi_fluview_clinical`, and `delphi_flusurv`.

```bash
python scripts/pull_covariates.py pull delphi_claims_inpatient \
  --signal claims_inpatient_adm_pct_claims_flu --geo-type nation \
  --report-time 2026-09-11
```

Resume requires identical query selectors and a matching saved `request.json`.
Use the original command with `--resume-from` added; `--workers` can change.
Staging directories without a query record must be restarted.

### Hub historical state

```bash
python scripts/pull_covariates.py --data-root data pull hub_flusight_current \
  --hub-as-of 2025-01-15
```

Use `--hub-ref` for an explicit branch, tag, or commit.

## Explorer

```bash
# Build only; reuse a current index.
python scripts/explore_covariates.py --data-root data index

# Force a complete index rebuild.
python scripts/explore_covariates.py --data-root data index --force

# Ensure the index is current, then serve it.
python scripts/explore_covariates.py --data-root data serve

# Serve the existing index immediately.
python scripts/explore_covariates.py --data-root data serve --no-index

# Bind another local port without opening a browser.
python scripts/explore_covariates.py --data-root data serve \
  --port 8877 --no-browser

# Write the thinned static copy published as the live explorer on GitHub Pages.
python scripts/explore_covariates.py --data-root data export --out docs/explorer/data

# Index, export, and optionally preview the published copy in one step.
scripts/update_published_explorer.sh --preview
```

See [Published explorer](../explorer/overview.md#published-explorer) for what the
export keeps and how it is committed.

## Shared selection inventory

```bash
PYTHONPATH=src python -m tapestry.data.selection --data-root data
# After installing the package:
tapestry-select --data-root data
```

This read-only command reports policy version, the 25-to-15 grouping, the
53-measure NHSN allowlist size, Delphi/CDC crosswalks, and missing downloads. See
[Shared selection](../data/selection.md) for the downstream interface.


## Nowcasting and forecasting

One manager serves `task=nowcast`, `task=forecast`, and `task=pipeline`:
`python -m tapestry.experiment.planner`. Use `nowcast.<field>` and
`forecast.<field>` for stage-specific settings. The current
[training guide](../design/nowcast-forecast.md#manager-commands) gives plan,
launch, status and rank commands, including Longleaf.


For the fresh standalone forecasting comparison, use
`bash experiments/forecast-covariates.sh`. See the
[protocol and launch commands](../design/forecast-covariates.md).
