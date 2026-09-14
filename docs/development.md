# Development

## Repository layout

```text
src/influpaintx/
  data/          Acquisition, snapshots, readers and selection
  model_data/    Canonical weekly dataset and windows
  models/        B0, training and season CV
  evaluation/    Shared scoring, hub comparisons, exports and reports
  explorer/      Index, HTTP server, CLI and browser assets
experiments/b0/  Staged runner, holdout audit and experiment reports
analysis/wval/   Standalone wastewater audit and evidence
scripts/        Small launchers and CSV utility
docs/workflows/ Current commands
docs/results/   Completed experiment findings
docs/design/    Research proposals
tests/          Unit and local integration tests
```

Runtime data, Git mirrors, staging partitions, and the disposable SQLite index
are intentionally excluded from version control.

## Tests

Tests use local fixtures. A few integration checks require R/EpiBench:

```bash
python -m pip install -e '.[model,explorer,evaluation]' pytest
PYTHONPATH=src python -m pytest -q
```

Syntax-only checks used during development:

```bash
PYTHONPATH=src python -m compileall -q src scripts experiments analysis
node --check src/influpaintx/explorer/static/app.js
```

Live publisher contract tests should remain separate because schemas and row
counts change over time.

Use pytest to collect both unittest classes and pytest functions. See the
[test and feature review](maintenance.md) for focused commands and what each file protects.

## Documentation

Documentation dependencies are isolated in the `docs` extra:

```bash
.venv/bin/python -m pip install -e '.[docs]'
mkdocs serve
```

The documentation configuration is intentionally independent of the data tree;
building documentation must not require downloaded surveillance data.

## Adding a source

1. Add a complete `DatasetSpec` to `data/catalog.py`.
2. Reuse an existing fetcher or implement one under `data/sources/`.
3. Preserve native rows and record source-specific metadata.
4. Add offline tests for pagination, retries, version semantics, and failure
   atomicity as applicable.
5. Document unusual missing-value or geographic-support caveats.

Selection regression tests cover patient-count allowlists, canonical hub files,
LFS unavailability, native geography, release cutoffs, and grouped signal identity.
Change the shared policy in `data/selection.py`, not only the explorer UI.
