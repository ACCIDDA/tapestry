# Development

## Repository layout

```text
src/tapestry/
  data/          Acquisition, snapshots, readers and selection
  model_data/    Canonical weekly dataset and windows
  models/        B0, training and season CV
  evaluation/    Shared scoring, hub comparisons, exports and reports
  explorer/      Index, HTTP server, CLI and browser assets
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

Use actual runs and inspection for routine research iteration. Add or run tests
when they address a concrete scientific invariant or silent failure; there is no
requirement to preserve a representative test for each component. See the
[test cull decision](maintenance.md#test-cull-decision).

The retained scientific checks use local fixtures:

```bash
uv sync
uv run pytest -q
```

Syntax-only checks used during development:

```bash
uv run python -m compileall -q src scripts analysis
node --check src/tapestry/explorer/static/app.js
```

Live publisher contract tests should remain separate because schemas and row
counts change over time.

See the [test and feature review](maintenance.md) for the consequential errors
each remaining file targets.

## Documentation

Documentation dependencies are isolated in the `docs` extra:

```bash
uv run --no-dev --extra docs mkdocs serve
```

The documentation configuration is intentionally independent of the data tree;
building documentation must not require downloaded surveillance data.

## Adding a source

1. Add a complete `DatasetSpec` to `data/catalog.py`.
2. Reuse an existing fetcher or implement one under `data/sources/`.
3. Preserve native rows and record source-specific metadata.
4. Add a test only if a plausible silent error could corrupt scientific results.
5. Document unusual missing-value or geographic-support caveats.

Change shared selection policy in `data/selection.py`, not only the explorer UI.

## GitHub Pages

The `Documentation` workflow builds MkDocs with `--strict` on pull requests and
pushes to `main`; it can also be started manually. The build installs only the
package's `docs` extra and needs no local datasets, model training, or credentials.
On `main`, it uploads the generated `site/` directory as a Pages artifact and a
separate deployment job submits that artifact to GitHub Pages. Pull requests
only validate the build. Generated HTML is never committed to a publishing branch.

The same build publishes the **live explorer** at `/explorer/`: MkDocs copies the
committed export in `docs/explorer/data/`, and the workflow adds `index.html`,
`app.js`, and `style.css` from `src/tapestry/explorer/static/`. CI never rebuilds
that data; refresh it locally with `scripts/update_published_explorer.sh`, commit
`docs/explorer/data/`, and push (see
[Published explorer](explorer/overview.md#published-explorer)). The docs header
links to it via `docs/javascripts/explorer-link.js`.

Repository Settings → Pages must use **GitHub Actions** as its publishing source.
The deployment uses the `github-pages` environment, with `pages: write` and
`id-token: write` permissions scoped to its deployment job. This follows
[GitHub's custom workflow deployment](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).
