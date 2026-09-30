# Development

## Repository layout

| Directory | Responsibility |
|---|---|
| `src/tapestry/data` | Acquisition, snapshots, readers and shared selection |
| `src/tapestry/dataset` | Panel construction, episodes and scientific partitions |
| `src/tapestry/model` | Network, scenarios, objective and sampled-history handoff |
| `src/tapestry/experiment` | Planner, independent/two-stage fitting, shared GPU dispatch |
| `src/tapestry/evaluation` | Canonical WIS scoring, matched effects, plots and dated reports |
| `src/tapestry/explorer` | Source indexing, API and interactive browser assets |
| `experiments` | Scenario-grid definitions that call the shared planner |
| `scripts` | Cluster launch, notifications, explorer publishing and documentation navigation |
| `docs/experiments` | Completed experiment reports and their evidence |

## Research iteration

Use runs and inspection for routine iteration. Tests are reserved for plausible
silent scientific errors: loss/score mathematics, weighting, leakage and data
units/alignment. No broad regression suite or plot tests are required. When those
calculations change, run the relevant focused checks in `tests/`.

Experiment analysis belongs in `tapestry.evaluation`, called by `planner rank`.
Use saved canonical ranking tables for written interpretation; keep the prose
inside report write-up markers. Avoid another experiment-specific scoring or
report-generation script. Dataset diagnostics use `tapestry.dataset.analyze_dataset`.

## Documentation

```bash
uv run --no-dev --extra docs mkdocs serve
uv run --no-dev --extra docs mkdocs build --strict
```

The site builds without training data or model execution. Each experiment has
`report.json` with an ISO report date, a title and `stage` (`nowcast` or `forecast`).
The reporting module writes the model-runs index, and `scripts/docs_hooks.py`
orders the sidebar by the same dates. Sidebar subsections are expanded.

The availability staircase is embedded in a docs page. Its standalone interactive
view includes a return link. The general source explorer is published separately
at `/explorer/`, using committed exports under `docs/explorer/data/` and the UI in
`src/tapestry/explorer/static/`. Refresh that export with
`scripts/update_published_explorer.sh`; the docs build does not download sources.

## Publishing

The Documentation GitHub Action builds with `--strict` on pull requests and main.
On main it publishes the site through GitHub Pages. Repository Pages settings use
GitHub Actions as the publishing source. Model training and source credentials
are not part of the documentation build.
