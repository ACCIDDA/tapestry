# Development

## Repository layout

| Directory | Responsibility |
|---|---|
| `src/chromantis/data` | Acquisition, snapshots, readers and shared selection |
| `src/chromantis/dataset` | Panel construction, episodes, folds, reporting-error bootstrap, ILI archive |
| `src/chromantis/model` | Network, scenarios, objective, correction trees |
| `src/chromantis/experiment` | The training route (`fit.py`), fitting loop (`training.py`), planner, shared GPU dispatch |
| `src/chromantis/evaluation` | The scorer (`standard.py`), ranking, saved-forecast ensembles, reports |
| `src/chromantis/production` | Release replay, Hub export, submission plots and comparisons |
| `src/chromantis/explorer` | Source indexing, API and interactive browser assets |
| `experiments` | Study files (candidates, seeds, protocol) and ensemble groups files |
| `production` | Releases, submitted files, Hub clones (ignored), metadata |
| `scripts` | Cluster launch, notifications, explorer publishing and documentation navigation |
| `docs/experiments` | Completed experiment reports and their evidence |

Each step has one implementation: one training route, one scorer, one ranking, one
combination rule (shared by research ensembles and the production export). Add a
candidate to a study file, not a planner script; add an analysis to
`chromantis.evaluation`, not an experiment-specific script. Replaced code is deleted
(history: [restructuring log](workflow.md#restructuring-log-8-october-2026)).

## Research iteration

Use runs and inspection for routine iteration. Tests are reserved for plausible
silent scientific errors: loss/score mathematics, weighting, leakage and data
units/alignment. No broad regression suite or plot tests are required. When those
calculations change, run the relevant focused checks in `tests/`.

Experiment analysis belongs in `chromantis.evaluation`, called by `planner rank` or `evaluation.ensembles`.
Use saved canonical ranking tables for written interpretation; keep the prose
inside report write-up markers. Avoid another experiment-specific scoring or
report-generation script. Dataset diagnostics use `chromantis.dataset.analyze_dataset`.

## Documentation

```bash
uv run --no-dev --extra docs mkdocs serve
uv run --no-dev --extra docs mkdocs build --strict
```

The site builds without training data or model execution. Each experiment has
`report.json` with an ISO report date, a title and `stage` (`nowcast` or `forecast`).
The model-runs index (`docs/experiments/index.md`) is curated by hand, one line per
phase; `scripts/docs_hooks.py` orders the sidebar by the report dates. Sidebar subsections are expanded.

The availability staircase is embedded in a docs page. Its standalone interactive
view includes a return link. The general source explorer is published separately
at `/explorer/`, using committed exports under `docs/explorer/data/` and the UI in
`src/chromantis/explorer/static/`. Refresh that export with
`scripts/update_published_explorer.sh`; the docs build does not download sources.

## Publishing

The Documentation GitHub Action builds with `--strict` on pull requests and main.
On main it publishes the site through GitHub Pages. Repository Pages settings use
GitHub Actions as the publishing source. Model training and source credentials
are not part of the documentation build.
