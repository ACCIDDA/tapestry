# Code, features, and tests review

Reviewed September 14, 2026. User requirements: separate explorer responsibilities,
group B0 experiments, share repeated scoring code, update the documentation, move
the wastewater analysis, and retain snapshots. Review potentially unnecessary
features and explain the tests.

The initial organization pass preserved behavior. The subsequent user-approved
cleanup removes HHS parent-context helpers and their two tests, plus the duplicate
builder CLI; all acquisition sources and snapshots remain supported. Historical
forecasts, scores, source snapshots, and experiment provenance are not regenerated.

## What changed

- Explorer code now lives in `explorer/index.py`, `server.py`, and `cli.py`.
  The seven-line initializer retains the public `ExplorerIndex`, `make_server`,
  `main`, and `normalize_state` imports. Indexing and query methods remain together
  in the existing class; this first split introduces no new storage abstraction.
- B0 experiment recipes, audits, and reports moved to `experiments/b0/`; that dated
  staged runner, audit, and report were later removed in favor of the
  [experiment manager](workflows/experiment-manager.md) (see git history).
  `models/experiments.py` remains the small reusable model-option helper, not a runner.
- `evaluation/scoring.py` owns quantile/date validation, frozen forecast matching,
  scored-task matching, geographic/horizon aggregation, ranking, and the exploratory
  objective shared by the staged runner and configuration sweep. The runner now
  also applies the sweep's exact weekly-date and finite-score checks on fresh scores.
  Cached historical summaries retain the dated runner's existing reuse behavior.
- `analysis/wval/` contains the former `VWAL analysis/`; its root-relative raw-data
  lookup was adjusted. Evidence files remain alongside the analysis.
- Current commands live under `docs/workflows/`, completed B0 reports under
  `docs/results/`, and proposals under `docs/design/`. README and navigation now
  describe the implemented dataset/model pipeline. Generated report destinations
  and internal links follow the new locations.

## Which data features are optional?

Observed fact: `model_data/finalized.py` builds the current six-channel panel
from only `cdc_nhsn_final` and `cdc_nssp_trajectories`. `models/run.py` and
`models/season_cv.py` load the saved NPZ. Neither requires the explorer, Delphi,
or historical raw snapshots while fitting.

A canonical training artifact and raw snapshots serve different purposes.
The artifact gives every experiment the same tensor/masks; raw snapshots and
hashes explain where it came from and allow you to rebuild it. Publisher revision
archives can additionally support comparisons of preliminary versus later values.
A retrieval snapshot alone does not reconstruct revisions the publisher never saved.
The current model remains a finalized retrospective experiment even when raw
revision archives are present.

| Candidate | Observed use | Recommendation |
|---|---|---|
| HHS membership and parent-broadcast helpers in `data/geography.py` | Referenced by their two tests, but no other current source-code callers; current selection excludes HHS observations | Removed in the follow-up cleanup, along with the membership CSV and two helper tests. State normalization and native-support checks remain. |
| Second builder CLI at the bottom of `model_data/finalized.py` | Repeats the build/save command in `model_data/cli.py`; the supported package command uses the latter | Removed in the follow-up cleanup. Use `python -m tapestry.model_data build` or `tapestry-model-data build`; both use `model_data/cli.py`. |
| All-source catalog and broad downloads | 25 catalog entries versus two sources used by this model; other sources support comparison and exploration | Default research instructions to the two needed sources, as now documented. Retain adapters until those research questions are retired. Most catalog code is declarations. |
| Explorer schema discovery, filters, revision ledger and as-of queries | Useful for multi-source inspection; absent from the training path | Optional application. Skip building/running it for model-only work. Replacing it with a panel viewer would sacrifice source/revision comparison. |
| Snapshot checksums, staging, atomic publication | Prevent incomplete downloads becoming training inputs and record exact provenance | Keep. These protect reproducibility even with one canonical training dataset. |
| Delphi retries, resumable partitions and concurrency | Needed for large archive pulls; irrelevant to a two-CDC-source-only workflow | Keep while Delphi archives are supported; avoiding those downloads removes runtime cost without changing the adapter. |
| Source lineage and native geography filtering | Shared selection uses these to avoid false equivalence between measures, providers and geographic support | Keep. A state label on a county/site/HHS record does not make it a state observation. |
| PDF reports, EpiBench diagnostics and CSV companions | Evaluation presentation/interoperability rather than model fitting | Optional outputs. PDF is already separate; separating EpiBench plotting from `sweep` scoring would be a useful next simplification. |

These are source-based recommendations, not proof that no external notebook uses
an API. Only the explicitly approved HHS helpers and duplicate CLI were removed. Keep dated hard-coded experiment
recipes small; do not turn them into an experiment framework just to reduce filenames.

## What the tests do

Simplified September 14, 2026: the suite keeps only tests that protect reported
results. Downloader (`test_repository`, `test_socrata`, `test_delphi`, `test_hubverse`),
explorer, selection, catalog and experiment-manager tests were removed, along with
CLI-plumbing and file-format checks. They remain in Git history (commit
"simplify test") if those features need coverage again. GitHub Actions
(`.github/workflows/tests.yml`) runs the suite on pushes and pull requests to `main`.

| File | What it catches |
|---|---|
| `test_model_data.py` | Window/target alignment, masks, observed zeros versus missing values, season boundaries, builder units and conflicting rows |
| `test_season_cv.py` | Held-out season values cannot change training inputs, labels or scales; WIS; five-quantile selection from saved archives |
| `test_b0.py` | Fair CRPS mathematics, missing-label exclusion, masked inputs not changing predictions, location ordering, population-transform inversion, checkpoint round trip, separate state/US head gradients |
| `test_hub_evaluation.py` | Invalid quantile tasks excluded, agreement with R `scoringutils`, equal scoring support for best-model selection, horizon/channel/FIPS export, ranking on identical tasks |
| `test_evaluation_sweep.py` | Seed averaging and target weighting in configuration ranking, stable configuration identities, Hubverse round trip, frozen-task and truth matching, equal-geography objective, end-to-end R/EpiBench sweep |
| `test_experiment_manager.py` | Short scenario strings round-trip and reject typos, CLI flags reproduce scenarios, one-factor suites, plan/run/resume/status with relative paths and commits |
| `test_epibench_pipeline.py` | EpiBench scoring with numeric FIPS and a zero reference, WIS values, refusal to reuse scores for changed inputs or missing tasks |

The former holdout audit in `experiments/b0/` checked the actual dataset and the
dated staged run's fitted-fold artifacts; it was removed with that runner and
remains in git history. Existing report prose contains historical counts;
those are observations at the time, not the current suite size.

### Focused commands

Run from the repository root after `uv sync`.

```bash
# All tests; local fixtures, no publisher downloads.
uv run pytest -q

# Model/data changes: scientific correctness and leakage.
uv run pytest -q tests/test_model_data.py tests/test_b0.py tests/test_season_cv.py

# Scoring/export changes.
uv run pytest -q tests/test_hub_evaluation.py tests/test_evaluation_sweep.py tests/test_epibench_pipeline.py
```

Four integration tests invoke R `scoringutils` and/or EpiBench. They skip if
Rscript or EpiBench is absent; an installed Rscript without `scoringutils` and
`purrr` still fails, so run `Rscript scripts/setup_r.R`. Tests use temporary
fixtures and do not train the full research sweep.

## Validation of this reorganization

- All **88 tests passed in 7.77 seconds**, including the available R/EpiBench checks.
  The two added regressions cover shared frozen-task matching and equal-geography
  experiment objectives with unequal numbers of state and US forecast tasks.
- Every explorer function/class body matched the original syntax tree after the split.
- Replayed the previous and new staged aggregation using saved baseline forecasts
  and scores in temporary output directories: all 90 summary rows matched to
  numerical tolerance (1e-12). No forecasts were resampled or research results overwritten.
- Python compilation and a strict MkDocs build passed. Existing pandas/PyArrow
  deprecation warnings remain; no new dependency was introduced.

The remaining explorer index class is still about 1,300 lines. Splitting its
storage/query responsibilities further is a possible later change, not a claim
that this first pass reduced all complexity or code volume.

## Initial Git repository

The initial commit contains source, tests, documentation, experiment/analysis
scripts, and bibliography files. It excludes `data/`, `output/`, `tmp/`, downloaded
analysis evidence and reference PDFs/text, environments, caches, and credentials.
These files remain local. All 25 acquisition sources and snapshot behavior remain.
The HHS helper removal reduces the suite from 88 to 86 tests; the existing builder
test now also exercises the canonical CLI's saved panel and metadata.

## Package rename

The Python distribution/import package is now `tapestry`, with installed commands
`tapestry-data`, `tapestry-model-data`, `tapestry-select`, and `tapestry-explore`.
Module commands use `python -m tapestry...`. The checkout folder stays named
`Tapestry`. This is a direct rename without an old-package compatibility alias.
Existing saved datasets, checkpoints, scores, and their historical metadata remain
unchanged; stored checkpoints contain configuration and state dictionaries rather
than pickled model classes. New source hashes and new default report/model labels
reflect the rename.

To update an existing environment:

```bash
python -m pip install -e '.[model,explorer,evaluation]'
```
