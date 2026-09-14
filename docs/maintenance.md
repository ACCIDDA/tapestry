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
- B0 experiment recipes, audits, and reports moved to `experiments/b0/`.
  `models/experiments.py` remains the small reusable model-option helper, not a runner.
  Existing results stay under `data/experiments/`.
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

You do not need every test for every edit. Keeping the suite is currently cheap:
the pre-refactor local baseline was **86 passing tests in 8.85 seconds**, including
available R/EpiBench integration checks. The following counts describe that baseline;
parameterized model tests count as separate cases.

| Files | Cases | What they catch | Priority |
|---|---:|---|---|
| `test_model_data.py`, `test_season_cv.py` | 6 | Window/target alignment, masks, season boundaries, save/load, held-out values changing training inputs/scales, independently checked WIS | Essential for research validity |
| `test_b0.py` | 9 | Fair CRPS mathematics, missing-label exclusion, gradients, stochastic output, population-transform inversion, location ordering, feature masking | Essential while B0 is used |
| `test_hub_evaluation.py`, `test_evaluation_sweep.py` | 9 | Correct horizon/channel export, quantile validity, equal scoring support, stable configuration identities, agreement with R scoring, end-to-end ranking/plots | Essential for reported comparisons |
| `test_selection.py` | 17 | Native geography, conflict/retraction handling, source labels, release cutoffs, canonical hub observations and provider identity | Keep while building from raw sources |
| `test_repository.py`, `test_socrata.py`, `test_delphi.py`, `test_hubverse.py` | 17 | Immutable commits, failed pulls, complete pagination, partial-stream retries, resume matching, API parameters, read-only Git history | Keep for supported downloaders |
| `test_explorer.py`, `test_explorer_versions.py` | 23 | HTTP queries, indexing, revision cutoffs, suppression, SQLite/Parquet consistency, interrupted and concurrent rebuilds | Optional only if explorer features are retired |
| `test_catalog.py`, `test_geography.py` | 5 | Catalog metadata/revision semantics and HHS membership/broadcast semantics | Small; exact catalog-count assertions and unused HHS coverage are the first candidates to reconsider |

The catalog's hard-coded count of 25 is an inventory assertion, not a scientific
invariant. It can create busywork when intentionally changing the source list.
The two HHS tests were removed with their helper API in the follow-up cleanup. Most remaining tests
check externally meaningful behavior rather than just repeating code structure.

The holdout audit in `experiments/b0/` overlaps unit tests intentionally: unit tests
use small synthetic examples; the audit checks the actual canonical dataset and
saved fitted-fold artifacts. Run that audit for an experiment being reported,
not after every small edit. Existing report prose contains historical counts;
those are observations at the time, not the current suite size.

### Focused commands

Run from the repository root after installing model/explorer/evaluation dependencies
and pytest. Pytest also runs unittest classes; `unittest discover` omits the newer
function-style tests and should not be the documented full-suite command.

```bash
# All tests; local fixtures, no publisher downloads.
PYTHONPATH=src python -m pytest -q

# Model/data changes: scientific correctness and leakage.
PYTHONPATH=src python -m pytest -q tests/test_model_data.py tests/test_b0.py tests/test_season_cv.py

# Explorer changes.
PYTHONPATH=src python -m pytest -q tests/test_explorer.py tests/test_explorer_versions.py tests/test_selection.py

# Scoring/export changes.
PYTHONPATH=src python -m pytest -q tests/test_hub_evaluation.py tests/test_evaluation_sweep.py
```

Three existing integration tests invoke R scoring and/or the sibling EpiBench
checkout. They skip if Rscript or the required checkout is absent; an installed
Rscript without required R packages still fails and needs its dependencies fixed.
Tests use temporary fixtures and do not train the full research sweep.

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
`influpaintX`. This is a direct rename without an old-package compatibility alias.
Existing saved datasets, checkpoints, scores, and their historical metadata remain
unchanged; stored checkpoints contain configuration and state dictionaries rather
than pickled model classes. New source hashes and new default report/model labels
reflect the rename.

To update an existing environment:

```bash
python -m pip uninstall -y influpaintx
python -m pip install -e '.[model,explorer,evaluation]'
```
