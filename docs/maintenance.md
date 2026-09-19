# Features and tests

## Which data features are optional?

`model_data/finalized.py` builds the six-channel panel from only `cdc_nhsn_final`
and `cdc_nssp_trajectories`. `models/run.py` and `models/season_cv.py` load the
saved NPZ. Neither requires the explorer, Delphi, or historical raw snapshots
while fitting.

A canonical training artifact and raw snapshots serve different purposes.
The artifact gives every experiment the same tensor/masks; raw snapshots and
hashes explain where it came from and allow you to rebuild it. Publisher revision
archives can additionally support comparisons of preliminary versus later values.
A retrieval snapshot alone does not reconstruct revisions the publisher never saved.
The model is a finalized retrospective experiment even when raw revision archives
are present.

| Candidate | Observed use | Recommendation |
|---|---|---|
| All-source catalog and broad downloads | 25 catalog entries versus two sources used by this model; other sources support comparison and exploration | Research instructions default to the two needed sources. Retain adapters until those research questions are retired. Most catalog code is declarations. |
| Explorer schema discovery, filters, revision ledger and as-of queries | Useful for multi-source inspection; absent from the training path | Optional application. Skip building/running it for model-only work. Replacing it with a panel viewer would sacrifice source/revision comparison. |
| Snapshot checksums, staging, atomic publication | Prevent incomplete downloads becoming training inputs and record exact provenance | Keep. These protect reproducibility even with one canonical training dataset. |
| Delphi retries, resumable partitions and concurrency | Needed for large archive pulls; irrelevant to a two-CDC-source-only workflow | Keep while Delphi archives are supported; avoiding those downloads removes runtime cost without changing the adapter. |
| Source lineage and native geography filtering | Shared selection uses these to avoid false equivalence between measures, providers and geographic support | Keep. A state label on a county/site/HHS record does not make it a state observation. |
| PDF reports, EpiBench diagnostics and CSV companions | Evaluation presentation/interoperability rather than model fitting | Optional outputs. PDF is separate; separating EpiBench plotting from `sweep` scoring would be a useful simplification. |

These are source-based recommendations, not proof that no external notebook uses
an API.

## What the tests do

Keep tests only for plausible silent errors that would materially corrupt a
scientific result. There is no coverage target or requirement for one test per
module. Routine failures, CLI parsing, naming, plotting, artifact creation,
resume mechanics and architecture execution are checked through research runs.

| File | Consequence it guards against |
|---|---|
| `test_b0.py` | Wrong CRPS or masked-label gradients; count/proportion transforms changing native values |
| `test_objective.py` | Missingness or batching changing scientific loss weights; incorrect native loss scales |
| `test_season_cv.py` | Held-out or validation values leaking into training/scales; incorrect WIS |
| `test_model_data.py` | Misaligned targets, observed zeros treated as missing, wrong season assignment or percentage units |
| `test_totals.py` | Incorrect WIS, location/season/target/seed weighting, undefined relative scores or comparisons on different locations |
| `test_hub_evaluation.py` | Forecasts assigned to the wrong target channel, location or week |

The retained checks use local fixtures and do not invoke R or EpiBench.
When a relevant scientific calculation changes, a focused check can be run with
`uv run pytest -q tests/test_objective.py` (substitute the relevant file).
GitHub Actions runs the remaining suite on pushes and pull requests to `main`.

## Test cull decision

Assumption: this research code is frequently rewritten, and full reruns plus
inspection handle ordinary execution and presentation failures. Tests are
reserved for consequential mistakes that can survive a successful run and
produce plausible but wrong results.

The initial cull removed broad model/report execution checks and fixed snapshots.
The stricter cull removed the experiment-manager, evaluation-sweep and EpiBench
adapter test files; model architecture/scaler/dynamics checks; early-stopping
execution; export round trips; and routine malformed-input assertions. The
remaining suite has 15 test functions in 355 lines, down from 1,238 lines before
the cull. No replacement tests were added to preserve coverage. CI no longer
installs R for deleted integration tests.

Validation: Python syntax parsing and whitespace review only; tests and research
runs were not executed. This does not certify remaining tests against concurrent
model changes.

## Nowcasting publication log — 2026-09-18

Every Markdown page under `docs/` is included in the site menu, including older
B1 experiment notes and the archived architecture note. Research notes under
`analysis/` remain repository documents, outside the published site.

Assumption: the detailed revision-season `cells.csv` and preliminary 300-epoch
CSV exports are regenerable local analysis outputs, not required website assets.
They are ignored along with local audit logs and status snapshots. Published
result CSVs, plotting inputs, compact seasonal summaries, provenance, scripts,
and research notes are retained. Ignored outputs remain on disk.
