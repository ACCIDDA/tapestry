# Features and tests

## Which data features are optional?

`tapestry.dataset.build` builds the one training array, `data/processed/panel.npz`,
from the Hub target data, Delphi NHSN/NSSP/claims archives, the derived NWSS
indices and PopHIVE Kinsa (see [training](workflows/training.md)).
`tapestry.experiment.planner` loads only that saved NPZ while fitting; the explorer
and raw snapshots are not read during training.

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
| CSV companions for Hubverse archives (`export_hubverse_csv.py`) | Interoperability rather than model fitting | Optional output. PDF reports, EpiBench diagnostics, and `sweep` scoring were removed with the R/EpiBench path (2026-09 restructuring). |

These are source-based recommendations, not proof that no external notebook uses
an API.

## What the tests do

Keep tests only for plausible silent errors that would materially corrupt a
scientific result. There is no coverage target or requirement for one test per
module. Routine failures, CLI parsing, naming, plotting, artifact creation,
resume mechanics and architecture execution are checked through research runs.

| File | Consequence it guards against |
|---|---|
| `test_network.py` | Wrong CRPS or masked-label gradients; count/proportion transforms changing native values |
| `test_objective.py` | Missingness or batching changing scientific loss weights; incorrect native loss scales |
| `test_dataset.py` | Held-out-season or validation-week values reaching training inputs, labels, loss scales/weights or covariate standardization (panel masking, `dataset.cv`); score labels outside the held-out season; vintaged episodes using as-of values at the wrong weeks or misaligned with the issuance; wrong season assignment |
| `test_planner.py` | Channel/location or covariate/location axes swapped between the panel and `Model` |
| `test_hub_history.py` | Git full-snapshot deletions resurrected, or Git overriding native Hub `as_of` coverage, in vintage resolution |
| `test_totals.py` | Incorrect WIS, location/season/target/seed weighting, undefined relative scores or comparisons on different locations |
| `test_hub_evaluation.py` | Forecasts assigned to the wrong target channel, location or week |

The retained checks use local fixtures; there is no R or EpiBench dependency
left to invoke (removed in the 2026-09 restructuring, see
[docs/design/restructure-2026-unified.md](design/restructure-2026-unified.md)).
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
