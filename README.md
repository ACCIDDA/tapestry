# Tapestry

Tapestry is a research project for multi-disease epidemic forecasting. The Python
package and commands retain `influpaintx`; the checkout is named `influpaintX`.

The working pipeline downloads surveillance data, builds a canonical six-channel
weekly dataset, trains the stochastic B0 model, runs season cross-validation,
and compares saved forecasts with hub models. A local explorer lets you inspect
source series and publisher revisions.

```text
CDC / Delphi / Hubverse
          ↓
data/ raw snapshots → shared selection ──→ explorer
                              ↓
                         model_data/
                    weekly values + masks
                              ↓
                           models/
                   B0 → training / season CV
                              ↓
                         evaluation/
                 frozen tasks → scores / reports
```

## Start with the canonical training dataset

Run from the repository root:

```bash
python -m pip install -e '.[model,explorer,evaluation]'
# Needed only when acquiring or refreshing the two training sources:
python -m influpaintx.data --data-root data pull cdc_nhsn_final cdc_nssp_trajectories
python -m influpaintx.model_data build --data-root data \
  --output data/processed/build_b_finalized.npz
python -m influpaintx.model_data inspect
python -m influpaintx.models.season_cv --output data/experiments/my_b0_cv
```

The dataset contains weekly NHSN admissions and NSSP ED proportions for
flu/COVID/RSV, with values and availability masks for 50 states, DC, and native US.
Training reads the saved dataset; it does not need to download data or run the explorer.
Use a new output directory for a new experiment. See the
[dataset contract](docs/data/build-b-finalized.md) and
[training/prediction guide](docs/workflows/training.md).

This is finalized retrospective research: NSSP's latest saved values are assumed
truth, and season CV does not recreate the observations available in real time.
Raw snapshots and source hashes are retained so results can be traced to the
exact inputs. Snapshots support reproducibility; publisher revision archives
add historical release information where the source provides it.

## Explore and compare

```bash
python -m influpaintx.explorer --data-root data serve
```

The explorer builds a disposable SQLite index and Parquet revision ledger.
See [explorer usage](docs/explorer/index.md), [selective acquisition](docs/reference/cli.md),
[source catalog](docs/data/sources.md), [selection policy](docs/data/selection.md),
and [storage/provenance](docs/data/storage.md). Broad `--group all` downloads are
optional; they are not required to train the current six-channel model.
Delphi requires an API key; Git is needed for Hubverse sources.

Saved CV forecasts can be evaluated without refitting. Hub scoring uses R
`scoringutils`; configuration plots also use the sibling EpiBench checkout.
Follow [hub comparison](docs/workflows/hub-evaluation.md), then
[configuration comparison](docs/workflows/configuration-evaluation.md).

## Where things live

| Directory | Purpose |
|---|---|
| `src/influpaintx/data/` | Acquisition, snapshots, source readers, geography, selection |
| `src/influpaintx/model_data/` | Canonical dataset, windows, masks |
| `src/influpaintx/models/` | Model and reusable training/CV code |
| `src/influpaintx/evaluation/` | Shared scoring, hub comparison, exports and reports |
| `src/influpaintx/explorer/` | `index.py`, `server.py`, `cli.py`, browser assets |
| `experiments/b0/` | Staged B0 runner, holdout audit and experiment reports |
| `analysis/wval/` | Standalone wastewater analysis and evidence |
| `scripts/` | Small checkout launchers and CSV conversion utility |
| `docs/workflows/` | Current commands and behavior |
| `docs/results/` | Completed experiment findings |
| `docs/design/` | Proposals and design history |
| `tests/` | Scientific correctness, data integrity and interface checks |

Downloaded data, checkpoints, and generated results remain in their existing
`data/`, `output/`, and `tmp/` locations. The [B0 experiment guide](experiments/b0/README.md)
explains the dated runner and reports. See [development](docs/development.md) and
[the feature and test review](docs/maintenance.md) for further simplification candidates.

## Tests

```bash
python -m pip install -e '.[model,explorer,evaluation]' pytest
PYTHONPATH=src python -m pytest -q
```

Use pytest: it runs both the original unittest classes and newer pytest functions.
The old `unittest discover` command omits the latter. Tests use fixtures and
local temporary files; three integration tests exercise local R scoring and/or
EpiBench and skip when their prerequisite executable/checkouts are absent.
See [which tests to run](docs/maintenance.md#what-the-tests-do).

## Files kept locally

Git excludes surveillance data, processed panels, checkpoints, generated output
folders, downloaded analysis evidence, reference PDFs/extracted text, and credentials.
The source catalog remains in `src/influpaintx/data/catalog.py`; all acquisition
sources are retained. Download/build the data separately using the commands above.
The supported dataset command is `python -m influpaintx.model_data build`.
