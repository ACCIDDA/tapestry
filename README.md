# Tapestry
influpaint v2, or maybe not too much like influpaint.  Tapestry is a research project for multi-disease epidemic forecasting. In alpha, not ready, changing.

* [Documenations (WIP)](https://accidda.github.io/tapestry/)
* [Data explorer with revisions](https://accidda.github.io/tapestry/explorer/)

## Environment

From the repository root:

```bash
uv sync
uv run python -m tapestry.experiment.planner --help
uv run pytest -q
```

`uv` manages the Python 3.11 `.venv` and Python dependencies for training,
evaluation, explorer and tests. Scoring is pure Python (`tapestry.evaluation.totals`);
no R or EpiBenchmark dependency remains. See [environment setup](docs/getting-started.md)
for lighter Python installs.

## Start with the canonical training dataset

Run from the repository root:

```bash
uv sync
# Needed only when acquiring or refreshing the training sources:
uv run python -m tapestry.data --data-root data pull cdc_nhsn_final cdc_nssp_trajectories
uv run python -m tapestry.dataset.build build --data-root data
uv run python -m tapestry.dataset.build show --dataset data/processed/finalized.npz
uv run python -m tapestry.experiment.planner plan -e my_experiment -s '' --seeds 42
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

## B1: Wednesday snapshots and masked training

B1 uses finalized older history and recent Wednesday reports, supplying flagged
reference finals where recent reports are absent. Visible known finals bypass
nowcasting and its loss; hidden finals become nowcast targets again. Four future
weeks are forecast from the resulting recent values. This is retrospective
conditional forecasting; later finals are not claimed available on Wednesday.
MLP, convolution, multiscale and spatial formulations have canonical scenario
strings, with independently configurable masking rates.
See the [B1 implementation and commands](docs/design/b1.md).

## Explore and compare

```bash
uv run python -m tapestry.explorer --data-root data serve
```

The explorer builds a disposable SQLite index and Parquet revision ledger.
See [explorer usage](docs/explorer/overview.md), [selective acquisition](docs/reference/cli.md),
[source catalog](docs/data/sources.md), [selection policy](docs/data/selection.md),
and [storage/provenance](docs/data/storage.md). Broad `--group all` downloads are
optional; they are not required to train the current six-channel model.
Delphi requires an API key; Git is needed for Hubverse sources.

Saved CV forecasts can be evaluated without refitting; hub scoring is pure
Python (`tapestry.evaluation.totals`).

## Where things live

| Directory | Purpose |
|---|---|
| `src/tapestry/data/` | Acquisition, snapshots, source readers, geography, selection |
| `src/tapestry/dataset/` | `extract`/`build`/`splits`: the two training arrays |
| `src/tapestry/model/` | The unified `Model` network, `Scenario` codec, objective |
| `src/tapestry/experiment/` | `planner.py` plan/run/status/rank, `dispatch.py`, `provenance.py` |
| `src/tapestry/evaluation/` | Shared scoring, hub comparison, exports |
| `src/tapestry/explorer/` | `index.py`, `server.py`, `cli.py`, browser assets |
| `analysis/wval/` | Standalone wastewater analysis and evidence |
| `scripts/` | Small checkout launchers and CSV conversion utility |
| `docs/workflows/` | Current commands and behavior |
| `docs/results/` | Completed experiment findings |
| `docs/design/` | Research proposals |
| `tests/` | Checks that protect reported results: leakage, masks, scoring, export |

Downloaded data, checkpoints, and generated results live in `data/`, `output/`,
and `tmp/`. `tapestry.experiment.planner` plans, runs, and scores experiments.
See [development](docs/development.md) and [features and tests](docs/maintenance.md).

## Tests

```bash
uv sync
uv run pytest -q
```

The suite only keeps tests that protect reported results. GitHub Actions runs it
on pushes and pull requests to `main`. Tests use fixtures and local temporary files.
See [what the tests do](docs/maintenance.md#what-the-tests-do).

## Files kept locally

Git excludes surveillance data, processed panels, checkpoints, generated output
folders, downloaded analysis evidence, reference PDFs/extracted text, and credentials.
The source catalog is `src/tapestry/data/catalog.py`. Download/build the data
separately using the commands above.
The supported dataset command is `python -m tapestry.dataset.build build`.
