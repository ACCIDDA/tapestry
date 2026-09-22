# Getting started

## Requirements

- `uv` (the checkout defaults to Python 3.11)
- Git for Hubverse sources
- A Delphi API key for archive-scale Delphi downloads

Delphi acquisition requires `epidatpy` 0.6.x (installed with this package).
PyArrow is an optional explorer dependency for Parquet tables.

## Install from a checkout

For the cluster module and workspace commands used on Longleaf, see
[Longleaf setup](longleaf-setup.md).

On a fresh Mac with Homebrew, install the tools once:

```bash
brew install uv git
```

On other systems, install uv and Git using their supported installers. Run the
following from the Tapestry repository root:

```bash
uv sync
uv run python -m tapestry.experiment.planner --help
```

The sync command creates `.venv` and installs Tapestry in editable mode; no
sibling checkout or non-Python dependency is required. The default `dev` group
includes training, evaluation, exploration and pytest; package extras are
available for smaller installs. Use `uv run` before commands to use this
environment without shell activation or `PYTHONPATH`. Alternatively, after
syncing, `source .venv/bin/activate` enables the plain `python` commands
without the `uv run` prefix.

`pyproject.toml` declares dependencies and `.python-version` selects Python 3.11.

For a smaller environment or documentation:

```bash
uv run --no-dev --extra model python -m tapestry.experiment.planner --help
uv run --no-dev --extra docs mkdocs serve
```

Keep those flags on subsequent `uv run` commands for that profile; a plain
`uv run` restores the default research dependencies. Standard
`python -m pip install -e '.[model]'` is also supported.
Do not create the environment with `--system-site-packages`.

Scoring is pure Python (`tapestry.evaluation.totals`); there is no R or
EpiBenchmark dependency to install.

## Configure credentials

```bash
export DELPHI_EPIDATA_KEY="your-key"
# Or load the local top-level key file without printing it:
# export DELPHI_EPIDATA_KEY="$(cat DELPHI_API_KEY)"
# Optional: improves CDC Socrata rate limits.
export CDC_APP_TOKEN="your-token"
```

`DELPHI_API_KEY` is also accepted as an environment alias; the canonical variable
takes precedence. Credentials are sent in request headers and are not written to source URLs,
manifests, or catalog files.

## Initialize and download

```bash
uv run python scripts/pull_covariates.py --data-root data init
uv run python scripts/pull_covariates.py --data-root data pull cdc_nhsn_final cdc_nssp_trajectories
```

These two CDC sources supply the current training dataset. Broader acquisition
is optional. A selective Delphi example, when researching revisions:

```bash
uv run python scripts/pull_covariates.py --data-root data pull delphi_nhsn \
  --mode snapshot --signal confirmed_admissions_flu_ew --geo-type nation
```

## Open the explorer

Build the index when necessary and start the server:

```bash
uv run python scripts/explore_covariates.py --data-root data serve
```

Open an existing index without checking raw data or rebuilding:

```bash
uv run python scripts/explore_covariates.py --data-root data serve --no-index
```

See the [command-line reference](reference/cli.md) for selective downloads,
resume behavior, historical Hub commits, and verification commands.

## Build the dataset and run the unified model

```bash
uv run python -m tapestry.dataset.build build --data-root data
uv run python -m tapestry.dataset.build show   # data/processed/panel.npz
uv run python -m tapestry.experiment.planner plan -e my_experiment -s '' --seeds 42
```

See [training/prediction commands](workflows/training.md). Scoring is
pure Python (`tapestry.evaluation.totals`); no R or EpiBenchmark dependency remains.
