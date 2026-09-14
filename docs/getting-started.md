# Getting started

## Requirements

- Python 3.11 or newer
- Git for Hubverse sources
- A Delphi API key for archive-scale Delphi downloads

Delphi acquisition requires `epidatpy` 0.6.x (installed with this package).
PyArrow is an optional explorer dependency for Parquet tables.

## Install from a checkout

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[model,explorer,evaluation]'
```

The checkout scripts use the same installed dependencies.

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
python scripts/pull_covariates.py --data-root data init
python scripts/pull_covariates.py --data-root data pull cdc_nhsn_final cdc_nssp_trajectories
```

These two CDC sources supply the current training dataset. Broader acquisition
is optional. A selective Delphi example, when researching revisions:

```bash
python scripts/pull_covariates.py --data-root data pull delphi_nhsn \
  --mode snapshot --signal confirmed_admissions_flu_ew --geo-type nation
```

## Open the explorer

Build the index when necessary and start the server:

```bash
python scripts/explore_covariates.py --data-root data serve
```

Open an existing index without checking raw data or rebuilding:

```bash
python scripts/explore_covariates.py --data-root data serve --no-index
```

See the [command-line reference](reference/cli.md) for selective downloads,
resume behavior, historical Hub commits, and verification commands.

## Build the dataset and run B0

```bash
python -m tapestry.model_data build --data-root data
python -m tapestry.model_data inspect
python -m tapestry.models.season_cv --output data/experiments/my_b0_cv
```

See the [dataset contract](data/build-b-finalized.md),
[training/prediction commands](workflows/training.md), and
[hub evaluation](workflows/hub-evaluation.md). Hub evaluation additionally requires
R with `scoringutils`; configuration diagnostics use the sibling EpiBench checkout.
The model is a finalized retrospective pilot; it does not reconstruct real-time vintages.
