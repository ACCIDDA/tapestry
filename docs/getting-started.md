# Getting started

## Requirements

- `uv` (the checkout defaults to Python 3.11)
- Git for Hubverse sources
- R with `scoringutils` and `purrr` for scoring only
- A Delphi API key for archive-scale Delphi downloads

Delphi acquisition requires `epidatpy` 0.6.x (installed with this package).
PyArrow is an optional explorer dependency for Parquet tables.

## Install from a checkout

For the cluster module and workspace commands used on Longleaf, see
[Longleaf setup](longleaf-setup.md).

On a fresh Mac with Homebrew, install the tools once:

```bash
brew install uv git r
```

R is optional for training and exploration. On other systems, install uv and Git
using their supported installers and R using the system package manager or a
cluster module. Run the following from the Tapestry repository root:

```bash
uv sync --upgrade-package epibenchmark
Rscript scripts/setup_r.R  # Needed for scoring; installs missing R packages.
uv run python -m tapestry.models --help
uv run python -m epibench --help
```

The sync command creates `.venv`, installs Tapestry in editable mode, and fetches
EpiBenchmark from GitHub `main`. No sibling checkout is required.
The default `dev` group includes training, evaluation, exploration and pytest;
the existing package extras remain available for smaller installs. Use `uv run`
before commands to use this environment without shell activation or `PYTHONPATH`.
Alternatively, after syncing, `source .venv/bin/activate` enables the plain
`python` commands without the `uv run` prefix.

`pyproject.toml` declares dependencies and `.python-version` selects Python 3.11.
User preference: track upstream EpiBenchmark rather than commit a fixed revision.
Run `uv sync --upgrade-package epibenchmark` whenever setting up or updating the
environment. Plain `uv run` uses the currently installed/resolved revision; it does
**not** check GitHub for a new commit every time.

uv always records a resolved commit in its generated `uv.lock`, including for
branch dependencies. This file is deliberately Git-ignored and stays local;
there is no committed lockfile or fixed `rev` constraint. The update command
refreshes EpiBenchmark while retaining other resolved versions where compatible.
A fresh machine resolves Python dependencies anew, so environments created at
different times can differ. This is an intentional research-workflow tradeoff.
See uv's [Git upgrade behavior](https://docs.astral.sh/uv/concepts/projects/sync/#upgrading-locked-package-versions).
Evaluation records source hashes and R/package versions for the scorer actually used.

For development only, `--epibench /path/to/checkout` on the evaluation sweep uses
that checkout's scorer and plotting code instead. Its Python dependencies must
be compatible with the environment. Normal use needs no separate checkout.

For a smaller environment or documentation:

```bash
uv run --no-dev --extra model python -m tapestry.models --help
uv run --no-dev --extra docs mkdocs serve
```

Keep those flags on subsequent `uv run` commands for that profile; a plain
`uv run` restores the default research dependencies. Standard
`python -m pip install -e '.[model]'` also remains supported without EpiBenchmark.

### Migrating the old local environment

The original `.venv/pyvenv.cfg` had `include-system-site-packages = true` and used
Anaconda Python 3.11.0. It inherited global packages, so a successful import did
not imply that dependencies were installed in the project. On a checkout that
still has that old environment, stop running jobs and move it aside once:

```bash
mkdir -p tmp
mv .venv tmp/venv-before-uv  # Choose a fresh backup name if this already exists.
uv sync --upgrade-package epibenchmark --python-preference only-managed
```

This preserves the old directory and creates an isolated environment using
uv-managed Python. The backup must be moved back to `.venv` to restore it; virtual
environment entrypoints contain absolute paths. No global Anaconda packages are
changed. Do not create the new environment with `--system-site-packages`.

### R for EpiBenchmark scoring

Python launches the external `Rscript` executable and exchanges CSV files with R.
There is no `rpy2` dependency. `uv` does not install R or its CRAN packages.
EpiBenchmark does not install them either: its scoring bridge checks for
`Rscript` and reports installation instructions when R or its packages are missing.
Install R once using your system package manager (for example, `brew install r`
on macOS), or load the site's R module on a cluster. Then run:

```bash
Rscript --version
Rscript scripts/setup_r.R
uv run Rscript -e 'library(scoringutils); library(purrr); sessionInfo()'
uv run pytest -q
```

The setup script reuses available packages and installs missing packages from
CRAN into your personal R library, without an interactive prompt or administrator
permissions. It prints the R and package versions and fails if packages cannot be
loaded. Re-run it after installing a new R version. It does not update already
available packages. On systems without CRAN binaries, installing R packages from
source may require the compiler and development libraries supplied by your system.

The existing Mac installation was R 4.5.0 with scoringutils 2.1.1. The current
workflow uses the selected R installation and its libraries; it does not lock R
or CRAN versions with `renv`. Evaluation records R, scoringutils and purrr versions
in its output. Exact historical reproduction requires restoring those recorded
versions as well as Python dependencies and EpiBenchmark source. Training and
exploration do not need R. Tests needing R/EpiBenchmark skip when those prerequisites
are absent; an installed R missing its packages causes the scoring tests to fail.

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

## Build the dataset and run B0

```bash
uv run python -m tapestry.model_data build --data-root data
uv run python -m tapestry.model_data inspect
uv run python -m tapestry.models.season_cv --output data/experiments/my_b0_cv
```

See the [dataset contract](data/build-b-finalized.md),
[training/prediction commands](workflows/training.md), and
[hub evaluation](workflows/hub-evaluation.md). Hub evaluation additionally requires
R with `scoringutils` and `purrr`; configuration diagnostics use installed EpiBenchmark.
The model is a finalized retrospective pilot; it does not reconstruct real-time vintages.
