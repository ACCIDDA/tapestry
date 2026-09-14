# Longleaf setup

These commands set up the checkout at
`/proj/jlessler/projects/tapestry-all/tapestry` on Longleaf using uv-managed
Python 3.11 and the cluster's R module. Adjust the workspace paths if installing
elsewhere. Python tools and caches live in the workspace; R packages use the
personal library selected by the R module.

## Clone and install uv

Run once. If uv is already available, skip the installer commands.

```bash
cd /proj/jlessler/projects/tapestry-all
git clone https://github.com/ACCIDDA/tapestry.git

curl -LsSf https://astral.sh/uv/install.sh -o /tmp/tapestry-uv-install.sh
env UV_INSTALL_DIR=/proj/jlessler/projects/tapestry-all/.local/bin \
    UV_NO_MODIFY_PATH=1 sh /tmp/tapestry-uv-install.sh
```

## Python environment

Set these variables in each new shell that uses this installation:

```bash
export PATH=/proj/jlessler/projects/tapestry-all/.local/bin:$PATH
export UV_CACHE_DIR=/proj/jlessler/projects/tapestry-all/.cache/uv
export UV_PYTHON_INSTALL_DIR=/proj/jlessler/projects/tapestry-all/.local/share/uv/python
cd /proj/jlessler/projects/tapestry-all/tapestry
```

Install or update the default research environment:

```bash
uv sync --upgrade-package epibenchmark --python-preference only-managed
```

This creates `.venv`, installs Tapestry in editable mode, and includes training,
evaluation, explorer, and test dependencies. EpiBenchmark follows GitHub `main`;
the generated `uv.lock` remains local. Managed Python avoids inheriting the
login shell's Anaconda installation.

## Download the training sources

```bash
uv run python scripts/pull_covariates.py --data-root data init
uv run python scripts/pull_covariates.py --data-root data pull cdc_nhsn_final cdc_nssp_trajectories
```

These are the two CDC sources required by the current training dataset.
Broader acquisition is optional. The September 14, 2026 setup downloaded
21,306 NHSN rows and 657,758 NSSP rows, with three files per source. These counts
come from the downloader's completion output; no additional data audit was run.
Downloading sources does not build the processed training tensor; see
[the dataset workflow](data/build-b-finalized.md) for that next step.

## R scoring

Use Longleaf's R module, then run the repository's package setup script:

```bash
module load r/4.5.0
Rscript scripts/setup_r.R
```

The script installs missing `scoringutils` and `purrr` packages and their
dependencies into the personal R library, reuses available packages, and checks
that both scoring packages load. R is separate from the uv environment.

Check that R is also accessible through uv:

```bash
uv run Rscript -e 'library(scoringutils); library(purrr); sessionInfo()'
```

Verified on September 14, 2026: R 4.5.0, `scoringutils` 2.2.0, and `purrr`
1.0.4 loaded successfully through uv. The setup installed `scoringutils` and
`scoringRules` 1.1.3 in `~/R/x86_64-pc-linux-gnu-library/4.5`; `purrr` was already
available from the module.

Repeat `module load r/4.5.0` in new shells and Slurm job scripts that run scoring,
along with the Python environment exports above. If a noninteractive Bash shell
does not define `module`, initialize it with `source /etc/profile.d/modules.sh`.
Use a Slurm allocation for training and substantial evaluation runs; this setup
does not request a GPU or submit a training job.
