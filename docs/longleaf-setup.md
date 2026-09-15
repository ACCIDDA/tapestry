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
does not define `module`, initialize it with
`source /usr/share/lmod/lmod/init/bash`. Do not source `/etc/profile.d/modules.sh`:
it selects the legacy Tcl module loader, which cannot load Longleaf's Lua R module.
Use a Slurm allocation for training and substantial evaluation runs; this setup
does not request a GPU or submit a training job.

## Build the frozen inputs

Run from `/proj/jlessler/projects/tapestry-all/tapestry`. Experiment `b0-rebuilt`
uses the essential suite: 14 configurations × three seeds = 42 CV runs and
126 season fits. The rebuilt CDC data define a new frozen experiment; historical
observations may differ from the original September 4 snapshots. These are
exploratory finalized-data comparisons, not prospective validation.

The existing editable installation is sufficient. Skip acquisition above when
both CDC sources are already downloaded, then build the tensor:

```bash
.venv/bin/python -m tapestry.model_data build \
  --data-root data --start 2023-09-01 --end 2026-08-29 \
  --output data/processed/build_b_finalized.npz
```

The adjacent JSON records input provenance. The current build uses NHSN snapshot
`20260914T201316.612613Z` and NSSP snapshot `20260914T201420.344071Z`, with shape
`[157, 6, 2, 52]` and weekly dates September 2, 2023–August 29, 2026.

Restore the pinned hub commits and population table before registering the
experiment:

```bash
(
set -euo pipefail
mkdir -p data/mirrors data/metadata
while read -r hub repository revision; do
  mirror="data/mirrors/hub_${hub}_current.git"
  if [[ ! -d "$mirror" ]]; then
    git init --bare "$mirror"
  fi
  git --git-dir="$mirror" fetch --depth=1 \
    "https://github.com/${repository}.git" "$revision"
  git --git-dir="$mirror" update-ref --no-deref HEAD "$revision"
done <<'HUBS'
flusight cdcepi/FluSight-forecast-hub db3d8a9b8f022404affe56166c63a9d224a11861
covid CDCgov/covid19-forecast-hub df7965dd1d4832a96fe6712f9de9f000abc01f3e
rsv CDCgov/rsv-forecast-hub 53baf7db03a2429f5a3710d859319b9029df76cd
HUBS
git --git-dir=data/mirrors/hub_flusight_current.git \
  show HEAD:auxiliary-data/locations.csv > data/metadata/b0_locations.csv
printf '%s  %s\n' \
  80aaa24750044a837e063812a1d0a22339c8b3a92513f71a23dac613eaaa29e0 \
  data/metadata/b0_locations.csv | sha256sum --check
)
```

Keep these inputs fixed after registration. Changed data or model code require
a new experiment name.

## Prepare evaluation support

```bash
mkdir -p output/slurm
sbatch scripts/b0_prepare.sbatch
```

This one-GPU job loads R, runs three one-epoch bootstrap fits with eight evaluation
members, builds `data/evaluation/b0_hub_comparison_rebuilt`, and registers the
full suite with `--device cuda`. The bootstrap fits supply evaluation dates and
locations; they are separate from the 126 full training fits. Completed support
is reused. If support construction fails, inspect and archive its incomplete
output directory before resubmitting.

Wait for preparation to complete before distributing training.

## Train across six GPUs

Partition `jlessler` provides:

| Node | CPU cores | Host RAM | GPUs |
|---|---:|---:|---|
| `g1803jles01` | 56 | 500,000 MB | 4 × L40, 48 GB each |
| `g1803jles02` | 64 | 2,048,000 MB | 2 × H100 NVL, 96 GB each |

Each array task requests one GPU, four CPUs, and 64 GiB RAM. Six concurrent tasks
can use all six GPUs. Memory and time limits are resource allowances, not measured
requirements. Mixed GPU hardware may introduce small numerical differences;
each task records its allocation and logs the device model.

Prepare independent manager registries once, with no active manager writing the
experiment:

```bash
.venv/bin/python scripts/distribute_b0.py prepare
```

The helper preserves completed runs and reports the array range for unfinished
runs. Use its reported range; a fresh 42-run experiment reports `0-41%6`:

```bash
sbatch --gres=gpu:1 --array=0-41%6 scripts/b0_distributed.sbatch
# Replace ARRAY_ID with the returned job ID.
sbatch --dependency=afterany:ARRAY_ID --job-name=b0-collect \
  scripts/b0_distributed.sbatch collect
# Replace COLLECTOR_ID with the returned collector ID.
sbatch --dependency=afterok:COLLECTOR_ID scripts/b0_evaluation_parallel.sbatch
```

The collector merges the independent registries into `b0-rebuilt/runs.json`
and fails if any runs are incomplete. Scoring starts only after collection
succeeds. Source files under `src/tapestry` retain their registered hashes;
cluster orchestration lives in `scripts/`.

## Monitor and resume

```bash
squeue -u "$USER"
.venv/bin/python scripts/distribute_b0.py status
sacct -j ARRAY_ID --format=JobID,State,Elapsed,ExitCode,NodeList
tail -f output/slurm/b0-distributed-ARRAY_ID_TASK_ID.log
```

Live worker status is in
`data/experiments/b0-rebuilt/distributed/<task>/b0-rebuilt/runs.json`.
Each record points to its attempt log. The central registry is updated during
collection. `distributed/tasks.json` maps array indices to scenario/seed pairs.

After the array and collector have ended, inspect failures and retry only the
affected indices, for example:

```bash
sbatch --gres=gpu:1 --array=2,9%6 scripts/b0_distributed.sbatch
sbatch --dependency=afterany:RETRY_ARRAY_ID --job-name=b0-collect \
  scripts/b0_distributed.sbatch collect
sbatch --dependency=afterok:COLLECTOR_ID scripts/b0_evaluation_parallel.sbatch
```

Reuse the existing shards rather than rerunning `prepare`. The manager preserves
attempt history and restarts all three folds for interrupted or failed runs.
Avoid overlapping workers for the same task or concurrent collectors.

## Score and review

To score an already collected experiment:

```bash
sbatch scripts/b0_evaluation_parallel.sbatch
tail -F output/slurm/b0-eval-parallel-JOB_ID.log
```

The scoring job requests 36 CPU cores and 256 GiB RAM on `g1803jles01`. It loads
saved predictions concurrently and runs nine target/season cases in parallel,
with four numerical-library threads per worker. Existing Hubverse exports are
assumed complete and reused; a new comparison writes them once. EpiBench reuses
completed scores when their provenance matches. Partial scoring directories
are archived before retrying. Rankings and plotting follow scoring.

`comparison.json` records status and the output directory. The execution record
includes the launcher checksum and Slurm allocation. On failure, inspect the
job log and resubmit the scoring script after the previous job has ended.

Review `REPORT.md`, `leaderboard.csv`, `run_ranking.csv`,
`configuration_ranking.csv`, and `scores.parquet`. Report WIS, bias, and 50%/95%
coverage by pathogen, horizon, season, and states/DC versus native US, including
means and variability across three seeds. Use the matched controls in
[the manager workflow](workflows/experiment-manager.md) to assess separate heads,
decoder uncertainty, and candidates for combination.
