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
Broader acquisition is optional. Downloading sources does not build the processed training tensor; see
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

Repeat `module load r/4.5.0` in new shells and Slurm job scripts that run scoring,
along with the Python environment exports above. If a noninteractive Bash shell
does not define `module`, initialize it with
`source /usr/share/lmod/lmod/init/bash`. Do not source `/etc/profile.d/modules.sh`:
it selects the legacy Tcl module loader, which cannot load Longleaf's Lua R module.
Use a Slurm allocation for training and substantial evaluation runs; this setup
does not request a GPU or submit a training job.

## Build the frozen inputs

Run from `/proj/jlessler/projects/tapestry-all/tapestry`. The architecture sweep
has 4,097 configurations × three seeds = 12,291 CV runs and 36,873 season fits;
the essential suite has 14 configurations × three seeds = 42 CV runs. These are
exploratory finalized-data comparisons, not prospective validation.

Skip acquisition above when
both CDC sources are already downloaded, then build the tensor:

```bash
.venv/bin/python -m tapestry.model_data build \
  --data-root data --start 2023-09-01 --end 2026-08-29 \
  --output data/processed/build_b_finalized.npz
```

The adjacent JSON records input provenance, including the NHSN and NSSP snapshot
IDs. The tensor has shape `[157, 6, 2, 52]` with weekly dates September 2,
2023–August 29, 2026.

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

Keep these inputs fixed while an experiment runs. Experiments are not locked to a
code version: each attempt records its git commit and whether the checkout had
uncommitted changes, and results for the paper are rerun from a clean tree.

## Prepare evaluation support

```bash
mkdir -p output/slurm
sbatch scripts/b0_prepare.sbatch b0-sweep grid
```

This one-GPU job loads R, runs three one-epoch bootstrap fits with eight evaluation
members, builds `data/evaluation/b0_hub_comparison_q23` on the hub's 23-quantile
grid, and plans the named experiment (default `b0`) and suite (default `essential`)
with `--device cuda`. The bootstrap fits supply evaluation dates and locations; they
are separate from the experiment's fits. Completed support is reused. If support
construction fails, inspect and archive its incomplete output directory before
resubmitting. Support built earlier for five quantiles cannot score new runs, and
`manager run` refuses to start with it.

Wait for preparation to complete before distributing training.

## Run the architecture sweep on the shared GPU partitions

The sweep has 4,097 array tasks, one per configuration; each runs three seeds of
three season folds and writes `totals.csv` for every seed. `scripts/b0_sweep.sbatch`
requests one GPU, four CPUs, 16 GiB, and six hours on `a100-gpu,l40-gpu,jlessler`
with QOS `gpu_access`, like InfluPaint's inpainting arrays. These limits are
allowances; runtime has not been measured.

```bash
.venv/bin/python -m tapestry.models.manager status -e b0-sweep | tail -8
```

`status` prints one submission per chunk of 1,000 tasks, adding `OFFSET` so array
indices stay small. For a fresh sweep:

```bash
sbatch --array=0-999 --export=ALL,OFFSET=0 scripts/b0_sweep.sbatch b0-sweep
sbatch --array=0-999 --export=ALL,OFFSET=1000 scripts/b0_sweep.sbatch b0-sweep
sbatch --array=0-999 --export=ALL,OFFSET=2000 scripts/b0_sweep.sbatch b0-sweep
sbatch --array=0-999 --export=ALL,OFFSET=3000 scripts/b0_sweep.sbatch b0-sweep
sbatch --array=0-96 --export=ALL,OFFSET=4000 scripts/b0_sweep.sbatch b0-sweep
```

Logs are `output/slurm/b0-sweep-ARRAY_ID_INDEX.log`; the jobs.csv task is
`OFFSET + INDEX`. When the arrays have finished, rank on a CPU allocation:

```bash
.venv/bin/python -m tapestry.models.manager rank -e b0-sweep
```

The printed folder holds `configuration_ranking.csv`, `run_scores.csv`, and
`season_scores.csv` ([definitions](workflows/experiment-manager.md#ranking)).

## Train across six GPUs

Partition `jlessler` provides:

| Node | CPU cores | Host RAM | GPUs |
|---|---:|---:|---|
| `g1803jles01` | 56 | 500,000 MB | 4 × L40, 48 GB each |
| `g1803jles02` | 64 | 2,048,000 MB | 2 × H100 NVL, 96 GB each |

Each array task is one scenario and runs its three seeds in sequence. It requests
one GPU, four CPUs, 64 GiB RAM, and one day. Six concurrent tasks can use all six
GPUs. Memory and time limits are resource allowances, not measured
requirements. Mixed GPU hardware may introduce small numerical differences;
each task records its allocation and logs the device model.

`status` prints the array tasks that still have unfinished seeds; a fresh
14-scenario experiment reports `0,1,...,13`:

```bash
.venv/bin/python -m tapestry.models.manager status -e b0-explore
sbatch --array=0-13%6 scripts/b0_array.sbatch b0-explore
# Replace ARRAY_ID with the returned job ID.
sbatch --dependency=afterok:ARRAY_ID scripts/b0_compare.sbatch b0-explore
```

Array task numbers are rows of `data/experiments/b0-explore/jobs.csv`. Each seed
attempt writes only its own folder, so there is no prepare or collect step.
`compare` refuses to score while any planned run is incomplete.

## Monitor and resume

```bash
squeue -u "$USER"
.venv/bin/python -m tapestry.models.manager status -e b0-explore
sacct -j ARRAY_ID --format=JobID,State,Elapsed,ExitCode,NodeList
tail -f output/slurm/b0-array-ARRAY_ID_TASK_ID.log
```

`status` rebuilds `runs.csv` (task, seed, status, attempt path, git commit) and
prints the tasks with unfinished seeds. Each attempt folder holds `run.json` and
`run.log`. After the array has ended, resubmit only the printed tasks, for example:

```bash
sbatch --array=2,9%6 scripts/b0_array.sbatch b0-explore
```

Completed seeds are skipped; failed or interrupted seeds restart all three folds in
a new attempt folder, keeping the previous one. A killed job leaves its attempt
marked `running`, so check `squeue` first and never run the same task twice at once.

## Score and review

To score an experiment whose runs are complete:

```bash
sbatch scripts/b0_compare.sbatch b0-explore
tail -F output/slurm/b0-compare-JOB_ID.log
```

The scoring job requests 36 CPU cores and 256 GiB RAM on `g1803jles01`. It runs
`manager compare --workers 9`: saved predictions load concurrently and nine
target/season cases score in parallel, with four numerical-library threads per
worker. Hubverse exports are rewritten each time. EpiBench reuses completed scores
when their provenance matches; cases interrupted after writing scores are moved to
`interrupted-scoring/` and rescored. Rankings and plotting follow scoring.

`comparison.json` records status, output directory, host, Slurm IDs, the scoring
commit, and the commits of the compared runs. On failure, inspect the job log and
resubmit the scoring script after the previous job has ended.
Review `REPORT.md`, `leaderboard.csv`, `run_ranking.csv`,
`configuration_ranking.csv`, and `scores.parquet`. Report WIS, bias, and 50%/95%
coverage by pathogen, horizon, season, and states/DC versus native US, including
means and variability across three seeds. Use the matched controls in
[the manager workflow](workflows/experiment-manager.md) to assess separate heads,
decoder uncertainty, and candidates for combination.
