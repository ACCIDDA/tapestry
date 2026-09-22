# Longleaf setup

See [the canonical workflow](workflows/training.md) for the experiment planner
and shared-dispatch commands, used after the environment setup below.

These commands set up the checkout at
`/proj/jlessler/projects/tapestry-all/tapestry` on Longleaf using uv-managed
Python 3.11. Adjust the workspace paths if installing elsewhere. Python tools
and caches live in the workspace; there is no R dependency to set up.

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
uv sync --python-preference only-managed
```

This creates `.venv`, installs Tapestry in editable mode, and includes training,
evaluation, explorer, and test dependencies. Managed Python avoids inheriting
the login shell's Anaconda installation.

## Download the training sources

```bash
uv run python scripts/pull_covariates.py --data-root data init
uv run python scripts/pull_covariates.py --data-root data pull \
    delphi_nhsn delphi_nssp delphi_claims_inpatient delphi_claims_outpatient \
    delphi_nwss delphi_nwss_aux hub_flusight_current hub_covid_current \
    hub_rsv_current pophive_kinsa_ili
```

These are the sources `tapestry.dataset.build` reads (targets, claims,
wastewater, Kinsa; see [the source catalog](data/sources.md)). The
`cdc_nhsn_*`/`cdc_nssp_*` Socrata specs and the legacy/RSVNet Hub mirrors
remain in the catalog for comparison and are not required for the training
arrays. Downloading sources does not build them; see
[the canonical workflow](workflows/training.md) for that next step.

Scoring is pure Python (`tapestry.evaluation.totals`) -- no R module or
package setup is needed. Use a Slurm allocation for training and substantial
evaluation runs; the environment setup above does not request a GPU or submit
a training job.

## Build the training arrays

Run from `/proj/jlessler/projects/tapestry-all/tapestry`. Skip acquisition
above once every source is already downloaded, then build both arrays:

```bash
.venv/bin/python -m tapestry.dataset.build build --data-root data
.venv/bin/python -m tapestry.dataset.build show --dataset data/processed/finalized.npz
.venv/bin/python -m tapestry.dataset.build show --dataset data/processed/vintaged.npz
```

`finalized.npz` and `vintaged.npz` share one covariate name -> column index;
see [the canonical workflow](workflows/training.md#2-build-the-two-training-arrays).

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
  show HEAD:auxiliary-data/locations.csv > data/metadata/locations.csv
printf '%s  %s\n' \
  80aaa24750044a837e063812a1d0a22339c8b3a92513f71a23dac613eaaa29e0 \
  data/metadata/locations.csv | sha256sum --check
)
```

Keep these inputs fixed while an experiment runs. Experiments are not locked to a
code version: each attempt records its git commit and whether the checkout had
uncommitted changes, and results for the paper are rerun from a clean tree.

## The frozen evaluation denominator

`tapestry.experiment.planner`'s `fit`/`plan`/`rank` score every run against a
pinned hub-ensemble denominator on the 23-quantile grid,
`data/evaluation/b0_hub_comparison_q23` (`FROZEN` in `planner.py`). Preserve
that directory across checkouts -- **there is currently no supported command
to rebuild it from scratch.** It used to be built by bootstrap-fitting
`tapestry.models.season_cv` and comparing against the hub with
`tapestry.evaluation.compare`, both removed in the 2026-09 restructuring (see
[docs/design/restructure-2026-unified.md](design/restructure-2026-unified.md)).
Rebuilding it against the unified `Model`/array datasets is unimplemented; if
you need a fresh one, that's new work, not a documented workflow step.

## Run an experiment on the shared GPU partitions

```bash
mkdir -p output/slurm
sbatch --job-name=my-experiment --array=0-13 scripts/jlessler.sbatch my-experiment
```

`jlessler.sbatch` is model-agnostic: it reads the scenario from
`experiment.json` and dispatches through `tapestry.experiment.dispatch`'s
shared job queue, drawing from one queue across every array element instead
of a static task-per-index slice (unlike the old `b0_array.sbatch`/
`b0_sweep.sbatch`, archived in `scripts/archive/2026-09-b0-b1-b2/` along with
the rest of the B0/B1/B2-era launchers). It requests one GPU, four CPUs,
110 GiB, and two days on `g1803jles01`; set `LANES`/`GPUS` to change fitting
processes per GPU and GPU count. See
[the canonical workflow](workflows/training.md#shared-gpu-cluster-launch-longleaf).

```bash
.venv/bin/python -m tapestry.experiment.planner status -e b0-sweep | tail -8
```

When the array has finished, rank on a CPU allocation:

```bash
.venv/bin/python -m tapestry.experiment.planner rank -e b0-sweep
```

The printed folder holds `configuration_ranking.csv`, `run_scores.csv`, and
`season_scores.csv` ([definitions](workflows/training.md#4-plan-run-rank)).

## CPU versus GPU runtime

Benchmark recorded 2026-09-15 UTC for the raw-count B0 baseline: 50 epochs,
eight history weeks, objective loss weights, three season folds per seed,
256 evaluation samples per forecast date, and frozen 23-quantile scoring.
The comparison below covers three seeds (42, 43, 44), or nine season fits.

| Component | CPU: 3 seeds estimated | L40 GPU: 3 seeds measured |
|---|---:|---:|
| Training | 12m 09s | 35s |
| Forecast evaluation | 10m 00s | 1m 50s |
| Scoring and per-run overhead | 42s | 36s |
| Job startup/shutdown | 1m 30s | 10s |
| **Total** | **24m 21s** | **3m 11s** |

The CPU measurement used two cores of an Intel Xeon Gold 6140 on `c0404`,
16 GiB allocated RAM and two numerical-library threads. Its single-seed job
(`1164287`, experiment `b0-cpu-benchmark-2core`) took **9m 07s** including
startup, with about **3.3 GiB peak RAM**. Training took 243.03 seconds,
evaluation 199.93 seconds, and the complete seed run 457.13 seconds.

**Estimation assumption:** each additional CPU seed takes the same time as the
measured seed; job startup is paid once. Thus the three-seed estimate is
`3 × 457.13 + (547 − 457.13) = 1,461.26 seconds`. This is an extrapolation,
not a measured three-seed CPU job; caching and warm-up can change subsequent
seed times.

The GPU measurement used one patron L40 on `g1803jles01`, four CPU cores and
16 GiB allocated host RAM (`1164961_0`, experiment `b0-crosses`, raw reference).
All three seeds completed in 191 seconds. Training and evaluation times come
from the CV manifests; scoring and per-run overhead are the remaining seed
runtime, and startup/shutdown is the remaining Slurm elapsed time. Queue waiting
is excluded. Table entries are rounded.

For this baseline, the estimated GPU advantage is **7.6× overall**, approximately
**21× for training** and **5.5× for evaluation**. CPU execution is feasible, but
most CPU time is computation rather than startup. These results do not establish
timings for other architectures, CPU types, or concurrent runs sharing one GPU.

## Concurrent fits on one L40

Measured 2026-09-16 UTC for B0.1. Each setting completed **the same eight-model
workload**, once, on an L40 on `g1803jles01`. The conditions ran on three L40s
with overlapping execution. Each allocation had eight CPU cores; each fit used
one CPU thread. The outcome is total time to finish all eight models.

| Concurrent fits per GPU | Total time, same 8 models | Models/hour | Speedup versus 1 fit |
|---|---:|---:|---:|
| 1 | 11m 28s | 41.8 | 1.00× |
| 4 | 6m 30s | 73.9 | 1.77× |
| **8** | **5m 24s** | **89.0** | **2.13×** |

**Use eight concurrent fits per L40 for this workload.** Eight reduced total
completion time by **53% versus one** and **17% versus four**. Individual fits
can slow down while sharing a GPU, yet the full workload finishes sooner.
Spare GPU memory and a 100% utilization reading alone did not answer this;
measured aggregate throughput did.

The panel contains the six B0.1 references, a width-128 joint multiscale model,
and the larger conv/residual2 control. Every model used seed 42, the 2025–2026
outer fold, **20 inner-fit epochs plus 20 refit epochs**, and 128/256/2,048
training/validation/evaluation members. Early stopping was disabled to keep
work equal. Timing includes process startup, fitting, validation, forecast
export and waiting for a worker slot; Slurm queue waiting is excluded.

This is a single-pass comparison of this fixed mixed workload, not a measured
full-suite duration or an H100 concurrency result. Eight is the best of the
three tested settings; higher concurrency was not tested. The one-off benchmark
drivers and raw artifacts were removed after recording these results.

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
.venv/bin/python -m tapestry.experiment.planner status -e b0-explore
sbatch --job-name=b0-explore --array=0-13 scripts/jlessler.sbatch b0-explore
```

Array task numbers are rows of `data/experiments/b0-explore/jobs.csv`. Each
seed attempt writes only its own folder, so there is no prepare or collect
step. `jlessler.sbatch` queues a follow-up `notify.sbatch` job automatically
(`NTFY=0` to disable); see [`scripts/b01_notify.py`](../scripts/b01_notify.py).

## Monitor and resume

```bash
squeue -u "$USER"
.venv/bin/python -m tapestry.experiment.planner status -e b0-explore
sacct -j ARRAY_ID --format=JobID,State,Elapsed,ExitCode,NodeList
tail -f output/slurm/tapestry-ARRAY_ID_TASK_ID.log
```

`status` rebuilds `runs.csv` (task, seed, status, attempt path, git commit) and
prints the tasks with unfinished seeds. Each attempt folder holds `run.json` and
`run.log`. After the array has ended, resubmit only the printed tasks, for example:

```bash
sbatch --job-name=b0-explore --array=2,9 scripts/jlessler.sbatch b0-explore
```

Completed seeds are skipped; failed or interrupted seeds restart all three folds in
a new attempt folder, keeping the previous one. A killed job leaves its attempt
marked `running`, so check `squeue` first and never run the same task twice at once.

## Score and review

To score an experiment whose runs are complete, see
[the canonical workflow's plan/run/rank step](workflows/training.md#4-plan-run-rank):

```bash
uv run python -m tapestry.experiment.planner rank -e b0-explore
```

`rank` computes WIS in pure Python (`tapestry.evaluation.totals`) against the
frozen ensemble denominator and writes `season_scores.csv`,
`season_composite_scores.csv`, `run_scores.csv`, and `configuration_ranking.csv`
under `data/experiments/<experiment>/ranking-<hash>/`; pass `--allow-incomplete`
to rank before every run has finished. `configuration_ranking.csv` averages
across seeds -- read that for the WIS-vs-ensemble ratio by scenario.
