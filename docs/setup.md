# Setup

For local work, run `uv sync` from the checkout; use `.venv/bin/python` for commands.

See [the canonical workflow](workflow.md) for the experiment planner
and shared-dispatch commands, used after the environment setup below.

These commands set up the checkout at
`/proj/jlessler/projects/tapestry-all/tapestry` on Longleaf using uv-managed
Python 3.11. Adjust the workspace paths if installing elsewhere. Python tools
and caches live in the workspace; there is no R dependency to set up.

## Clone and install uv

Run once. If uv is already available, skip the installer commands.

```bash
cd /proj/jlessler/projects/tapestry-all
git clone https://github.com/ACCIDDA/chromantis.git tapestry

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

This creates `.venv`, installs Chromantis in editable mode, and includes training,
evaluation, explorer, and test dependencies. Managed Python avoids inheriting
the login shell's Anaconda installation.

## Download the training sources

```bash
uv run python -m tapestry.data --data-root data init
uv run python -m tapestry.data --data-root data pull \
    delphi_nhsn delphi_nssp delphi_claims_inpatient delphi_claims_outpatient \
    delphi_nwss delphi_nwss_aux hub_flusight_current hub_covid_current \
    hub_rsv_current pophive_kinsa_ili \
    delphi_fluview_ilinet delphi_fluview_clinical delphi_flusurv
```

These are the sources `tapestry.dataset.build` reads (targets, claims,
wastewater, Kinsa, ILINet, clinical labs, FluSurv-NET; see [the source catalog](data/sources.md)). The
`cdc_nhsn_*`/`cdc_nssp_*` Socrata specs and the legacy/RSVNet Hub mirrors
remain in the catalog for comparison and are not required for the training
panel. Downloading sources does not build them; see
[the canonical workflow](workflow.md) for that next step.

Scoring is pure Python (`tapestry.evaluation.standard`) -- no R module or
package setup is needed. Use a Slurm allocation for training and substantial
evaluation runs; the environment setup above does not request a GPU or submit
a training job.

## Sync code and panel from the Mac

`data/processed/panel.npz` (about 10 MB) is tracked in git despite the `/data/`
ignore rule (added with `git add -f`), so code and panel travel together. From the
Mac, push to a side branch of the cluster checkout (git refuses to update its
checked-out branch), then fast-forward on Longleaf:

```bash
git remote add longleaf chadi@longleaf.unc.edu:/proj/jlessler/projects/tapestry-all/tapestry
git push longleaf HEAD:refs/heads/incoming
ssh chadi@longleaf.unc.edu 'cd /proj/jlessler/projects/tapestry-all/tapestry && git merge --ff-only incoming'
```

Rebuilding the panel on the cluster is not needed: sources are acquired and built
on the Mac. A planned experiment pins the panel's sha256, so replacing the panel
under a running experiment makes its remaining runs refuse to fit.

**Not synced by git:** the population file `data/metadata/locations.csv` (the one
population file; `LOCATIONS` in `planner.py`) and the frozen support
`data/evaluation/b0_hub_comparison_q23` are git-ignored. Restore the population
file as below (sha256 checked) and copy the frozen support to the cluster once
(e.g. `rsync -a data/evaluation/b0_hub_comparison_q23 longleaf:.../data/evaluation/`).
`plan` pins the sha256 of both (the frozen support through its `manifest.json`)
next to the panel's, and runs refuse to fit if either changed.

## Build the training panel

Run from `/proj/jlessler/projects/tapestry-all/tapestry`. Skip acquisition
above once every source is already downloaded, then build the one array,
`data/processed/panel.npz` (about a minute; one process per source):

```bash
.venv/bin/python -m tapestry.dataset.build build --data-root data
.venv/bin/python -m tapestry.dataset.build show
```

It holds the truth panel and the exact Wednesday as-of store that both input modes
cut episodes from; see [the canonical workflow](workflow.md#acquire-and-build).
`plan` records its sha256, so rebuild it before planning, not while an
experiment is running.

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

Keep these inputs fixed while an experiment runs. `plan` copies `src/` into
`data/experiments/<experiment>/code`, and `jlessler.sbatch` runs that snapshot, so a
Slurm experiment keeps the code it was planned with (re-planning re-pins it). A local
`planner run` runs the working tree instead. Each attempt's `run.json` also records
the git commit and whether the checkout had uncommitted changes; results for the
paper are rerun from a clean tree.

## The frozen evaluation denominator

`tapestry.experiment.planner`'s `fit`/`plan`/`rank` score every run against a
pinned hub-ensemble denominator on the 23-quantile grid,
`data/evaluation/b0_hub_comparison_q23` (`FROZEN` in `planner.py`). Preserve
that directory across checkouts -- **there is currently no supported command
to rebuild it from scratch.** A fresh frozen denominator requires an explicit new implementation; copying the
existing pinned support is the supported setup path.


## GPU execution

The shared launcher uses the hidden `jlessler` partition. Patron resources are:

| Node | GPUs | Host memory | Physical CPU cores |
|---|---|---:|---:|
| `g1803jles01.ll.unc.edu` | 4 × L40, 48 GB each | 512 GB | 56 |
| `g1803jles02.ll.unc.edu` | 2 × H100, 96 GB each | 2 TB | 64 |

Use the [workflow's complete manager commands](workflow.md#plan-launch-and-resume)
for planning, launching, checking and ranking. `scripts/jlessler.sbatch` and the
shared dispatcher control fitting concurrency; use `LANES` for processes per GPU.
Regular GPU partitions are an alternative when explicitly chosen for a run.

## Documentation hosting

The GitHub repository is `ACCIDDA/chromantis`. Documentation is published at
[accidda.github.io/chromantis/](https://accidda.github.io/chromantis/); the repository and Pages path are lowercase. GitHub Pages uses the existing Documentation
Actions workflow on `main`. The Python package remains `tapestry`, and existing
local and Longleaf checkout paths remain unchanged.

For an existing checkout, update its GitHub remote with:

```bash
git remote set-url origin https://github.com/ACCIDDA/chromantis.git
```

Hosting migration, 2026-10-09: updated the repository and documentation URLs after
the GitHub rename to lowercase `chromantis`. Documentation branding is Chromantis;
the importable package and existing compute paths retain `tapestry`. Historical
submitted metadata is preserved as recorded.
