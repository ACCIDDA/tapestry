# Named experiments

**One manager, one scorer, both models.** B0 and B1 share every command below,
the same attempt/resume records, the same GPU dispatcher and the same ranking:
season-equal location-relative WIS against the official ensembles on the frozen
Hub tasks. The manager has no model-specific branches; the only differences live
in `src/tapestry/models/backends.py`, which says how to expand a suite, how to
launch a fold's fit, and which artifacts prove a seed finished.

B1 adds one thing B0 does not have: it also predicts the two completed weeks at
offsets -2 and -1. No Hub ensemble forecasts a week that has already happened, so
those are ranked separately against **preliminary-value persistence** — the naive
nowcast that Wednesday's visible value is already final — using the identical
weights and aggregation. `rank` writes that table under `ranking-<hash>/nowcast/`.
Forecast and nowcast scores share weights but not support and are never combined.

Plan B1 with `--suite B1` for the open formulation grid, or with one of the two
attribution suites, `--suite B1-onlynowcast` / `--suite B1-onlymask`, which run
B0's four best configurations while changing exactly one thing. See the
[B1 design](../design/b1.md#named-experiments-and-cluster-launch).

The manager uses an immutable `TrainingScenario`, a short readable scenario
string, a job list whose rows are Slurm array tasks, and one output folder per run. It uses local JSON/CSV
files and the existing CV/EpiBench commands, with no MLflow server.

Run from the repository root in the installed environment. Stored paths are
relative to that root or to the experiment folder, so an experiment folder can be
copied between Longleaf and a laptop.

```bash
# Inspect a suite and its exact number of fits; no files are written.
.venv/bin/python -m tapestry.models.manager list

# Register scenarios × seeds in jobs.csv and settings in experiment.json; nothing is fitted.
.venv/bin/python -m tapestry.models.manager plan -e b0-explore

# Run every task in sequence on this machine. Repeat the command to resume.
.venv/bin/python -m tapestry.models.manager run -e b0-explore

# Per-run status; also prints the pending Slurm array tasks.
.venv/bin/python -m tapestry.models.manager status -e b0-explore

# Rank by season-equal location-relative WIS, with US weight 20%.
.venv/bin/python -m tapestry.models.manager rank -e b0-explore

# Optional, for a few runs: full EpiBench scoring, diagnostic plots, and fans.
.venv/bin/python -m tapestry.models.manager compare -e b0-explore
```

Once the fits are done, [Postprocessing an experiment](experiment-postprocessing.md)
walks through checking completion, finding the attempt a run actually used,
ranking, and turning the ranking into figures and a results page.

`tapestry-experiments` is an equivalent installed entrypoint.
`run` fits each seed's three season folds, then scores them against the official
ensembles on the frozen tasks (`totals.csv`), so `rank` needs no separate scoring
job. `run` refuses to start when the frozen support is missing or was built for a
different quantile grid (`--frozen`, default `data/evaluation/b0_hub_comparison_q23`).

## Scenario strings

A scenario fixes every training choice, including the runtime settings:

```text
b0:h12:tr_4rt:ed_lin:geo1:dyn1:lw_first:enc_mlp:sp_none:hd_sh:dec_leg:nz_glob:z16:w64:ep50:pat0:bs8:m8:lr0.001
```

| Token | Field | Values |
|---|---|---|
| `h` | history weeks (`lookback`) | integer |
| `tr_` | count transform | `raw` (counts), `rate` (per 100,000), `sqrt`, `4rt` (fourth root), `log1p` |
| `ed_` | ED transform | `lin` (original), `logit`, `4rt` |
| `geo`, `dyn` | geography and dynamics features | `0`, `1` |
| `lw_` | loss weights | `first` (influenza first), `bal` (balanced admissions), `fluonly`, `obj` (`[1,1,1,.5,.5,.5]`) |
| `enc_` | encoder | `mlp`, `conv` |
| `sp_` | spatial exchange | `none`, `attn` (one attention block across locations) |
| `hd_` | prediction heads | `sh` (shared), `su` (separate state/US) |
| `dec_` | decoder | `leg` (legacy), `res2` (two residual blocks) |
| `nz_` | noise | `glob` (global latent), `loc` (global plus per-location latent) |
| `z` | latent dimension | integer |
| `w`, `ep`, `bs`, `m` | width, epochs, batch size, training members | integers |
| `pat` | early-stopping patience | `0` trains `ep` epochs; otherwise `ep` is the cap |
| `lr` | learning rate | number |

Every token is always present, in this order. Parsing is strict: a typo, a missing
token, or a non-canonical number (`h012`, `lr1e-3`) fails rather than falling back
to a default. Pass an alias such as `state_us` or a quoted full string to
`--scenario`; `TrainingScenario` and `dataclasses.replace` give the same interface
from Python.

The scenario string is the configuration ID in every ranking and comparison output,
and a run appends its seed: `<scenario>:s42`. Code, data, and git versions are
provenance, not part of the ID.

The dataset, population file, frozen scoring inputs, evaluation draws (2,048 by
default), and device are experiment settings in `experiment.json`, not scenario fields.

## Suites

`--suite essential` (the default) is the 14-configuration comparison below, and
`--suite grid` is the [architecture sweep](#architecture-sweep). For a new
exploration, add a named suite to `SUITES` in `src/tapestry/models/scenarios.py`.
`ofat` builds one-factor-at-a-time variants around an anchor:

```python
SUITES = {
    ...,
    'capacity': lambda: {'anchor': ANCHOR, **ofat(ANCHOR, width=(32, 128), epochs=(100,))},
}
```

This suite contains `anchor`, `width_32`, `width_128`, and `epochs_100`. Suites and
`--scenario` selections can be added to an existing experiment at any time.

## Architecture sweep

`--suite grid` is the full factorial below plus the baseline:
**4,097 configurations × 3 seeds = 12,291 CV runs = 36,873 season fits**, with one
array task per configuration.

| Factor | Levels |
|---|---|
| History (`h`) | 8, 12 weeks |
| Dynamics features (`dyn`) | off, on |
| Temporal encoder (`enc_`) | MLP, convolution |
| Decoder (`dec_`) | `legacy` (single modulation), two modulated residual blocks |
| Latent dimension (`z`) | 16, 32 |
| Spatial exchange (`sp_`) | none, one attention block across locations |
| Noise (`nz_`) | global latent; global plus per-location latent |
| Heads (`hd_`) | shared, separate state/US |
| Count transform (`tr_`) | rate per 100,000, square root, fourth root, log1p |
| ED transform (`ed_`) | logit, fourth root |
| Stopping (`ep`, `pat`) | fixed 50 epochs; early stopping with patience 20 and a 300-epoch cap |

Fixed: geography features, loss weights `[1,1,1,.5,.5,.5]` (matching the selection
score), width 64, batch size 8, 8 training draws, learning rate .001, and 2,048
evaluation draws. The baseline is raw counts, 8 weeks, no geography or dynamics,
the `legacy` decoder, linear ED inputs, and fixed 50 epochs, with the same loss
weights. A transform sets both the input representation and the space where
residuals and latent perturbations act; the switches are described in
[training](training.md#experiment-switches).

Earlier B0 comparisons, including the published essential-suite results, were
withdrawn; the sweep recreates them under this protocol (23 quantiles, location-relative season-first
ranking). All three seasons inform development, so the rankings are exploratory,
not validation.

## Layout and resume

```text
data/experiments/<experiment>/
  experiment.json      dataset, population file, frozen inputs, evaluation draws, device
  jobs.csv             task,name,scenario,seeds — one row per Slurm array task
  runs.csv             rebuilt by status/rank/compare: one row per scenario × seed
  <scenario>/
    s42/
      attempt-001/
        run.json       status, commands, settings, times, host, Slurm IDs, git commit/dirty
        run.log        CV fitting, then totals scoring
        cv/
          manifest.json
          scores.csv
          totals.csv   model and ensemble WIS sums on the frozen tasks
          eval_2023-2024/{model.pt,forecasts.npz,training.json,scores.csv}
                       with early stopping, also validation_forecasts.npz and validation_scores.csv
          eval_2024-2025/...
          eval_2025-2026/...
    s43/...
  ranking-<set-hash>/
    configuration_ranking.csv, run_scores.csv, season_scores.csv, manifest.json
  comparison.json
  comparison-<set-hash>/
    REPORT.md, rankings, EpiBench inputs and scores, Hubverse forecasts, plots
```

`plan` appends scenarios and seeds to `jobs.csv`; existing task numbers never
change. One task runs its scenario's seeds in sequence (by default three seeds,
each fitting three season folds). A seed is complete when an attempt's `run.json`
says so and its manifest, `totals.csv`, and all three folds' artifacts exist; `run`
then skips it. Otherwise `run` starts the next `attempt-NNN`, preserving failed and
interrupted attempts. This resumes whole scenario/seed runs, not optimizer state or
single folds. `--keep-going` continues with the remaining seeds after a failure and
still exits unsuccessfully.

Each attempt writes only its own folder, and `runs.csv` is rebuilt by scanning
attempts, so array tasks never write a shared registry. Nothing is locked: do not
run the same task twice at once. A killed job leaves its attempt marked `running`;
check `squeue` before resubmitting that task.

## Provenance instead of locks

Experiments are not locked to a code or data version, so scenarios can be added
after changing model code. Every attempt records the git commit and whether the
checkout had uncommitted changes (`git_dirty`), along with its settings, host,
and Slurm IDs. `cv/manifest.json` also keeps code and dataset hashes.
`rank` and `compare` warn when runs span several commits or include uncommitted
changes. Assumption: results reported in the paper will be rerun from a clean tree,
so mixed commits are acceptable only during exploration.

A later `plan` with different settings updates `experiment.json` and prints the
changed keys; each attempt keeps the settings it actually used.

## Smaller calibration: crosses

The `crosses` suite replaces the cancelled full `b0-sweep` with 51 unique
configurations (153 seed runs, 459 season fits). Three reference recipes are
chosen in advance: historical raw-count B0, the existing fourth-root/geography/
dynamics anchor, and an anchor with convolution, residual decoder, latent 32,
spatial attention, local noise, separate state/US heads and logit ED inputs.
These references are design choices, not winners selected from the partial sweep.

Each reference gets every original grid level as a one-factor change. Raw count,
linear ED and geography on/off are also included to connect the historical
references. Epoch cap and patience change together as one stopping policy.
Duplicate configurations run once. All runs use objective loss weights, seeds
42/43/44 and the same frozen 23-quantile scoring support. This screens local
effects around different recipes; exhaustive interaction testing is out of scope.
The cancelled sweep's artifacts are retained separately; this small experiment
runs fresh so partial attempts are not treated as completed results.

```bash
.venv/bin/python -m tapestry.models.manager plan -e b0-crosses --suite crosses --device cuda
.venv/bin/python -m tapestry.models.manager status -e b0-crosses
sbatch --array=0-50 --export=ALL,OFFSET=0 scripts/b0_sweep.sbatch b0-crosses
```

## Slurm

| Script | Shape | Use |
|---|---|---|
| `scripts/jlessler.sbatch` | One GPU per array element, several fitting lanes inside, all drawing from one shared queue | Any experiment planned with a source snapshot, B0 or B1 |
| `scripts/b0_sweep.sbatch` | One `jobs.csv` row per array task | Large static sweeps |
| `scripts/b0_array.sbatch` | One `jobs.csv` row per array task | Small experiments on the lab's six GPUs |

`jlessler.sbatch` takes the experiment name and reads the model from
`experiment.json`, so the same launcher serves both models:

```bash
sbatch --job-name=B1-onlynowcast --array=0-3 scripts/jlessler.sbatch B1-onlynowcast
LANES=4 GPUS=6 sbatch --job-name=B1-onlynowcast --array=0-1 \
  --nodelist=g1803jles02 scripts/jlessler.sbatch B1-onlynowcast
```

The two static launchers run `manager run --task <row> --device cuda --keep-going`
for one `jobs.csv` row.

The calibration now uses the patron nodes `g1803jles01` (four L40 GPUs) and
`g1803jles02` (two H100 GPUs). Previous shared-partition settings remain
commented in `scripts/b0_sweep.sbatch`.

Volta V100 is excluded because the installed PyTorch 2.14 CUDA 13 build lacks
its compute-capability 7.0 kernels. A100 and L40 remain supported.

Resource limits are allowances, not measurements. Slurm caps array indices, so
`b0_sweep.sbatch` adds `OFFSET` to `SLURM_ARRAY_TASK_ID`. `status` prints
ready-to-submit commands for the unfinished tasks, in chunks of 1,000:

```bash
mkdir -p output/slurm
.venv/bin/python -m tapestry.models.manager status -e b0-sweep
sbatch --array=0-999 --export=ALL,OFFSET=0 scripts/b0_sweep.sbatch b0-sweep
sbatch --array=0-999 --export=ALL,OFFSET=1000 scripts/b0_sweep.sbatch b0-sweep
# ... one line per printed chunk
```

To retry failures, resubmit only the pending tasks printed by `status`.
`scripts/b0_compare.sbatch` runs `manager compare --workers 9` on the 36-core node;
the default of two workers suits a 32 GiB laptop. Arguments after the experiment
name are passed to the manager, for example `--root`.

## Ranking

`rank` reads every complete run's `totals.csv` and writes `ranking-<set-hash>/`.
The hash includes the scoring version and run set. It refuses incomplete runs
unless `--allow-incomplete` is given. The score follows
[architecture §10.3](../design/architecture.md#103-metrics):

1. Per target/season/location: total native model WIS / total ensemble WIS over
   identical dates and horizons 0–3, on the hub's 23 quantiles.
2. State/DC ratios average equally with 80% weight; native US gets 20%.
3. Within each season, average available targets with admission weights 1 and
   ED weights .5, normalized by available target weight.
4. Average season composites equally, then report mean and SD across seeds.

| File | Contents |
|---|---|
| `configuration_ranking.csv` | Rank; six per-target means/SDs; combined mean/SD; states/DC and US combined scores |
| `run_scores.csv` | Per-run/geography target means and season-first combined score |
| `season_scores.csv` | Target/season location-relative ratio, pooled ratio, native sums, weighted coverage, effective US weight |
| `season_composite_scores.csv` | Target scores and weighted composite within each season |
| `manifest.json` | Runs, quantiles, target weights, US weight, scoring version and definition |

Missing target/season support in a run cannot silently improve its score. A
challenge absent from the shared frozen support receives no weight in that
season. `totals.csv` retains location/horizon sums. Nonpositive ensemble WIS
at a location raises an error rather than creating an undefined ratio.
Old totals without locations must first be regenerated from saved forecasts:

```bash
.venv/bin/python -m tapestry.evaluation.totals score --run '<cv-folder>' --frozen data/evaluation/b0_hub_comparison_q23
```

Rescoring old predictions evaluates the new ranking objective; it does not
retroactively change their training loss. Use a fresh experiment for new fits.

## Comparison

`compare` scores every run in `runs.csv` through `tapestry.evaluation.sweep`, with
the full EpiBench pipeline, diagnostic plots, and fans. It suits tens of runs, such
as a shortlist from `rank`, not the full sweep. It refuses incomplete runs unless
`--allow-incomplete` is given. Each set of runs has its own `comparison-<hash>`
folder, so partial and full rankings never mix. EpiBench scores are reused when
their inputs and scorer match. A case interrupted after EpiBench wrote scores, but
before provenance was saved, is moved to `interrupted-scoring/` and rescored.
Publish a completed comparison with
`scripts/publish_evaluation_docs.py --comparison data/experiments/<experiment>/comparison-<hash>`.

## Essential suite and controls

**14 configurations × 3 seeds = 42 CV runs = 126 season fits.**
Every candidate receives all three seeds; there is no seed-42 screening step.
One run means one configuration and seed evaluated in all three held-out seasons.
The baseline control is raw counts, 8 weeks, no geography or dynamics, and the
`legacy` decoder. The anchor is the fourth-root, geography, 12-week, dynamics
candidate with influenza-first supervision, shared MLPs, shared decoder,
and latent dimension 16. Neither is assumed superior at state level.

| Alias | Change | Matched control |
|---|---|---|
| `baseline` | Raw-count B0 | Reference |
| `anchor` | Feature/representation candidate | `baseline` |
| `state_us` | Separate state and native-US stochastic heads | `anchor` |
| `residual2` | Two modulated residual decoder blocks; latent stays 16 | `anchor` |
| `latent32` | Latent 32 with `legacy` decoder | `anchor` |
| `residual2_z32` | Two modulated blocks and latent 32 | `residual2` and `latent32` |
| `mlp_h8` | 8 weeks, dynamics off | `mlp_h12` |
| `mlp_h12` | 12 weeks, dynamics off | `anchor` for the dynamics effect |
| `mlp_h26` | 26 weeks, dynamics off | `mlp_h12` |
| `balanced` | `[1,1,1,.1,.1,.1]` | `anchor` |
| `flu_only` | `[1,0,0,0,0,0]`, retaining all six inputs | `anchor` |
| `conv_h12` | Small temporal convolution, 12 weeks | `anchor` |
| `mlp_h26_dynamics` | 26 weeks with dynamics | `mlp_h26` |
| `conv_h26` | Small temporal convolution, 26 weeks | `mlp_h26_dynamics` |

The extra 26-week MLP with dynamics prevents confounding history, dynamics, and
encoder in the convolution comparison. The decoder's depth and latent size form
a 2×2 comparison, so any improvement from more blocks need not be attributed to
latent size. Loss weights never change native-unit channel/location normalization.
Unsupervised auxiliary outputs in `flu_only` are not trained auxiliary forecasts.

Scenario defaults: width 64, 50 epochs with no early stopping, learning rate .001,
batch size 8, and 8 training draws. Experiment defaults: 2,048 evaluation draws,
seeds 42/43/44, four horizons, and three seasons. Parameter counts are saved for
every fold. History changes MLP input size; convolution reuses its filters across
weeks. These are fixed-width recipe comparisons, not parameter-count-matched
experiments.

## Architecture and evaluation assumptions

- Without spatial attention, B0 stays local. Attention mixes the per-location
  context embeddings within one episode; shared randomness alone does not transmit
  other locations' observed histories.
- Separate heads share context/focal encoders, source/horizon embeddings, the
  spatial block, and latent draws. They have separate modulation (including local
  noise) and output parameters, initialized identically, and route by the exact
  `US` location ID.
- `residual2` uses two width-sized residual MLP blocks, with layer normalization
  and latent affine modulation in each block. The global draw per member and
  episode is shared across locations, horizons, and channels; the optional local
  draw is shared across one location's horizons and channels. No independent
  observation noise is appended; native-unit fair CRPS is retained.
- The temporal option uses two kernel-3 convolutions with SiLU activations on
  value/mask pairs, then concatenates mean and latest features for projection.
  It shares detectors across context weeks, and focal detectors across channels.
  Symmetric padding operates entirely within observed context. Calendar,
  geography, and optional dynamics enter after temporal encoding.
- Totals and frozen EpiBench comparisons report target, season, states/DC versus
  US, and horizon separately. Exports and scoring use the hub's 23 quantiles.
  Raw six-channel CV diagnostics are also saved.
- Early stopping uses validation blocks inside the training seasons, never the
  held-out season. Their out-of-sample forecasts are saved for later calibration;
  no calibration is applied, and a blanket interval multiplier is not used.

All three seasons inform development. These comparisons are exploratory
finalized-data CV, with later seasons in the fitting set for the first two folds;
they are not prospective validation.

## B0.1 architecture suite

The [B0.1 design](../design/b0.1.md#run-b01) contains 172 recipes and three seeds.
`LANES=6` runs six fits per GPU. Submit four array elements on `g1803jles01` and
two on `g1803jles02`, using `scripts/jlessler.sbatch` as shown in the design.
All six allocations draw from one shared queue; large and small jobs are spread
by estimated workload, and each seed can move to another GPU when it starts.
Sources are read from the prepared snapshot. Never refresh it during a run.
