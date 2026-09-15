# Named B0 experiments

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

# Rank completed runs by total-WIS ratios to the hub ensembles.
.venv/bin/python -m tapestry.models.manager rank -e b0-explore

# Optional, for a few runs: full EpiBench scoring, diagnostic plots, and fans.
.venv/bin/python -m tapestry.models.manager compare -e b0-explore
```

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
withdrawn; the sweep recreates them under this protocol (23 quantiles, total-WIS
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

## Slurm

Both launchers run `manager run --task <row> --device cuda --keep-going` for one
`jobs.csv` row:

| Script | Partitions | Per task | Use |
|---|---|---|---|
| `scripts/b0_sweep.sbatch` | `a100-gpu,l40-gpu,volta-gpu`, QOS `gpu_access` | 1 GPU, 4 CPUs, 16 GiB, 6 h | Large experiments such as the sweep |
| `scripts/b0_array.sbatch` | `jlessler` | 1 GPU, 4 CPUs, 64 GiB, 1 day | Small experiments on the lab's six GPUs |

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
It refuses incomplete runs unless `--allow-incomplete` is given, and each set of
runs gets its own folder, so partial and full rankings never mix. The score follows
[architecture §10.3](../design/architecture.md):

1. **Per target and season:** total model WIS ÷ total ensemble WIS over the
   identical frozen tasks: every location including US, every reference date,
   horizons 0–3, on the hub's 23 quantiles. No per-task or per-location ratio is
   averaged.
2. **Per target:** the mean of its season ratios; each season counts equally.
3. **Combined:** `(2 × (flu + COVID + RSV admissions) + (flu + COVID + RSV ED)) / 9`.
4. **Per configuration:** the mean and seed SD of run scores.

| File | Contents |
|---|---|
| `configuration_ranking.csv` | Rank; mean and SD of the six target scores and the combined score; seed count; states/DC-only and US-only combined means |
| `run_scores.csv` | Per run and geography (`all`, `states_dc`, `US`): six target scores and the combined score |
| `season_scores.csv` | Per run, geography, target, and season: task count, WIS sums and components, ratio, 50/80/90/95% coverage for model and ensemble |
| `manifest.json` | Ranked runs, quantile levels, target weights, and the definition |

A run missing any target gets no combined score rather than a partial one.
`totals.csv` keeps sums by target, season, geography, and horizon, so other
aggregations can be recomputed without rescoring.

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
latent size. Loss weights never change native-unit channel normalization.
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
