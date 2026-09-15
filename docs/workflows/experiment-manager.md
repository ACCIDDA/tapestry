# Named B0 experiments

The manager follows InfluPaint's batch pattern (`influpaint/influpaint/batch/`):
an immutable `TrainingScenario`, a short readable scenario string, a job list whose
rows are Slurm array tasks, and one output folder per run. It uses local JSON/CSV
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

# Score completed runs on frozen ensemble-supported tasks through EpiBench.
.venv/bin/python -m tapestry.models.manager compare -e b0-explore
```

After reinstalling the package, `tapestry-experiments` is an equivalent entrypoint.
Training and scoring are separate commands; completing `run` alone produces CV
diagnostics, not official rankings.

## Scenario strings

A scenario fixes every training choice, including the runtime settings:

```text
b0:h12:tr_4rt:geo1:dyn1:lw_first:enc_mlp:hd_sh:dec_leg:z16:w64:ep50:bs8:m8:lr0.001
```

| Token | Field | Values |
|---|---|---|
| `h` | history weeks (`lookback`) | integer |
| `tr_` | count transform | `raw`, `sqrt`, `4rt` (fourth root) |
| `geo`, `dyn` | geography and dynamics features | `0`, `1` |
| `lw_` | loss weights | `first` (influenza first), `bal` (balanced admissions), `fluonly` |
| `enc_` | encoder | `mlp`, `conv` |
| `hd_` | prediction heads | `sh` (shared), `su` (separate state/US) |
| `dec_` | decoder | `leg` (legacy), `res2` (two residual blocks) |
| `z` | latent dimension | integer |
| `w`, `ep`, `bs`, `m` | width, epochs, batch size, training members | integers |
| `lr` | learning rate | number |

Every token is always present, in this order. Parsing is strict: a typo, a missing
token, or a non-canonical number (`h012`, `lr1e-3`) fails rather than falling back
to a default. Pass an alias such as `state_us` or a quoted full string to
`--scenario`; `TrainingScenario` and `dataclasses.replace` give the same interface
from Python. Stable strings replace InfluPaint's position-based numeric IDs.

The scenario string is the configuration ID in every comparison output, and a run
appends its seed: `<scenario>:s42`. Code, data, and git versions are provenance,
not part of the ID.

The dataset, population file, frozen scoring inputs, evaluation draws (2,048 by
default), and device are experiment settings in `experiment.json`, not scenario fields.

## Suites

`--suite essential` (the default) is the 14-configuration comparison below, and
`--suite grid` is the full factorial. For a new exploration, add a named suite to
`SUITES` in `src/tapestry/models/scenarios.py`. `ofat` builds one-factor-at-a-time
variants around an anchor:

```python
SUITES = {
    ...,
    'capacity': lambda: {'anchor': ANCHOR, **ofat(ANCHOR, width=(32, 128), epochs=(100,))},
}
```

This suite contains `anchor`, `width_32`, `width_128`, and `epochs_100`. Suites and
`--scenario` selections can be added to an existing experiment at any time.

## Layout and resume

```text
data/experiments/<experiment>/
  experiment.json      dataset, population file, frozen inputs, evaluation draws, device
  jobs.csv             task,name,scenario,seeds — one row per Slurm array task
  runs.csv             rebuilt by status/compare: one row per scenario × seed
  <scenario>/
    s42/
      attempt-001/
        run.json       status, command, settings, times, host, Slurm IDs, git commit/dirty
        run.log
        cv/
          manifest.json
          scores.csv
          eval_2023-2024/{model.pt,forecasts.npz,training.json,scores.csv}
          eval_2024-2025/...
          eval_2025-2026/...
    s43/...
  comparison.json
  comparison-<set-hash>/
    REPORT.md, rankings, EpiBench inputs and scores, Hubverse forecasts, plots
```

`plan` appends scenarios and seeds to `jobs.csv`; existing task numbers never
change. One task runs its scenario's seeds in sequence (by default three seeds,
each fitting three season folds). A seed is complete when an attempt's `run.json`
says so and its manifest plus all three folds' artifacts exist; `run` then skips it.
Otherwise `run` starts the next `attempt-NNN`, preserving failed and interrupted
attempts. This resumes whole scenario/seed runs, not optimizer state or single folds.
`--keep-going` continues with the remaining seeds after a failure and still exits
unsuccessfully.

Each attempt writes only its own folder, and `runs.csv` is rebuilt by scanning
attempts, so array tasks never write a shared registry. Nothing is locked: do not
run the same task twice at once. A killed job leaves its attempt marked `running`;
check `squeue` before resubmitting that task.

## Provenance instead of locks

Experiments are not locked to a code or data version, so scenarios can be added
after changing model code. Every attempt records the git commit and whether the
checkout had uncommitted changes (`git_dirty`), along with its settings, host,
and Slurm IDs. `cv/manifest.json` also keeps code and dataset hashes.
`compare` warns when compared runs span several commits or include uncommitted
changes. Assumption: results reported in the paper will be rerun from a clean tree,
so mixed commits are acceptable only during exploration.

A later `plan` with different settings updates `experiment.json` and prints the
changed keys; each attempt keeps the settings it actually used.

## Slurm

```bash
mkdir -p output/slurm
.venv/bin/python -m tapestry.models.manager plan -e b0-explore --device cuda
.venv/bin/python -m tapestry.models.manager status -e b0-explore
sbatch --array=0-13%6 scripts/b0_array.sbatch b0-explore
# After the array has finished:
sbatch scripts/b0_compare.sbatch b0-explore
```

`scripts/b0_array.sbatch` runs `manager run --task $SLURM_ARRAY_TASK_ID --device cuda --keep-going`
for one `jobs.csv` row. To retry failures, resubmit only the pending task numbers
printed by `status`. `scripts/b0_compare.sbatch` runs `manager compare --workers 9`
on the 36-core node; the default of two workers suits a 32 GiB laptop. Arguments
after the experiment name are passed to the manager, for example `--root`.

## Comparison

`compare` scores every run in `runs.csv` through `tapestry.evaluation.sweep`. It
refuses incomplete runs unless `--allow-incomplete` is given. Each set of runs has
its own `comparison-<hash>` folder, so partial and full rankings never mix. EpiBench
scores are reused when their inputs and scorer match. A case interrupted after
EpiBench wrote scores, but before provenance was saved, is moved to
`interrupted-scoring/` and rescored. Publish a completed comparison with
`scripts/publish_evaluation_docs.py --comparison data/experiments/<experiment>/comparison-<hash>`.

## Comparison size and controls

**Recommended: 14 configurations × 3 seeds = 42 CV runs = 126 season fits.**
Every candidate receives all three seeds; there is no seed-42 screening step.
One run means one configuration and seed evaluated in all three held-out seasons.
The historical control is raw counts, 8 weeks, no geography or dynamics, and the
original decoder. The anchor is the existing fourth-root, geography, 12-week,
dynamics candidate with influenza-first supervision, shared MLPs, shared decoder,
and latent dimension 16. Neither is assumed superior at state level.

| Alias | Change | Matched control |
|---|---|---|
| `baseline` | Historical raw-count B0 | Reference |
| `anchor` | Existing feature/representation candidate | `baseline`; bundled historical comparison |
| `state_us` | Separate state and native-US stochastic heads | `anchor` |
| `residual2` | Two modulated residual decoder blocks; latent stays 16 | `anchor` |
| `latent32` | Latent 32 with original decoder | `anchor` |
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

Scenario defaults: width 64, 50 epochs, learning rate .001, batch size 8, and 8
training draws. Experiment defaults: 2,048 evaluation draws, seeds 42/43/44, four
horizons, and the existing three seasons. Parameter counts are saved for every fold.
History changes MLP input size; convolution reuses its filters across weeks. These
are fixed-width recipe comparisons, not parameter-count-matched experiments.

For a literal full factorial comparison, `--suite grid` crosses 3 histories ×
2 dynamics settings × 3 losses × 2 encoders × 2 head choices × 2 decoder choices ×
2 latent sizes = **288 configurations**, with fourth-root/geography fixed. Adding
the historical baseline gives **289 configurations, 867 CV runs, 2,601 fits**.
The focused 14 are a subset of this grid. The grid is available but is not the
recommended first round. A later combined candidate adds **3 runs / 9 fits**.

## Architecture and evaluation assumptions

- B0 stays local. Separate heads share context/focal encoders, source/horizon
  embeddings, and latent draws. They have separate modulation and output
  parameters, initialized identically, and route by the exact `US` location ID.
- `residual2` uses two width-sized residual MLP blocks, with layer normalization
  and latent affine modulation in each block. Both receive the same independent
  Gaussian draw per member/episode, shared across locations, horizons, and channels.
  No independent observation noise is appended; native-unit fair CRPS is retained.
- The temporal option uses two kernel-3 convolutions with SiLU activations on
  value/mask pairs, then concatenates mean and latest features for projection.
  It shares detectors across context weeks, and focal detectors across channels.
  Symmetric padding operates entirely within observed context. Calendar,
  geography, and optional dynamics enter after temporal encoding.
- Frozen EpiBench comparisons report target, season, states/DC versus US, and
  horizon separately, including WIS, bias, dispersion, and 50%/95% coverage.
  Current repository exports use five quantiles, not the earlier 23-quantile
  historical protocol. Raw six-channel CV diagnostics are also retained.
- No calibration or holdout-driven stopping is implemented. Any future calibration
  needs inner out-of-sample predictions and inner-fold scalers; its extra fits
  are not included in these counts. A blanket interval multiplier is not used.
- Spatial attention is deferred to B1 under the instruction to remain in B0.
  Once implemented, one isolated spatial candidate at three seeds would add
  **3 CV runs / 9 season fits** against an already-run matched control. It is not
  part of either executable B0 suite. Shared randomness does not transmit other
  locations' observed histories.

All three seasons have already informed development. These comparisons remain
exploratory finalized-data CV, with later seasons in the fitting set for the first
two folds; they are not prospective validation. No performance improvement is
claimed by implementing the manager or running a smoke test.
