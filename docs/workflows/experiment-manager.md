# Named B0 experiments

The manager follows InfluPaint's `TrainingScenario` dataclass, readable
`scenario_string`, and essential-versus-full-grid pattern from
`influpaint/influpaint/batch/scenarios.py`. An experiment name groups scenarios,
seeds, logs, checkpoints, forecasts, and comparisons. It uses local JSON files
and the existing CV/EpiBench commands, with no new dependencies or MLflow server.

Run from the repository root in the installed environment:

```bash
# Inspect the fixed comparison and its exact number of fits; no files are written.
.venv/bin/python -m tapestry.models.manager list

# Save the protocol and planned runs without fitting anything.
.venv/bin/python -m tapestry.models.manager plan -e b0-next

# Run every scenario at seeds 42, 43, 44. Repeat this command to resume.
.venv/bin/python -m tapestry.models.manager run -e b0-next

.venv/bin/python -m tapestry.models.manager status -e b0-next

# Score all completed runs on frozen ensemble-supported tasks through EpiBench.
.venv/bin/python -m tapestry.models.manager compare -e b0-next
```

After reinstalling the package, `tapestry-experiments` is an equivalent entrypoint.
For a smaller first stage, use `--scenario baseline anchor state_us residual2` on
both `plan` and `run`. Scenarios can be added under the same experiment name with
the same protocol. A subsequent plain `run` registers the full essential suite.
`compare` requires all registered runs to have completed, so a partial set cannot
silently produce the planned full comparison. Training and scoring are separate
commands; completing `run` alone produces CV diagnostics, not official rankings.

## Strings and saved results

Aliases such as `state_us` resolve to complete, round-trippable strings:

```text
b0::lookback=12::count_transform=fourth_root::geography=1::dynamics=1::loss_weights=influenza_first::encoder=mlp::heads=state_us::decoder=legacy::latent=16
```

Pass either an alias or a quoted full string to `--scenario`. The latter also
supports explicit combinations after the individual tests. `TrainingScenario`
and `dataclasses.replace` provide the same interface from Python. Unknown aliases,
malformed strings, and unknown options fail rather than falling back to defaults.
Stable names replace InfluPaint's position-based numerical IDs.

```text
data/experiments/<experiment>/
  protocol.json
  runs.json
  <scenario_string>::s42/
    attempt-001/
      run.json
      run.log
      cv/
        manifest.json
        scores.csv
        eval_2023-2024/{model.pt,forecasts.npz,training.json,scores.csv}
        eval_2024-2025/...
        eval_2025-2026/...
  comparison.json
  comparison-<set-hash>/
    REPORT.md
    configuration_ranking.csv
    run_ranking.csv
    leaderboard.csv
    ... EpiBench inputs, scores, Hubverse forecasts, and plots ...
```

`runs.json` records planned/running/failed/complete status, seed, full scenario,
commands, timestamps, errors, attempt paths, and logs. A completed run is reused
only if its manifest and all three folds' required artifacts exist. Failed or
interrupted attempts are preserved; retry restarts that scenario/seed's three
folds in a fresh attempt directory. This is experiment resume, not optimizer or
mid-fold checkpoint resume. Runs execute sequentially and one process can write
an experiment at a time. `--keep-going` finishes the remaining runs after failures
and exits unsuccessfully if any failed.

The protocol records code hashes, dataset/population hashes, frozen scoring
inputs, shared runtime settings, and assumptions. Changing those requires a new
experiment name; changing the seed/scenario selection can extend an existing
experiment. Use a distinct name for smoke runs. Historical artifacts are not
automatically imported because their code and scoring protocol differ.

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

Shared defaults: width 64, 50 epochs, learning rate .001, batch size 8, 8 training
draws, 2,048 evaluation draws, seeds 42/43/44, four horizons, and the existing three
seasons. Parameter counts are saved for every fold. History changes MLP input
size; convolution reuses its filters across weeks. These are fixed-width recipe
comparisons, not parameter-count-matched experiments.

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
