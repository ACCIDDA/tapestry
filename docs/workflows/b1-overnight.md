# B1 overnight formulation screen

Prepared 2026-09-17; no jobs submitted. Suite `B1-overnight`, experiment
`B1-screen-256`. The executable design is
`src/tapestry/models/b1_overnight.py`; `manager plan` records each named scenario
in jobs.csv and pins source, inputs and this document.

## Evidence and references

[B0.1](../results/b0-1-crosses/index.md) favored independent target/pathogen
fits. Use rank 1 target MLP (100 epochs, .883), rank 4 pathogen MLP (100,
.897), rank 5 target multiscale convolution (100, .909), and rank 6 joint
location/target MLP with pathogen heads (300, .911). Epoch values are caps,
with patience 30. Additional epoch-cap contrasts are deferred.
These are B0-derived recipes under B1 inputs, not B1 performance claims.

[First good B1](../results/first-good-b1.md) motivates supplied-final flags as
the anchor: seed42 B scored .942 versus A .989 and C .959. C helped most under
outages. These historical numbers used 2,048 evaluation draws; rerun every
candidate here at 256. No historical forecasts/checkpoints are imported.

## Design: 40 configurations, 120 seed runs, 360 season folds

All four references receive the same panels:

| Panel | Settings | Configurations |
|---|---|---:|
| Formulation | A, B, C, two-stage, each at masking rate .5 | 16 |
| Masking rate around B | Additional episode rates 0 and .25 | 8 |
| Mask mechanism around B | Recent-only, gap-only, outage-only, rate .5 | 12 |
| Auxiliary nowcasting under outages | C with outage-only masking, rate .5 | 4 |

Pipelines are A `direct`, B `direct_finalflag`, C `joint_aux025`, and
`two_stage`. C uses forecast loss + .25 recent loss, a parallel recent head,
supplied-final flags, and forecast-only checkpoint selection. The two-stage
model feeds sampled recent values into the forecast, consumes supplied-final
flags and bypasses visible finals, and selects on equally weighted recent/future loss.
Thus two-stage comparisons evaluate a whole formulation; they do not isolate
feedback or auxiliary-loss strength. Varying that strength is outside this sweep.

Rate means probability of corrupting an episode, not a fraction of cells.
The mixed setting assigns conditional probabilities .5/.3/.2 to recent/gap/
outage. At rate .5, unconditional probabilities are .5 natural, .25 recent,
.15 gap, .10 outage. Recent removes 1–2 weeks of a channel or source family
across locations; gap removes 1–3 weeks of one channel/location; outage removes
one channel over the entire context and all locations. Natural missingness
always remains. Validation uses fixed masks drawn from the candidate's mixture;
mask contrasts therefore change training and checkpoint-selection conditions.

Fixed: seeds 42/43/44, all three season folds, select then refit, 12-week context,
width64, latent16, batch8, 128 training/256 validation members, 256 evaluation
trajectories, learning rate .001, native-unit scientific loss. References use
fourth-root admissions, logit ED, global noise, legacy decoder, geography,
dynamics and annual calendar. Independent models still see all six inputs.
There are 1,440 component/fold/seed combinations, each with selection and refit.
No additional transform/noise/head variants, cap contrasts, raw/rate admissions,
new data materialization or trend decoder are included. This first screen keeps
all four architectures and three seeds, reducing runs by 62% from the initial
104-configuration design. It estimates formulation effects at rate .5 and mask
effects mainly within B; other pipeline × rate/mechanism interactions are deferred.

## Evaluation and assumptions

Primary: natural-input location-relative WIS on the existing frozen 23-quantile
Hub support, equal seasons, admissions/ED weights 1/.5 and states/US 80%/20%.
`manager rank` gives the shared forecast ranking and separate nowcast ranking
against preliminary persistence. Never combine those two scores.
Each run also saves recent/gap/outage predictions, scored cells and
`stress-totals.csv`; these are diagnostics, not extra independent observations.
Inspect per-target/season/geography results and interval coverage before choosing
finalists. Natural score leads selection; stress robustness breaks close ties.
Use paired per-seed contrasts within each reference rather than averaging
unmatched architectures. Three seeds are a screen, not a significance claim.
Reevaluate finalists with more trajectories and seeds before resolving small gains.
The fixed A/B/C decisive-report command targets the older experiment names and
must not be used to report this grid.

This is retrospective development CV: older inputs are finalized and missing
recent vintage reports can receive supplied finals. The same seasons helped
select architectures. Neither B0 transfer nor performance on an untouched future
season is established. The 256-draw budget is a deliberate speed/precision tradeoff.

## Prepare and launch on Longleaf

Run in `/proj/jlessler/projects/tapestry-all/tapestry`. The suite is prepared
there with input checks and source snapshotting; the following plan command
recreates it. Do not replan while its jobs are running, because planning replaces
the experiment's source snapshot.

```bash
cd /proj/jlessler/projects/tapestry-all/tapestry
.venv/bin/python -m tapestry.models.manager plan -e B1-screen-256 \
  --suite B1-overnight --seeds 42 43 44 --eval-members 256 --retrospective --device cuda
mkdir -p output/slurm
LANES=8 GPUS=6 sbatch --job-name=B1-overnight-L40 --array=0-3 \
  --nodelist=g1803jles01 --time=12:00:00 \
  data/experiments/B1-screen-256/code/scripts/jlessler.sbatch B1-screen-256
LANES=8 GPUS=6 sbatch --job-name=B1-overnight-H100 --array=0-1 \
  --nodelist=g1803jles02 --time=12:00:00 \
  data/experiments/B1-screen-256/code/scripts/jlessler.sbatch B1-screen-256
.venv/bin/python -m tapestry.models.manager status -e B1-screen-256
.venv/bin/python -m tapestry.models.manager rank -e B1-screen-256 --allow-incomplete
# After all planned runs finish:
.venv/bin/python -m tapestry.models.manager rank -e B1-screen-256
```

Each allocation reserves one GPU, 12 CPUs and 110 GiB RAM, with eight concurrent
seed-run processes and one Torch thread per process. Four L40 allocations use
48 CPUs/440 GiB; two H100 allocations use 24 CPUs/220 GiB. A lane runs the three
season folds and component models sequentially. Across both arrays, the existing
NFS-locked dispatcher claims configuration/seed pairs from one queue: 48 lanes
maximum, no static architecture slices and no duplicate seed ownership.
The eight-lane setting has a B0 throughput benchmark; B1 memory and throughput
at this concurrency are assumptions to verify from the first allocations.
If memory failures occur, finish/cancel those allocations before relaunching with
fewer LANES and `--retry-failed`. Never use a CPU fallback.

The 12-hour limit defines the overnight window, not a completion guarantee.
B/C single-GPU seed42 runs took 25/35 minutes at 2,048 evaluation draws;
256 draws reduce evaluation cost, but sharing a GPU slows individual fits.
A rough planning estimate for the reduced screen is 3–5 hours across all six
GPUs, scaling the earlier unmeasured 8–12-hour estimate by 120/312. This is an
extrapolation assuming comparable average run costs, all allocations available,
and similar throughput; it is not a timing measurement or a guarantee.
Do not divide single-process duration by 48 to promise a finish time. Use observed
completed-run throughput to estimate the remaining time. Notifications are on
by default, one afterany summary per array, so one array's notification may
arrive before the other finishes.

After both arrays end, status and an incomplete rank show progress. To resume,
repeat the two launch commands (without replanning); add `--retry-failed` after
the experiment argument only when failed runs should be retried. Complete seed
runs are skipped; interrupted seed runs restart with fresh attempts. The
manager's printed resume command is also valid but defaults to L40 allocations;
use the explicit commands above to preserve the six-GPU, 12-hour layout.

## Log

- 2026-09-17: prepared the 104-configuration B0-informed B1 screen following the
  first successful supplied-final experiment. User requested preparation and
  commands only; no training/scoring jobs submitted.

- 2026-09-17: reduced the first screen to 40 configurations × three seeds at the
  user's request. Retained all four architecture references, A/B/C/two-stage,
  B masking rates and every masking mechanism, plus C's outage comparison.
  Dropped secondary architecture/cap panels and most pipeline/mask interactions.
  Replaced the unlaunched large plan with `B1-screen-256`; no jobs submitted.

- 2026-09-17: corrected the two-stage flag description against the frozen implementation; it does consume known-final flags.
