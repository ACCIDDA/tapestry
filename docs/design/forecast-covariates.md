# Geographic forecasting and covariate representations

Experiment `forecast-geography-v2` trains standalone forecasters from scratch.
Nowcasting stays separate. Each model receives finalized target/covariate values
only where a report existed by the historical Wednesday cutoff. Labels are
finalized truth. No old checkpoint or calibration is reused.

## Matched comparison

Three geographic designs each get five covariate bundles × four representations,
plus their own no-covariate control: **63 scenarios × three seeds = 189 runs**.
Every run has three outer season folds and three independent pathogen models;
early stopping adds inner selection fits.

| Geography (`spatial`) | Information shared across locations |
| --- | --- |
| `none` | Independent location contexts, with the national Kinsa signal broadcast when selected |
| `pooled` | Equal-weight mean of observed state/DC contexts plus the native US context, projected into every location |
| `attention` | Learned attention across all 52 location contexts, including their selected covariates |

The pooled model averages learned context vectors, not raw counts or proportions.
US is excluded from the state average and represented separately, avoiding double
counting. A location contributes if any target or selected covariate input is
observed. With no observed states the pooled state vector is zero; an unobserved
US vector is zero. A single linear 128→64 projection adds this common context to
every location. Attention uses the existing four-head spatial block after the
complete target/covariate context encoder. Geographic inputs (log population and
native-US flag) remain enabled in every arm. Neither architecture assumes distance
or adjacency; these test nationwide information sharing, not local diffusion.

| Covariate bundle | Sources | Channels |
| --- | --- | ---: |
| Claims | Inpatient/outpatient flu and COVID | 4 |
| New flu surveillance | ILINet ILI, clinical-lab flu % positive, FluSurv hospitalization rate | 3 |
| Mixed | Claims + new flu surveillance + wastewater WVAL-like indices + Kinsa | 11 |
| Wastewater | WVAL-like indices for flu/COVID/RSV | 3 |
| Kinsa | National cough/cold/flu | 1 |

**National Kinsa values and their availability are broadcast to all states, DC,
and US before fitting-fold standardization.** They remain one national series;
broadcasting does not invent state measurements. This applies to all input modes,
all representations and all geographic designs. It also works when cutting
inputs for a location subset without a US row. The on-disk panel is unchanged.
External callers supplying prepared `CovariateHistory` should likewise broadcast
national covariates; they must use the same input convention as training.

| Representation | Features per covariate |
| --- | --- |
| `raw` | Standardized values and masks for all 12 weeks: 24 |
| `smooth` | Signed-log standardized values, trailing-three-week observed mean and aggregate availability: 24 |
| `summary` | Last value, recent-three-week mean, observed-time slope, SD, coverage, latest-report age: 6 |
| `shared` | Shared 24→8→4 MLP on values/masks, plus coverage and latest-report age: 6 |

All non-raw arms use `sign(z) * log1p(abs(z))` after fitting-fold standardization.
These compare representation packages: they do not isolate transformation from
compression. Shared weights operate across sources and locations within each
pathogen model; concatenation preserves source identity. Smoothing reduces noise
without reducing parameters. Fitted manifests record `parameter_count`.

## Backbone and training

Use the past successful pathogen-specific MLP design. Its completed historical
three-seed comparison had relative WIS 0.9407; the exact configuration is in
`docs/legacy-v0/results/b1-overnight/epoch300/configuration-ranking.csv`.
That score is motivation for the architecture, not a comparable result under the
new protocol.

Common settings: 12 weeks, MLP width 64, latent 16, fourth-root admission rates,
logit ED, calendar/geography/dynamics, shared heads, finality flag, mixed artificial
target masking on 50% of training episodes, 300-epoch cap, patience 30. Batch size
8; 128 training members, 256 validation/evaluation members; Adam learning rate
0.001, no weight decay, objective loss weighting. Seeds 42, 43, 44. Selected epoch
counts are refitted on the full training split. All six target histories are
inputs to each independently fitted pathogen model. Covariates are not artificially
masked. Each geographic design has a fresh no-covariate control so improvements
can be separated from benefits of geographic modeling alone.

## Scientific assumptions

`input_mode=finalized_available` uses final values resolved at 2026-09-22 but masks
inputs with no report by 23:59:59.999999 UTC Wednesday. Later revisions are allowed;
validation and scoring use this same retrospective protocol, not historical
real-time values. Context ends the preceding Saturday, with the next four
Saturdays as forecast targets. Three leave-one-season-out folds use common frozen
Hub support. Held-out/validation reference weeks are masked before input creation
or scaling, including national covariates; geographic sharing cannot restore them.

Statistics use observed fitting cells only. Smoothing is causal within each
history and can bridge a gap when a report exists in the trailing three weeks.
Summary slope/SD use observed times; a singleton has zero slope/SD. No-report
histories have zero value features/coverage and age one; age is divided by the
12-week lookback. National replication supplies information rather than extra
labels or score weight. The pooled state mean is equal-weight, not population-
weighted. The second wastewater index remains outside this screen.

## Manager commands on Longleaf

From `/proj/jlessler/projects/tapestry-all/tapestry`:

```bash
# Plan only before launching; do not re-plan while workers are active.
bash experiments/forecast-covariates.sh
mkdir -p output/slurm
GPUS=6 sbatch --job-name=forecast-geography-v2 --array=0-3 scripts/jlessler.sbatch forecast-geography-v2
GPUS=6 sbatch --job-name=forecast-geography-v2-h100 --array=0-1 --nodelist=g1803jles02 scripts/jlessler.sbatch forecast-geography-v2
.venv/bin/python -m tapestry.experiment.planner status -e forecast-geography-v2
.venv/bin/python -m tapestry.experiment.planner rank -e forecast-geography-v2
```

Four L40 and two H100 allocations use the shared queue with eight fitting workers
per GPU. `squeue -a -u chadi` includes the hidden patron partition. `status` gives
resubmission commands. Completion notifications are enabled.

## Decision log

2026-09-24: The initial nonspatial run was stopped at the user's request to make
Kinsa national inputs available everywhere and compare geographic architectures.
Arrays 2393590/2393591 and their pending notifications were cancelled. The v1
snapshot/results remain separate; v2 starts fresh and never mixes the two input
protocols. The mixed bundle now includes Kinsa. The panel itself is unchanged.

Validation: 42 focused leakage, missingness, alignment and pooling checks passed.
All 63 configurations produced finite forward outputs and gradients on real-panel
episodes; checkpoint reload reproduced outputs. This validates execution, not skill.

Submitted on Longleaf: array `2393992` (four L40s, `g1803jles01`) and array
`2393993` (two H100s, `g1803jles02`), partition `jlessler`, 189 shared-queue runs.
