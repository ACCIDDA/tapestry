# Run the Build B0 pilot

B0 is a small PyTorch model for the finalized six-channel dataset. `train` and
`predict` fit and sample one model; season cross-validation is described below.
Use the repository root as the working directory and run `uv sync --upgrade-package epibenchmark`
first; see [environment setup](../getting-started.md).

Train with an explicit last permitted training-label date:

```bash
uv run python -m tapestry.models train \
  --train-end 2024-07-27 --epochs 50 \
  --output data/processed/b0.pt
```

Generate samples and quantiles after fitting:

```bash
uv run python -m tapestry.models predict \
  --checkpoint data/processed/b0.pt --context-end 2024-08-03 \
  --members 256 --output data/processed/b0_predictions.npz
```

Defaults are eight history weeks, four future weeks, all six channels, 52 native
locations, CPU, and seed 42. `--lookback 12` changes the training history length;
`--horizons 1 2 3 4 5 6 7 8` trains eight future weeks. Prediction reads those
settings from the checkpoint. `--locations NY NJ` restricts prediction locations.
`--device mps` or `--device cuda` is optional.
There is no implied train/test assignment: choose training dates explicitly.
Avoid comparing in-sample predictions as evidence of generalization.

Inspect the output:

```python
import numpy as np
p = np.load('data/processed/b0_predictions.npz', allow_pickle=False)
print(p['samples'].shape)    # (256, 4, 6, 52): member, horizon, channel, location
print(p['quantiles'].shape)  # (5, 4, 6, 52)
print(p['target_dates'], p['channels'], p['locations'])
```

Samples retain their member identity across channels, locations, and horizons.
Quantiles are in native units: nonnegative real-valued admission counts and ED
proportions between zero and one. This is a research NPZ, not a Hub submission;
admission quantiles have not been rounded to integers or exported to Hub schema.

## Model and deliberate shortcuts

B0 uses one shared context MLP and one shared focal-history MLP across locations
and channels, source embeddings, horizon offsets, and sine/cosine calendar
features. A single 16-dimensional Gaussian draw per member/episode modulates
its shared decoder through an affine scale and shift. The decoder predicts a
residual around the latest valid focal observation, using softplus for admissions
and sigmoid for ED proportions. Missing anchors use a small fixed starting prior.
Default width is 64, totaling 22,785 parameters. B0 has no spatial attention or
location embedding. Cross-channel context is included; cross-location information
exchange is deferred to B1.

Input transforms are fitted per channel/location using training-only history.
The loss is **fair CRPS on untransformed predictions**, divided by a separate
native-unit channel/location Q95. Admission/ED weights default to
`[1,1,1,.5,.5,.5]`; seasons are averaged equally after combining available targets
within each season. States/DC share 80% equally and native US has 20%. Sparse
loss scales pool toward the channel Q95 below 26 observed weeks, with floors of
1 admission and .001 ED proportion. See
[design choices for loss](../design/architecture.md#82-design-choices-for-b0-loss-and-weights).
This training normalization is a surrogate for the location-relative ensemble
WIS used in ranking, not the same denominator. Missing labels are masked before
arithmetic; observed zeros remain eligible. Whole-partition cell weights preserve
the objective across minibatches. Eight training draws are the direct CLI default;
the experiment recipes use 128. Sorted samples avoid a quadratic pairwise tensor.

Training uses September 2023 onward by default. Context origins, training labels,
and scaling statistics stop at `--train-end`; labels beyond it are masked even
for windows that start earlier. Earlier history may be used if a later
`--train-start` is selected. The checkpoint includes the model, fitted scales,
training dates, seed, dataset hash, channel/location registry, and loss history.
This is a finalized-data retrospective experiment, with the NSSP finality and
geography assumptions of the [dataset contract](../data/build-b-finalized.md).

The [named experiment manager](experiment-manager.md) organizes scenarios,
three-seed comparisons, and resume of completed runs. Early stopping is optional
(`--patience`, below); there is no calibration, ensemble, optimizer resume, or
performance claim.
Training minibatches are shuffled within
the explicitly bounded fitting period; there is no random train/test split.

## Experiment switches

Defaults give the baseline B0 behavior. The
[architecture sweep](experiment-manager.md#architecture-sweep) crosses these switches.

| Switch | Values / behavior |
|---|---|
| `--count-transform` | `raw` (baseline counts), or the rate per 100,000 as `rate`, `sqrt`, `fourth_root`, `log1p` |
| `--ed-transform` | `linear` (original: scaled proportions in, logit residual out), `logit` (centered logit in and out), `fourth_root` (in and out) |
| `--geography` | Log(population / 100000) and native-US indicator |
| `--lookback` | Compare `8`, `12`, `26`; same MLP architecture and width |
| `--dynamics` | Recent slope, change in slope, observation age, validity flags, Christmas timing |
| `--loss-weights` | `influenza_first`: `[1,.1,.1,.1,.1,.1]`; `balanced_admissions`: `[1,1,1,.1,.1,.1]`; `flu_only`: `[1,0,0,0,0,0]`; `objective`: `[1,1,1,.5,.5,.5]`, the adopted target coefficients (training uses Q95 normalization; selection uses ensemble WIS ratios) |
| `--population-file` | Default frozen `data/metadata/locations.csv`; custom CSV uses `location,population`, or `abbreviation` if present |
| `--encoder` | `mlp` (default) or `conv`: two shared temporal convolutions |
| `--heads` | `shared` (default) or `state_us`: separate modulation/output parameters |
| `--decoder` | `legacy` (default) or `residual2`: two latent-modulated residual blocks |
| `--latent` | Default `16`; compare `32` independently of decoder depth |
| `--spatial` | `none` (default) or `attention`: one attention block across locations |
| `--noise` | `global` (default) or `local`: adds a per-location latent |
| `--patience` | Season CV only. `0` (default) trains `--epochs` epochs; otherwise early stopping with `--epochs` as the cap |

All modes retain all six input channels and six output heads. Flu-only supervision
zeros the auxiliary loss contributions, so auxiliary forecasts from that mode are
unsupervised and should not be interpreted as trained forecasts.

For the population variants, admissions become rates per 100,000, then receive the
selected power transform and training-context Q95 scaling. The decoder inverts
both operations to admission counts **before** fair CRPS. ED remains proportional
with a bounded decoder. Native-unit channel/location loss Q95 scales are fitted separately
and stay identical across representation and loss-weight variants on a given fold.
Counts are not centered, preserving a simple nonnegative transformed residual anchor.
The transform also sets where uncertainty acts: residuals and latent perturbations
are added in transformed space, so `rate` spreads equally in absolute terms at every
level, `sqrt` roughly like Poisson variation, `fourth_root` between those, and `log1p`
proportionally. `logit` ED inputs are centered on the training mean and divided by
the training SD; `fourth_root` ED uses the count softplus residual and is capped at one.
Input scale fitting excludes held-out observations; direct training also excludes
pre-training context dates from scale fitting. Scales are saved in model buffers.

Assumptions: population denominators are fixed across retrospective seasons from
the frozen local FluSight table; source path, repository revision, and file hash are
in `data/metadata/locations.provenance.json`. The actual population mapping is
saved in the checkpoint, so prediction does not reread the CSV. Native US stays
separate from states. Geography and transform switches are independent.

Dynamics use adjacent observations in the final three calendar weeks in transformed,
scaled input units. Gaps invalidate the corresponding slope or acceleration and
are exposed through flags. Observation age is elapsed weeks since the latest valid
input divided by lookback; no observed history uses one. Christmas timing is signed
weeks relative to December 25 in the July–June winter containing the origin,
divided by 26. These features use only the context. Observation age describes
missing observations, not release latency in this finalized-data panel.

Spatial attention applies one pre-norm, four-head attention block with a
feed-forward layer to the per-location context embeddings of each episode (50
states, DC, and US). Each token summarizes only context observed by the forecast
date. Location identity comes from the geography features, with no learned
per-location embedding, so outputs follow any location order.

Local noise adds a four-dimensional latent per member, episode, and location,
shared across that location's horizons and channels. A linear map turns it into
extra scale and shift in every decoder modulation, multiplied by a learned
nonnegative magnitude (softplus, initialized at one). Each fold's `training.json`
records the learned magnitude. New modules are created after the original ones,
so global-noise, non-spatial configurations keep their original initialization.

Early stopping hides three consecutive target weeks out of every sixteen inside the
two training seasons of each fold (weeks 4–6, 20–22, and 36–38 of a season: start,
winter, and spring; 18–19 weeks per fold). Hidden weeks are removed from the inner
fit's context, labels, and scales, which keeps 80–85 of about 100 training windows;
the refit uses all of them. Validation episodes are the origins with a hidden week
among their targets, scored only on hidden weeks, so each hidden week is predicted
at all four horizons. After every epoch, the season/target/location-weighted normalized fair CRPS on validation
episodes uses fixed draws (32 members), including global, local, and shared-factor
noise. Validation does not consume the training RNG.
Training stops after `--patience` epochs without improvement, restoring the best
epoch; the model is then refit on all training weeks for that many epochs. The inner
model's validation forecasts are saved as `validation_forecasts.npz` and
`validation_scores.csv` for later calibration.

Example three-season experiment (choose a fresh output directory):

```bash
uv run python -m tapestry.models.season_cv \
  --count-transform sqrt --geography --lookback 8 \
  --output data/experiments/b0_sqrt_geo_8
```

Compare this with `fourth_root` and the unchanged raw baseline. Next vary only
lookback across 8/12/26, then toggle `--dynamics`. Finally vary `--loss-weights`
while keeping the selected representation and history fixed. Use the same seeds,
fit/evaluation dates, and ensemble-supported scoring sets. These same
switches are accepted by `python -m tapestry.models train`; prediction reads
all feature settings from its checkpoint and supports reordered location subsets.
The CV runner reports states/DC and native US separately.

## Saved three-season CV forecasts and evaluation

Season folds are 2023–24, 2024–25 and 2025–26 (CDC epiweeks 31–30).
Every fold excludes the held-out season from fitting contexts, labels and scales.
Evaluation conditions on already observed past context, including within that
season, and scores only target weeks in the held-out season. Weekly origins use
four future leads; the default history is eight weeks. Forecast files retain the
hub's 23 quantiles from 2,048 draws and 100 complete sample members per origin. Admissions
are rounded half-up for the CV export; ED values remain proportions.

Only training on 2023–24 and 2024–25 to evaluate 2025–26 is chronological. The
other folds train on later seasons. All inputs are finalized and all folds have
been examined during exploratory selection; none is an untouched final test set.

Each fold saves its checkpoint, forecasts, training manifest, and a diagnostic
`scores.csv`. Those diagnostics are not used for ranking: the manager scores each
run against the hub ensembles into `totals.csv` ([ranking](experiment-manager.md#ranking)),
and the [full EpiBench evaluation](configuration-evaluation.md) remains available
for shortlisted runs.
