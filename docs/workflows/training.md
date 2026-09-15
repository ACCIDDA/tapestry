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

By default, values are divided by each channel's training-only 95th percentile.
Population transforms and geographic/dynamics features are optional experiment
switches (below). No
learned zero-history prior or separate observation-noise layer is added. The loss is
masked fair CRPS in native units divided by each channel's fixed training scale,
with weights `[1,.1,.1,.1,.1,.1]` so influenza admissions are primary. Dividing the
primary count score by its fixed scale changes its numerical magnitude, not its
single-target optimum. These are explicit pilot choices, not tuned findings.
Eight independent training members are the default. Masks exclude missing labels
before arithmetic and retain observed zeros. Samples are sorted to compute fair
CRPS without a quadratic pairwise member tensor.

Training uses September 2023 onward by default. Context origins, training labels,
and scaling statistics stop at `--train-end`; labels beyond it are masked even
for windows that start earlier. Earlier history may be used if a later
`--train-start` is selected. The checkpoint includes the model, fitted scales,
training dates, seed, dataset hash, channel/location registry, and loss history.
This is a finalized-data retrospective experiment, with the NSSP finality and
geography assumptions of the [dataset contract](../data/build-b-finalized.md).

The [named experiment manager](experiment-manager.md) organizes scenarios,
three-seed comparisons, and resume of completed runs. There is no scheduler,
early stopping, calibration, ensemble, optimizer resume, or performance claim.
Training minibatches are shuffled within
the explicitly bounded fitting period; there is no random train/test split.

## Experiment switches

Defaults give the baseline B0 behavior. Spatial attention is a B1 proposal.
Comparisons are in the [canonical B0 results](../results/b0-configuration-comparison.md).

| Switch | Values / behavior |
|---|---|
| `--count-transform` | `raw` (baseline), `sqrt`, `fourth_root` |
| `--geography` | Log(population / 100000) and native-US indicator |
| `--lookback` | Compare `8`, `12`, `26`; same MLP architecture and width |
| `--dynamics` | Recent slope, change in slope, observation age, validity flags, Christmas timing |
| `--loss-weights` | `influenza_first`: `[1,.1,.1,.1,.1,.1]`; `balanced_admissions`: `[1,1,1,.1,.1,.1]`; `flu_only`: `[1,0,0,0,0,0]` |
| `--population-file` | Default frozen `data/metadata/b0_locations.csv`; custom CSV uses `location,population`, or `abbreviation` if present |
| `--encoder` | `mlp` (default) or `conv`: two shared temporal convolutions |
| `--heads` | `shared` (default) or `state_us`: separate modulation/output parameters |
| `--decoder` | `legacy` (default) or `residual2`: two latent-modulated residual blocks |
| `--latent` | Default `16`; compare `32` independently of decoder depth |

All modes retain all six input channels and six output heads. Flu-only supervision
zeros the auxiliary loss contributions, so auxiliary forecasts from that mode are
unsupervised and should not be interpreted as trained forecasts.

For the population variants, admissions become rates per 100,000, then receive the
selected power transform and training-context Q95 scaling. The decoder inverts
both operations to admission counts **before** fair CRPS. ED remains proportional
with a bounded sigmoid decoder. Native-unit loss Q95 scales are fitted separately
and stay identical across representation and loss-weight variants on a given fold.
No centering is used, preserving a simple nonnegative transformed residual anchor.
Input scale fitting excludes held-out observations; direct training also excludes
pre-training context dates from scale fitting. Scales are saved in model buffers.

Assumptions: population denominators are fixed across retrospective seasons from
the frozen local FluSight table; source path, repository revision, and file hash are
in `data/metadata/b0_locations.provenance.json`. The actual population mapping is
saved in the checkpoint, so prediction does not reread the CSV. Native US stays
separate from states. Geography and transform switches are independent.

Dynamics use adjacent observations in the final three calendar weeks in transformed,
scaled input units. Gaps invalidate the corresponding slope or acceleration and
are exposed through flags. Observation age is elapsed weeks since the latest valid
input divided by lookback; no observed history uses one. Christmas timing is signed
weeks relative to December 25 in the July–June winter containing the origin,
divided by 26. These features use only the context. Observation age describes
missing observations, not release latency in this finalized-data panel.

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
The CV runner reports states/DC and native US separately. See
[the canonical report](../results/b0-configuration-comparison.md) for the
14-variant, three-seed comparison.

## Saved three-season CV forecasts and evaluation

Season folds are 2023–24, 2024–25 and 2025–26 (CDC epiweeks 31–30).
Every fold excludes the held-out season from fitting contexts, labels and scales.
Evaluation conditions on already observed past context, including within that
season, and scores only target weeks in the held-out season. Weekly origins use
four future leads; the default history is eight weeks. Forecast files retain five
quantiles (0.025, 0.25, 0.5, 0.75, 0.975) from 2,048 draws and 100 complete sample members per origin. Admissions
are rounded half-up for the CV export; ED values remain proportions.

Only training on 2023–24 and 2024–25 to evaluate 2025–26 is chronological. The
other folds train on later seasons. All inputs are finalized and all folds have
been examined during exploratory selection; none is an untouched final test set.

Each fold saves its checkpoint, forecasts, training manifest, and a diagnostic
`scores.csv`. Those Python diagnostics are not used in the report or configuration
ranking. Use the [full EpiBench evaluation](configuration-evaluation.md) on the
saved forecasts.
