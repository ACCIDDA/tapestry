# Nowcasting and forecasting

One observation panel, independently fitted models, and an explicit history handoff.
Use `task=nowcast`, `task=forecast`, or `task=pipeline` through the same experiment
manager. No gradients flow between the stages.

## Module responsibilities

| Module | Responsibility |
|---|---|
| `dataset/episodes.py`, `dataset/cv.py` | Dates, as-of observations, labels and scientific partitions |
| `model/network.py` | Shared network architecture and checkpoint serialization |
| `model/scenario.py` | Configuration parsing and independent stage settings |
| `model/pipeline.py` | Dated history/covariate objects and model composition |
| `experiment/training.py` | Tensor preparation, preprocessing, fitting and evaluation |
| `experiment/two_stage.py` | Cross-fitting and composing the two training stages |
| `experiment/fitting.py` | Select the standalone or coupled fold workflow |
| `experiment/planner.py` | Plan jobs, launch, track completion and rank results |

Dependencies go from planner to fitting/orchestration to shared training and models.
Training and pipeline modules do not import the planner.

## Data and prediction boundary

`data/processed/panel.npz` remains the shared source. The 2026-09-22 build contains
160 Saturday weeks (2023-09-02–2026-09-19), 52 locations (states, DC, native US),
six targets, 13 state covariates, one US-only Kinsa covariate and 160 Wednesday
issuances. Each series has a reference version resolved on 2026-09-22 and as-of
versions. Metadata records the 12 raw source snapshots. The frozen reference
snapshot supplies labels; it is not a guarantee of permanent finality.

Let `t` be the Saturday four days before the Wednesday issuance.

| | Nowcaster | Forecaster |
|---|---|---|
| Inputs | As-of target reports and selected covariates | Sampled reconstructed histories and optional as-of covariates |
| Labels | Reference outcomes for completed weeks `t-R+1` through `t` | Reference outcomes for `t+1` through `t+4` |
| Outputs | Joint samples of all six targets for the reconstruction window | Joint samples of all six targets for four future weeks |

Defaults are `R=2` and four forecast weeks. Both stages use entirely as-of history
in the coupled pipeline. Older weeks pass through with their original missingness;
the last `R` weeks are reconstructed, including provisional reports and unpublished
cells. Two weeks is a configurable research choice, not a claim that older reports
are mature. Kinsa is stored nationally and broadcast to every location during episode construction.
Covariates are predictors, not additional reconstruction targets.

## Independent configuration

A pipeline has ordinary shared defaults plus `nowcast.<field>` and
`forecast.<field>` overrides. Every override is validated as an ordinary model
setting. Python uses mappings or immutable key/value pairs:

```python
from tapestry.model import Scenario

scenario = Scenario(
    task="pipeline", input_mode="vintaged",
    nowcast={"lookback": 12, "encoder": "mlp", "epochs": 50,
             "covariate_set": "inpatient+ilinet"},
    forecast={"lookback": 8, "encoder": "conv", "epochs": 100,
              "patience": 5},
)
nowcast_settings = scenario.stage("nowcast")
forecast_settings = scenario.stage("forecast")
print(scenario.scenario_string)
```

The corresponding string uses comma-separated fields such as
`task=pipeline,input_mode=vintaged,nowcast.lookback=12,forecast.lookback=8`.
Each stage can independently choose architecture, target grouping, covariates,
transforms, epoch budget, early stopping and missingness augmentation. Unprefixed
model/training settings supply defaults to both stages. `epochs` and `patience`
have the same meaning in each stage: select an epoch if requested, then refit.
`nowcast_weeks` controls the reconstruction window; `nowcast_members=16` controls
the finite history bank used for forecaster training. These are pipeline settings,
not stage overrides.

A longer input history covers `max(nowcast.lookback, forecast.lookback)` weeks.
Each stage consumes its own trailing window. The nowcaster must have at least
`nowcast_weeks` context weeks. Locations and target order must match between stages.

The earlier `nowcast_epochs` and `forecast_covariate_set` fields are removed.
Use `nowcast.epochs` and `forecast.covariate_set`. Standalone settings remain flat.
A standalone finalized forecaster remains an explicit retrospective benchmark.

## Training and evaluation

1. For each outer fold, mask all held-out and unused reference weeks before
   constructing fitting inputs, labels, loss scales or covariate statistics.
2. Within the permitted training partition, fit a nowcaster for each prediction
   season using other seasons only. Also exclude reconstructed and forecast label
   dates crossing its boundaries. Generate joint reconstructed histories.
3. Train the forecaster on these out-of-fold histories. If it uses early stopping,
   repeat cross-fitting inside its inner fitting partition. Its validation weeks
   cannot train any nowcaster used for that selection.
4. Fit the final nowcaster on the outer fitting partition and refit the forecaster
   after epoch selection. Evaluate their composition on the held-out season.

Nowcaster early stopping also stays inside its permitted fitting partition.
When both stages use early stopping, their validation schedules must leave
nowcaster validation labels after forecaster validation weeks have been excluded.
For example, use `nowcast.validation_offset=8` and `forecast.validation_offset=4`.
An empty fitting/validation partition raises an error; no validation fallback is used.

Training samples a whole history independently from the cached bank for each
predictive member, then draws one conditional future. This finite bank is a
Monte Carlo approximation. Evaluation draws fresh histories and conditional
futures. Never substitute the nowcast mean or shuffle its cells independently.
Native-unit fair CRPS and the existing season/target/location weighting remain
unchanged. Estimated cells never become reference-truth observations for scales.

Each pipeline fold saves `nowcaster.pt`, `forecaster.pt` and combined `model.pt`,
with resolved stage settings and fitting provenance in the manifest. Forecast
quantiles remain in `forecasts.npz` for hub-relative WIS. Separate nowcast quantiles
and normalized fair CRPS are in `nowcast/`. Standalone nowcasts place these at the
fold root, and `rank` writes `nowcast-ranking.csv`. Rank standalone nowcasts and
forecasts in separate experiments because they measure different tasks.

Season-held-out evaluation is retrospective. Operational historical performance
also requires chronological fitting and training labels available at that cutoff.
Recent reference labels may be immature. Propagating samples does not establish
calibrated dependence or interval coverage. An oracle-history benchmark is a
separate finalized forecast experiment; it is not fitted automatically.

## Explicit handoff

`HistorySamples` holds native-unit samples `[member, episode, week, target, location]`,
boolean availability/finality/estimated masks, target/location identifiers, context
dates, issuances and optional provenance. Counts remain counts; ED values remain
0–1 proportions. Calendar features are derived from context dates; any supplied
calendar must agree. Estimated and known-final masks cannot overlap.

`CovariateHistory` holds values `[episode, week, covariate, value_or_mask, location]`
with names, locations, dates and issuances. Names are reordered to each stage's
configuration. Date or issuance mismatches, wrong location order, missing covariates,
short histories and mismatched mask shapes raise errors. Raw unnamed covariate
tensors are not accepted at the public pipeline handoff. These checks establish
alignment, not the authenticity of an external caller's claimed as-of cutoff.

```python
import torch
from tapestry.model.network import load_model
from tapestry.model import CovariateHistory, reconstruct, forecast_histories

nowcaster = load_model(torch.load("nowcaster.pt", map_location="cpu", weights_only=False)).eval()
forecaster = load_model(torch.load("forecaster.pt", map_location="cpu", weights_only=False)).eval()

# One tuple of consecutive Saturday dates per episode; one Wednesday issuance per episode.
# values/available/known_final: [episode, context_week, 6, location].
# The named covariate object may contain the union needed by both stages.
covariates = CovariateHistory(covariate_values, covariate_names, locations,
                              context_dates, issuances)
with torch.no_grad():
    histories = reconstruct(
        nowcaster, values=values, available=available, known_final=known_final,
        locations=locations, context_dates=context_dates, issuances=issuances,
        covariates=covariates, members=256,
    )
    futures = forecast_histories(forecaster, histories, covariates=covariates)
# futures: [member, episode, 4, 6, location]
```

Use `None` for covariates when neither model needs them. A loaded combined
`model.pt` is callable with the same dated inputs as `reconstruct` and returns
future samples. The stages can also be loaded and used independently.

## Manager commands

Small local validation run with different encoders/lookbacks and early stopping
in both stages:

```bash
.venv/bin/python -m tapestry.experiment.planner plan -e stage-cleanup-smoke -s 'task=pipeline,input_mode=vintaged,width=8,latent=4,epochs=2,members=2,validation_members=2,nowcast_members=2,nowcast.lookback=6,nowcast.patience=1,nowcast.validation_offset=8,nowcast.covariate_set=inpatient,forecast.lookback=4,forecast.encoder=conv,forecast.patience=1,forecast.covariate_set=ilinet' --seeds 42 --eval-members 8 --device cpu
.venv/bin/python -m tapestry.experiment.planner run -e stage-cleanup-smoke
.venv/bin/python -m tapestry.experiment.planner status -e stage-cleanup-smoke
.venv/bin/python -m tapestry.experiment.planner rank -e stage-cleanup-smoke
```

Example full-budget Longleaf experiment (not launched by this cleanup):

```bash
.venv/bin/python -m tapestry.experiment.planner plan -e independent-stages -s 'task=pipeline,input_mode=vintaged,nowcast.covariate_set=inpatient+outpatient+ww_wval_like+ww_pct_rank+ilinet+clinical_lab+flusurv+kinsa,nowcast.lookback=12,forecast.lookback=8' --seeds 42 43 44 --device cuda
sbatch --job-name=independent-stages --array=0-3 scripts/jlessler.sbatch independent-stages
.venv/bin/python -m tapestry.experiment.planner status -e independent-stages
.venv/bin/python -m tapestry.experiment.planner rank -e independent-stages
```

## Decision log

- 2026-09-23: implemented separate weights/checkpoints, cross-fitted training and
  sampled-history composition from one panel. Vintage inputs without finality
  flags default to unknown finality. Initial real-panel validation completed all
  three seasons; those small fits established execution, not forecast skill.
- 2026-09-24: moved shared training out of the planner; added resolved stage
  overrides, separate architecture/lookback settings and stage-local early stopping;
  required dated, named handoffs. Replaced the first pipeline configuration fields
  rather than maintaining a compatibility layer. Scientific label dates, native
  units, objective weights and outer season partitions are unchanged.


## Cleanup validation (2026-09-24)

58 focused checks passed for leakage, masks, date/covariate alignment, objective
mathematics and configuration. `stage-cleanup-smoke` completed all three held-out
seasons and ranked successfully on its first attempt. It used an MLP nowcaster
with six history weeks and a convolutional forecaster with four, independent
covariate groups, and separate early-stopping schedules. Reloaded combined and
separate checkpoints produced exactly equal samples with a fixed seed, including
when covariate columns were supplied in reverse name order.

This is an execution check with two training epochs, not a model-selection result.
The forecast plots and scores are in
`data/experiments/stage-cleanup-smoke/ranking-84f8a6c9fa2f/`; the generated report
is [stage cleanup validation](../results/stage-cleanup-smoke/index.md).


## Current experiment

The [forecasting covariate comparison](forecast-covariates.md) uses fresh fits.
Nowcasting is trained separately.
