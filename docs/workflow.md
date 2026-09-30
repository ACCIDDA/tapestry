# Workflow

Use one panel, one scenario interface and one experiment manager for nowcasting,
forecasting and their composition. Run commands from the repository root.
[Environment and Longleaf setup](setup.md) · [Scenario fields](reference/scenario.md).

## Acquire and build

Training reads the saved panel; refresh sources only when creating a new dataset.

```bash
.venv/bin/python -m tapestry.data --data-root data pull delphi_nhsn delphi_nssp \
    delphi_claims_inpatient delphi_claims_outpatient delphi_nwss delphi_nwss_aux \
    hub_flusight_current hub_covid_current hub_rsv_current pophive_kinsa_ili \
    delphi_fluview_ilinet delphi_fluview_clinical delphi_flusurv
.venv/bin/python -m tapestry.dataset.build nwss-indices --data-root data
.venv/bin/python -m tapestry.dataset.build build --data-root data
.venv/bin/python -m tapestry.dataset.build show
.venv/bin/python -m tapestry.dataset.build check
.venv/bin/python -m tapestry.dataset.analyze_dataset
```

Source query options are documented under [Data](data/index.md). Wastewater index
construction is separate because it changes only when its input snapshots change.
Pin the reference resolution day with `build --truth-day`. A dataset rebuild
requires a new experiment name: planned runs verify the recorded input hashes.

## Choose the prediction task

| Setting | Meaning |
|---|---|
| `task=forecast` | Fit a standalone four-week forecaster |
| `task=nowcast` | Reconstruct recent completed weeks |
| `task=pipeline` | Fit independent nowcaster and forecaster with cross-fitted histories |
| `input_mode=finalized` | Use frozen reference histories; retrospective benchmark |
| `input_mode=scheduled_final` | Use final values subject to explicit source-lag assumptions |
| `input_mode=vintaged` | Use reported values at the issuance for the configured recent target window; covariates are as-of throughout |

For standalone vintaged models, `asof_weeks` controls the recent target window;
older target context uses reference values. Set it at least as large as `lookback`
for an entirely as-of standalone history. The coupled pipeline uses as-of inputs
throughout. Missing input values and missing labels have separate masks.

A scenario string contains comma-separated `key=value` fields. Omitted fields
use the current `Scenario` defaults. National Kinsa is broadcast with its reporting
mask; state covariates retain their native geography. Artificial training masking
is configured separately from actual reporting gaps and scheduled source lags.

## Plan, launch and resume

Example standalone forecasting experiment, using a new name:

```bash
.venv/bin/python -m tapestry.experiment.planner plan -e my-experiment \
    -s 'task=forecast,input_mode=scheduled_final,evaluation_seasons=recent_two' \
    --seeds 42 43 44 --device cuda
sbatch --job-name=my-experiment --array=0-3 scripts/jlessler.sbatch my-experiment
.venv/bin/python -m tapestry.experiment.planner status -e my-experiment
.venv/bin/python -m tapestry.experiment.planner rank -e my-experiment
```

For local execution, plan with `--device cpu` (or `mps`) and use
`.venv/bin/python -m tapestry.experiment.planner run -e my-experiment` as the launch.
`status` prints the exact resubmission command for unfinished work.
The shared GPU queue can fit multiple jobs per GPU. Notifications are enabled by
default; `NTFY=0` disables them and `NTFY_URL` selects the topic.

`plan` saves the scenarios, seeds, dataset/support hashes and a source snapshot
under `data/experiments/<name>/`. Cluster jobs use that pinned code. Replanning
repins it, so use `status` to resume an existing experiment. Local `run` uses the
working tree. A rebuilt panel or changed scientific protocol needs a new name.

## Analyze a run

**`planner rank` is the canonical analysis command.** It calls the shared Python
scorer, computes seed-paired matched effects, generates figures and writes the
report under `docs/experiments/<name>/`. No experiment-specific analysis wrapper
is needed. Default fans and heatmaps show the three best configurations, each
at its lowest seed for fans; heatmaps average seeds.

The ranking directory contains `season_scores.csv`, `season_composite_scores.csv`,
`run_scores.csv`, `configuration_ranking.csv`, `matched_pairs.csv`,
`matched_effects.csv`, a manifest and plots. The score is the location-relative WIS
ratio to the frozen Hub ensemble, with US 20%, equally weighted states/DC 80%,
admissions weight 1, ED weight 0.5, and equal season weights. Only available
support contributes. Lower is better; one means ensemble parity.

Matched effects change one scenario field at a time, holding every other field
fixed. Each context uses the same seeds on both sides, and contexts in a summary
use the same seed set. Deltas are averaged within each seed before a Student-t
interval is computed across seeds. Contexts are not independent replicates.
Intervals describe fitting randomness on fixed evaluation tasks, not uncertainty
for new seasons or multiple model selection. Fewer than two seeds give no interval.
Nowcasts are ranked separately by normalized CRPS, not Hub-relative forecast WIS.

`--allow-incomplete`, `--seeds`, and alternative score weights produce exploratory
rankings; only a complete default-weight ranking publishes a report. `--no-plots`
writes analysis tables without plotting or publishing. Use `rank --configs ...`
to choose report configurations or `plots --configs ... --dates ...` to redraw.

The report preserves text inside its write-up markers. Its adjacent `report.json`
records a stable report date, title and stage; the model-runs index and sidebar
order reports by that date, newest first. Dates describe the report, not necessarily
the last training job. Bring reports generated on Longleaf back with:

```bash
rsync -a chadi@longleaf.unc.edu:/proj/jlessler/projects/tapestry-all/tapestry/docs/experiments/ docs/experiments/
```

## Cross-validation

`dataset.cv.fold` controls fitting, early-stopping and scoring boundaries. Held-out
and unused reference weeks are masked before constructing fitting inputs, labels,
scales and covariate statistics. Validation weeks are hidden inside the fitting
partition; refitting uses all permitted training weeks. Scoring labels stay inside
the held-out season. The scenario selects evaluation seasons and the pinned panel
supplies the training calendar. These retrospective folds can train on seasons
later than the scored season; they are not forward deployment simulations.

## Two-stage models

One observation panel, independently fitted models, and an explicit history handoff.
Use `task=nowcast`, `task=forecast`, or `task=pipeline` through the same experiment
manager. No gradients flow between the stages.

### Module responsibilities

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

### Data and prediction boundary

The shared [data panel](data/index.md#current-dataset) supplies as-of inputs and separately frozen reference labels.

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

### Independent configuration

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

### Training and evaluation

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

### Explicit handoff

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

## One tuple of consecutive Saturday dates per episode; one Wednesday issuance per episode.
## values/available/known_final: [episode, context_week, 6, location].
## The named covariate object may contain the union needed by both stages.
covariates = CovariateHistory(covariate_values, covariate_names, locations,
                              context_dates, issuances)
with torch.no_grad():
    histories = reconstruct(
        nowcaster, values=values, available=available, known_final=known_final,
        locations=locations, context_dates=context_dates, issuances=issuances,
        covariates=covariates, members=256,
    )
    futures = forecast_histories(forecaster, histories, covariates=covariates)
## futures: [member, episode, 4, 6, location]
```

Use `None` for covariates when neither model needs them. A loaded combined
`model.pt` is callable with the same dated inputs as `reconstruct` and returns
future samples. The stages can also be loaded and used independently.



## Documentation scope

The documentation describes the current implementation. Completed forecasting
results remain as dated scientific reports; archived scientific reports have their own Archive folder. Retired workflow
instructions, preliminary analyses and smoke reports are excluded. Only `docs/experiments/` holds model-run
reports. Runtime artifacts under `data/experiments/` and scenario grids under
`experiments/` serve different purposes. Documentation organization does not change
saved models, raw data, score values or the two-stage model implementation.

## Report organization

Each experiment keeps its protocol on the report page. Reports begin with the
three best configurations (or all when fewer exist), season means versus the Hub
ensemble, and an explicitly identified B0 reference. A comparison across different
input/training protocols is contextual; it is not a paired feature effect.
Missing scores, season diagrams and other outputs are labelled **No known**.
The common order is season comparison, protocol, season splits, findings,
forecast fans, score diagnostics, matched comparisons, full ranking and appendix.
Original plot files are displayed in scrollable frames; layout changes do not
regenerate or alter plots. Archive dates mean first recorded publication in Git.

Only the main report for each experiment is retained. Supporting audits, checks,
reproductions and overview pages are excluded from the archive. The B0 comparison
uses the saved stage09 season scores in `docs/experiments/b0-reference.csv`;
keeping these numeric reference values does not require a separate report page.
