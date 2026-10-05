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
| `task=finalize` | Estimate final target/covariate values over a recent window ending at T-X; [selected seasonal model and commands](experiments/seasonal-nowcast-20261001/index.md) |
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

## Standard evaluation

Decided 5 October 2026. Every forecasting experiment is evaluated the same way, whatever
its training inputs. The rules below are applied by the code, not by each experiment.

**Inputs at evaluation.** A model trained on any inputs is evaluated on **Wednesday
reports**: each past week as it had been published by the submission deadline, for the
whole 12-week history (`input_mode='reported'`, `cv.score_episodes`). Each pathogen uses
**its own Hub's deadline**: flu forecasts the FluSight deadline, COVID the COVID Hub's,
RSV the RSV Hub's. Deadlines are Wednesday 23:00 US Eastern, except the holiday
extensions read from each Hub's Git history (see [Data](data/index.md#deadlines)). The
Wednesday and its newest data week (the Saturday before) never move; only the moment
at which data are read moves. Training episodes use the FluSight deadlines.

**Availability.** The schedule at the top of [Data](data/index.md) decides what a
model may see: all six targets through the newest week (T-0), **including ED in every
season** (the 2026–27 regime; in 2024–25 ED was actually published a week later),
ILINet, clinical labs and FluSurv one week later (T-1), Vermont inpatient claims never.
Where the schedule says a value is available but no report was archived by the
deadline, the model receives the finalized value. Every such cell is recorded, and
every figure in the report carries a ★ footnote with the shares. Values absent even
from the finalized data stay unavailable.

**Samples.** 512 forecast samples per forecast for every reported score
(`planner.EVAL_MEMBERS`; `plan` no longer takes `--eval-members`).

**Scores** (`evaluation/standard.py`, written by `planner rank` next to the ranking):

| Table | What it compares | Scales |
|---|---|---|
| `standard_hub_relative.csv` | Where a Hub ran: each location's total WIS divided by the Hub ensemble's on identical Hub tasks; states/DC averaged (80%) and US (20%) | natural; log(x + 1) for admissions |
| `standard_pairwise.csv` | Where a Hub ran: CDC FluSight 2025–26 method. States/DC only, models with at least 75% of the Hub's tasks, geometric mean of pairwise mean-WIS ratios on shared tasks, divided by the Hub baseline. Rank among all qualifying Hub models, with Google's models (`Google_*`) flagged; a Google model below 75% of the tasks is listed as not qualifying (2024–25 flu and COVID, 2025–26 COVID at 74.7%). Each seed joins the Hub pool alone. | natural; log(x + 1) for admissions |
| `standard_raw_wis.csv` | All six targets in both seasons, with or without a Hub: mean WIS per forecast task, states/DC and US separately, against the finalized panel. Window: every Saturday from the season's first to last Hub admissions reference date for that pathogen, otherwise CDC epiweeks 40–20. | natural; log(x + 1) for admissions |
| `standard_input_fills.csv` | Share of available inputs that received a finalized value, per season and target: its own history (newest week, all weeks) and all six target histories read at its Hub's deadline; covariates per Hub deadline | — |

ED proportions are never transformed. Lower is better everywhere; 1 is the ensemble
(Hub-relative) or the Hub baseline (pairwise). The main ranking score (`totals.py`,
natural scale, ensemble-relative, admissions 1 / ED 0.5) is unchanged and still orders
configurations. The report's "Standard evaluation" section shows C1–C3 and a ranking
figure per season (`standard-ranking-<season>.png`).

Assumptions: the CDC report does not state its log offset; we use log(x + 1), as the
FluSight and COVID dashboards do. Applying the FluSight pairwise method to the COVID and
RSV Hubs is our choice. Pairwise rankings use the frozen Hub truth; raw WIS uses the
finalized panel truth. Runs fitted before this change (or replaying old inputs with
`replay_from`, or two-stage pipelines, which nowcast their own dated inputs) are not
scored by the standard tables; `rank` says so and they must be refitted.

### Audit log, 5 October 2026

Scope: the new `evaluation/standard.py`, its input construction and report callers,
and the local frozen Hub tables. This was a code and data-support audit; no model
was trained, no checkpoint was evaluated, and no existing score was recomputed.
The items below are unresolved findings, not changes to the evaluation protocol.

- **Finalized input fills assume more than archive completeness.** `episodes`
  replaces any missing archived target/covariate value with the finalized panel
  value where the assumed schedule permits it. It cannot distinguish an archive
  gap from an observation that was genuinely unpublished. In particular, giving
  2024–25 ED inputs through T-0 implements the chosen 2026–27 availability regime,
  not the information available to historical Hub submissions. Even for a genuine
  archive gap, the finalized value need not equal the original report. Comparisons
  with submitted Hub forecasts therefore mix information regimes. A footnote
  measures exposure to this assumption; it does not remove the advantage. Keep
  this retrospective experiment distinct from an evaluation using archived inputs
  only, with unavailable cells masked. Prediction labels remain finalized truth.
- **The frozen support is not the full Hub task calendar.** Its manifest explicitly
  restricts tasks to complete ensemble quantiles, frozen Hub truth, and overlap
  with the old B0 forecasts. `pairwise` uses this restricted set both for the 75%
  qualification denominator and for scores. For 2025–26 RSV admissions, local
  support contains September 27 followed by November 22, with all intervening
  reference dates absent. `hub_windows` also imposes these holes on raw WIS,
  including ED scores against panel truth. Treat results as rankings on the frozen
  subset; establish an independent task calendar before claiming full-season or
  official eligibility. The observed Google COVID coverage of 74.71% in 2025–26
  is coverage of this subset, not verified full-season eligibility.
- **Nowcast-corrected forecast evaluation does not honor each Hub's cutoff.**
  `augmentation.prepare` computes corrections once from the main FluSight panel,
  then `evaluate_hubs` applies that same correction function to every Hub's
  episodes. For reference dates December 27, 2025 and January 3, 2026, the configured
  FluSight cutoff is one day later than the COVID/RSV cutoff. Those corrections
  can incorporate reports unavailable at the latter deadlines. Corrections need
  their own Hub-specific inputs and cache identity. This concerns evaluation inputs
  for the forecast model retrained with `reporting_augmentation=nowcast`; its
  held-out prediction labels are unchanged. Also, `corrected` preserves `filled`
  after replacing values, so the saved mask no longer describes the final input
  values. No affected run's score impact was measured in this audit.
- **Finality flags change meaning for vintage-trained models.** Reported episodes
  set `known_final=available`, even for provisional reports. A forecast model
  trained with `input_mode=vintaged` and `supplied_final=True` learned that recent
  reported values have a false finality flag, but receives true flags at evaluation.
  This is a second evaluation-input treatment beyond replacing the values. Preserve
  the trained meaning or identify and evaluate this flag intervention explicitly.
  Models with `supplied_final=False` do not consume this flag.
- **Raw WIS can silently average away invalid predictions.** Unlike the frozen-task
  path, `raw_wis` does not validate finite, ordered, in-range quantiles. Pandas mean
  drops NaN WIS while `tasks` still counts those rows; seed summaries also skip NaN.
  It does not enforce identical raw task keys and truth across runs. This matters
  especially for targets/seasons without frozen Hub checks. Require valid scores
  on common tasks before averaging. Log transformation also clips negatives to
  zero instead of rejecting them. The inspected frozen truth and quantile tables
  contain no negative values or missing truth; this finding is about unguarded
  run outputs, not an observed corruption of those frozen tables.
- **Input-fill reporting loses cross-pathogen exposure.** `evaluate_hubs` retains
  each target channel's fill mask only at that channel's own Hub deadline. A joint
  model sees all six channels at the output pathogen's deadline. The stored masks
  cannot recover those other-channel fills when deadlines differ, and `input_fills`
  filters each input channel by its own pathogen's scoring dates. Save and summarize
  all input channels separately for each forecast Hub and its scored origins.

Other interpretation limits: log(count + 1), unchanged ED proportions, 80% states/DC
and 20% US weighting, and applying FluSight's pairwise method to COVID/RSV are
explicit research choices. Raw WIS uses panel truth while Hub comparisons use frozen
Hub truth, so differences between the tables are not purely normalization effects.
Each seed/configuration creates a different pairwise comparison pool; averaging
Google's scores across those pools produces a summary, not one common-pool ranking.
The WIS implementation matches its documented interval-score formula on inspection.

**Resolution, 5 October 2026 (user review).** Kept as accepted research choices:
the shared nowcast correction across Hub deadlines (affects only
`reporting_augmentation=nowcast` evaluation, two holiday rounds), restricted frozen
support for the Hub-relative and pairwise tables (no official-eligibility claim), and
finalized fills with T-0 ED (a retrospective regime, not historical real-time
performance). Fixed:

- *Raw-WIS window.* Raw WIS scores outside the Hub, so its window is now every
  Saturday from the season's first to last Hub admissions reference date, not the
  frozen subset's dates (2025–26 RSV now includes September 27 – November 15).
- *Final flag.* Reported score episodes keep the training meaning of
  `known_final`: models trained with `input_mode=vintaged` see their newest
  `asof_weeks` weeks marked non-final, older weeks final; all other training modes
  marked every available cell final. Only models with `supplied_final=True` read it.
- *Fill accounting.* A nowcast that replaces a value clears its `filled` mark.
  Forecast files keep `filled_<hub>`/`available_<hub>` for all six target histories
  at each Hub's deadline; `input_fills` reports, per target, its own history and all
  six target histories read at its Hub's deadline (`all_targets_share`).

- *Raw-WIS guards.* Raw WIS now rejects duplicate tasks, non-finite, negative or
  decreasing quantiles, invalid truth and ED values above 1, and log scoring rejects
  negative values instead of clipping them. Every run must score the same tasks with
  the same truth (a hash of task keys and truth per target and season), or ranking
  stops.

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
