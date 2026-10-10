# Multi-pathogen pipeline

Chromantis separates three things that used to be mixed in one scenario string:

1. A **dataset** (`datasets/*.json`) lists the signals that can be built: name, unit,
   pathogen group, Hub target, default training weight, the named input and covariate sets,
   and whether archived report vintages exist (`report_vintages`).
2. A **problem** (`problems/*.json`) picks targets from one dataset and fixes the forecast
   horizons, the cross-validation folds, the evaluation inputs, the primary score and an
   optional Hub comparison. Every model in one experiment is scored under one problem.
3. A **recipe** (the scenario string) only says how a model is trained: architecture,
   transforms, reporting-error treatment, correction, optimizer, budget. It cannot choose
   targets, seasons, horizons or the score.

A problem, not a pathogen, is the unit of comparison: four-week COVID-19 admissions and a
30-week COVID-19 trajectory are two problems on the same dataset.

## Implemented (9 October 2026)

- **Loaders and checks** in `src/chromantis/problem.py`.
  The loader rejects unknown or duplicate targets and folds, a headline target outside the
  targets, a log-scale headline on a non-count target, a Hub comparison that does not match
  the headline target's Hub, and sparse horizons (the weekly scorer labels array position
  *k* as Hub horizon *k*, so horizons must be 1, 2, …, *n* with *n* ≥ 4).
- **Named signals throughout.** Episodes, network, objective, correction trees, scorer and
  FluSight export read signal names and units from the dataset; nothing assumes six
  channels or that channels 0–2 are counts.
- **Problems checked in:** short-term flu, COVID-19 and RSV (`us-*-short-term.json`), flu
  with prescribed artificial evaluation inputs (`us-flu-short-term-prescribed.json`), all
  six targets jointly, flu production (no held-out season), and a flu rolling-origin
  example (`us-flu-rolling-origin.json`).
- **Planner:** `plan` requires `--problem` and copies the problem and dataset files into the
  experiment folder, so a run stays tied to the exact definition it was trained under.
- **Production:** a release names a problem; a checkpoint is replayed only under the problem
  whose hash it recorded. Checkpoints trained before 9 October have no problem hash and must
  be retrained (the B7 production fits will be retrained, user decision 9 October).

### Fold kinds

The problem's `folds.kind` decides which weeks a model trains on, which weeks are hidden
for early stopping, and which weeks it is scored on (`dataset/cv.py`, `Problem.fold_labels`,
`Problem.training_weeks`).

| Kind | Fold IDs | A model held out on fold *f* trains on | It is scored on |
|---|---|---|---|
| `leave_one_season_out` | CDC seasons (epiweek 31 to 30), e.g. `2024-2025` | every other season in `training_ids` | season *f*, reference dates October–May |
| `leave_one_period_out` | names in `periods: {id: [first, last]}` (e.g. outbreaks) | every other period in `training_ids`; weeks outside all periods are unused | every week of period *f* |
| `rolling_origin` | names in `periods` (evaluation windows); no `training_ids` | every week before the first week of *f* | every week of window *f* |

Period dates are week-ending Saturdays, like the panel calendar, and periods may not
overlap. Early-stopping weeks are counted from the start of each training season or period;
for `rolling_origin`, from the first training week. `training_window=recent2/last2` only
exists for season folds.

Two respiratory-specific pieces still assume season folds and are rejected otherwise: the
reporting-error stage (its donor errors and `error_reference` are per season) and Hub
comparisons (the frozen Hub support is stored per season). Production problems use season
folds.

### Optional reporting-error stage

The reporting-error stage is the whole report-vintage machinery: artificial reporting errors
in training histories, the correction model (trees) and its `corrected`/`half` views,
spread calibration, and scoring on archived Wednesday reports at each Hub's deadline.

A problem switches it off with `evaluation.inputs = "finalized"`. Its models are then
evaluated on finalized histories released on each source's normal schedule, and the
pipeline never builds error banks or correction models. The recipe must train on finalized
histories and forecast from raw inputs (`history_source=finalized`, `forecast_view=raw`, no
`error_reference`, `correction_noise` or `stress_views`); other recipes are rejected when
planned. Each fold then writes only `model.pt`, `forecasts.npz` (raw view) and a manifest;
`nowcaster.pkl` holds nothing.

`evaluation.inputs = "reported"` or `"prescribed"` requires a dataset with
`report_vintages: true`. The respiratory dataset has it; a dataset built only from finalized
outbreak data would not.

Without a Hub comparison (`comparison.kind = "none"`) the scorer writes only raw WIS and its
breakdowns; the Hub-relative and pairwise tables are empty.

### Checks run

- The season-fold week roles (training, early-stopping, scored, unused) are identical to
  the previous implementation for all flu folds and settings tried (19 combinations).
- One-epoch smoke fits on CPU, numbers meaningless as scores:
  - COVID-19 admissions and ED, 8-week horizon, trained on 2022–23, 2023–24 and 2025–26 with
    artificial reporting errors and corrected histories, scored on 2024–25 Wednesday reports:
    fits, writes every view, scores and ranks.
  - Flu admissions and ED, rolling origin, trained on every week before 4 October 2025 with
    finalized histories, scored on finalized histories from 4 October 2025 to 30 May 2026:
    fits without any reporting-error stage, scores (34 forecast dates) and ranks.

## Not built yet

- **One long forecast table.** Forecasts stay in `forecasts.npz` with signal names and
  units; the scorer reads that. A long table (fold, reference date, target, location,
  horizon, quantile, value) is worth adding when an outside forecaster or a second
  evaluator needs it.
- **A second evaluator.** The only evaluator is `weekly_quantile_wis`: weekly WIS on the
  Hub quantile grid, headline on horizons 0–3. Peak timing, peak size or cumulative
  incidence for full-season models, and an outbreak evaluator for Ebola, need a new
  evaluator kind declared in the problem.
- **Named dataset builds.** The respiratory panel stays at `data/processed/panel.npz`; a
  second dataset (Ebola) needs its own source adapters and build output.
- **Reporting-error stage for non-season folds.** It would need donor errors defined per
  period rather than per CDC season.
- Ebola itself: outbreaks, case definition, geography, folds and primary score are not yet
  chosen.

## Adding Ebola

1. Choose outbreaks, weekly case definition, geography, forecast dates, horizons and folds
   (likely `leave_one_period_out` over outbreaks or `rolling_origin` within one).
2. Add source catalog entries and extraction adapters to the raw data store.
3. Add an Ebola dataset file (`report_vintages: false` unless report archives exist) and
   build it.
4. Add an Ebola problem with `evaluation.inputs = "finalized"` and its primary score; add an
   evaluator only if weekly WIS is not the right score.
5. Train a simple recipe and score an unpublished forecast.

## Rules

- Changing targets, inputs, horizons, folds or the primary score makes a new problem; it
  does not make a differently named recipe.
- No workflow framework, database, plugin system or second scheduler.
- No compatibility layer: recipe strings with the removed task fields (`forecast_targets`,
  `pathogen_inputs`, `forecast_weeks`, `evaluation_seasons`) fail, and old artifacts are
  rerun with current code. Running Slurm jobs are unaffected because they use pinned code
  snapshots.
