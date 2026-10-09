# Workflow

One panel, one scenario interface, one training route, one experiment manager, one
scorer and one release path. Run commands from the repository root.
[Environment and Longleaf setup](setup.md) · [Scenario fields](reference/scenario.md) ·
[Reusable research workflow](reusable-research-workflow.md) · [Submissions](submissions.md).

| Step | Code | Command |
|---|---|---|
| Acquire sources, build the dated panel | `chromantis.data`, `chromantis.dataset` | `python -m chromantis.data pull ...`, `python -m chromantis.dataset.build build` |
| Define a study | `experiments/<name>.json` | candidates (name → scenario string), seeds, protocol text |
| Fit and evaluate each held-out season | `experiment/fit.py` | `planner plan` / `sbatch scripts/jlessler.sbatch` / `planner run` |
| Evaluate fits on other inputs, no refit | `experiment/fit.py` `replay` | `planner replay` |
| Rank runs, write the report | `evaluation/ranking.py`, `evaluation/report.py` | `planner rank` |
| Combine saved forecasts, no refit | `evaluation/ensembles.py` | `python -m chromantis.evaluation.ensembles GROUPS.json --out ...` |
| Release and weekly forecast | `chromantis.production`, `production/releases/` | `python -m chromantis.production run ...` ([production/README.md](https://github.com/ACCIDDA/chromantis/blob/main/production/README.md)) |

## Acquire and build

Training reads the saved panel; refresh sources only when creating a new dataset.

```bash
.venv/bin/python -m chromantis.data --data-root data pull delphi_nhsn delphi_nssp \
    delphi_claims_inpatient delphi_claims_outpatient delphi_nwss delphi_nwss_aux \
    hub_flusight_current hub_covid_current hub_rsv_current pophive_kinsa_ili \
    delphi_fluview_ilinet delphi_fluview_clinical delphi_flusurv
.venv/bin/python -m chromantis.dataset.build nwss-indices --data-root data
.venv/bin/python -m chromantis.dataset.build build --data-root data
.venv/bin/python -m chromantis.dataset.build show
.venv/bin/python -m chromantis.dataset.build check
.venv/bin/python -m chromantis.dataset.analyze_dataset
```

Source query options are documented under [Data](data/index.md). Wastewater index
construction is separate because it changes only when its input snapshots change.
Pin the reference resolution day with `build --truth-day`. A dataset rebuild
requires a new experiment name: planned runs verify the recorded input hashes.

## Scenarios

A scenario string contains comma-separated `key=value` fields; omitted fields use the
`Scenario` defaults ([field key](reference/scenario.md)). Six separate choices define a
model, explained in [Model choices A–F](reference/model-choices.md): A the training
histories (`history_source`: finalized, actual reports or artificial errors;
`history_correction`: corrected by cross-fitted trees; `reconstruction_labels`), B where the
artificial errors come from (`error_seasons`, `error_reference`), C which signals get them
(`error_signals`), D what the correction model learns from (`corrector_examples`,
`corrector_model`), E the evaluation inputs (`evaluation_inputs`: real Wednesday reports, or
`prescribed` artificial histories), and F the forecast view (`forecast_view`, default
`corrected`): the view a configuration is ranked and deployed in. Every fit is also scored in
the other views as diagnostics: `raw`, `corrected`, `half` and optional `calibrated`,
`sampled`, `delayed`, `nokinsa` (`experiment/fit.py`). National Kinsa is broadcast with its reporting mask; state
covariates retain their native geography.

## Plan, launch and resume

A study file lists the candidates and seeds (example: `experiments/b7-folds-20261007.json`):

```bash
.venv/bin/python -m chromantis.experiment.planner plan -e my-study --study experiments/my-study.json --device cuda
sbatch --job-name=my-study --array=0-3 scripts/jlessler.sbatch my-study
.venv/bin/python -m chromantis.experiment.planner status -e my-study
.venv/bin/python -m chromantis.experiment.planner rank -e my-study
```

`plan -s SCENARIO ...` still takes raw strings (named by run id). **Seeds.** Two seeds (42, 43; the `plan` default since 5 October 2026) screen
configurations; confirm the 3–5 finalists in a new experiment with five seeds
(`--seeds 42 43 44 45 46`) before choosing a model. In the B2 T-0 sweep (309
configurations, three seeds), one seed's per-configuration score varied by about
0.032 (median seed SD of the combined Hub-relative WIS). Two seeds ranked
configurations like three (Spearman 0.95), always picked a configuration within
0.008 of the three-seed best, and gave the right direction for 98% of matched
effects larger than 0.01. One seed could pick the 99th configuration. Neither two
nor three seeds separates configurations within about 0.01.

**Resuming.** A seed whose last attempt stopped (time limit, cancellation, failure)
after saving some fold models is continued in the same attempt when its scenario uses
prescribed evaluation histories: completed folds and forecast views are reused, and
`run.json` records `resumed`, `resumed_after` and the exact `resume_command`. Other
scenarios start a new attempt. Resubmit with `status`'s printed command
(`--retry-failed`). This replaced the one-off `scripts/resume_b7.py` (8 October 2026).

**Panel.** Plan only against a panel built after 5 October 2026 (per-Hub deadlines);
older panels are refused at fitting. Rebuild with
`.venv/bin/python -m chromantis.dataset.build build` and use a new experiment name.

For local execution, plan with `--device cpu` (or `mps`) and use
`.venv/bin/python -m chromantis.experiment.planner run -e my-study` as the launch.
The shared GPU queue fits several runs per GPU. Notifications are enabled by
default; `NTFY=0` disables them and `NTFY_URL` selects the topic.

`plan` saves the scenarios, seeds, the study file, dataset/support hashes and a source
snapshot under `data/experiments/<name>/`. Cluster jobs use that pinned code. Replanning
repins it, so use `status` to resume an existing experiment. Local `run` uses the
working tree. A rebuilt panel or changed scientific protocol needs a new name.

## Replay fitted models on other inputs

`planner replay -e NAME --inputs reported` evaluates every completed fit, fold by fold,
on archived Wednesday reports without refitting (`--inputs synthetic`: the prescribed
artificial histories). Results go to `data/experiments/NAME/replay-<inputs>/`, mirroring
the attempts, each with a manifest stating `refit: false`. Rank them with
`planner rank -e NAME --inputs reported`; they never replace the fits' own ranking.

## Analyze a run

**`planner rank` is the canonical analysis command.** It scores every run once
(cached as `scores-*.csv` in the run folder) and writes `ranking-<hash>/`:
`headline-rankings.csv` (every configuration and input view, mean and SD over seeds; only the
rows of each recipe's own `forecast_view` are ranked, `deployed=True`; other views are
diagnostics),
`headline-seed-scores.csv`, `metric-seed-scores.csv` (WIS, its dispersion/under/overprediction
components and 50–95% coverage per seed), and compact tables averaged over seeds and artificial
draws: `headline-season-scores.csv` (per season), `score-details.csv` (by month, horizon and
location, own view only), `hub-relative-scores.csv`, `hub-pairwise-scores.csv` (ours and every
Hub model) and `distribution-scores.csv` (coverage, own view, states/DC and US). Only
`headline-rankings.csv` carries the full scenario strings; the others name configurations.
**Size rule (user decision, 9 October 2026):** reports hold summaries, not raw score dumps.
Per-seed, per-location and per-view detail stays in each run's `scores-*.csv` cache on Longleaf;
`docs/` gets the page, its figures and tables of at most 1 MB (`planner rank` warns about a larger
table, and the report names it without copying it). Do not commit raw score tables to `docs/`.
Every run must be scored on identical tasks and truth, in every evaluated season, or
ranking stops. For a complete experiment it writes `docs/experiments/<name>/index.md`
with a model-choices table describing every configuration (grouped when identical: choices
A–F, labels, training seasons per fold and evaluated seasons, each linked to
[Model choices A–F](reference/model-choices.md)), the ranking table (each configuration in
its own forecast view), model-comparison bars, a seaborn
PairGrid dot plot per target (WIS states/DC and US, WIS components, coverage; every
configuration, seeds and seed mean), monthly WIS, Hub pairwise rankings, forecast fans and the fold layout
(`evaluation/report.py`). Text between the write-up markers is kept across
regenerations; a page without the markers is never overwritten.

`--allow-incomplete` and `--seeds` produce exploratory rankings without a report;
`--no-report` writes tables only. Bring reports generated on Longleaf back with:

```bash
rsync -a chadi@longleaf.unc.edu:/proj/jlessler/projects/tapestry-all/tapestry/docs/experiments/ docs/experiments/
```

## Ensembles of saved forecasts

`python -m chromantis.evaluation.ensembles GROUPS.json --out DIR` combines completed runs
without refitting: equal weight per recipe, equal weight per seed within a recipe,
quantile averaging (`vincent`) or distribution mixing (`mixture`), view by view. Each
ensemble is written as a run folder and ranked with the same scorer
(`DIR/ranking/`). The groups file names recipes by experiment, scenario and seeds (or
explicit run folders, or `"inputs": "reported"` for replays). Example:
`experiments/b7-folds-ensembles.json`. The production export uses the same `combine`.

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

**Scores** (`evaluation/standard.py`; cached per run, collected by `planner rank`):

| Table | What it compares | Scales |
|---|---|---|
| `headline-*.csv` (from raw WIS) | **The ranking.** Every predicted target in every held-out season, with or without a Hub: mean WIS per forecast task, states/DC and US separately, against the research latest panel. Window: every October–May reference date, horizons 0–3. | natural; log(x + 1) for admissions |
| `hub-relative-scores.csv` | Where a Hub ran: each location's total WIS divided by the Hub ensemble's on identical frozen Hub tasks; states/DC location ratios averaged equally, US separate | natural; log(x + 1) for admissions |
| `hub-pairwise-scores.csv` | Where a Hub ran: CDC FluSight 2025–26 method. States/DC only, models with at least 75% of the Hub's tasks, geometric mean of pairwise mean-WIS ratios on shared tasks, divided by the Hub baseline. Rank among all qualifying Hub models, with Google's models (`Google_*`) flagged; a Google model below 75% of the tasks is listed as not qualifying (2024–25 flu and COVID, 2025–26 COVID at 74.7%). Each seed joins the Hub pool alone. | natural; log(x + 1) for admissions |
| `score-details.csv`, `distribution-scores.csv` | WIS and interval coverage by month, horizon and location; weekly and four-week coverage | as above |
| ★ input fills (report footnote) | Share of available inputs that received a finalized value, per season and target: its own history (newest week, all weeks) and all six target histories read at its Hub's deadline; covariates per Hub deadline | — |

ED proportions are never transformed. Lower is better everywhere; 1 is the ensemble
(Hub-relative) or the Hub baseline (pairwise). The primary ordering is states/DC
log-admission WIS (user decisions 5 and 8 October 2026). Until 8 October 2026 the
ranking was the B0 composite (`totals.py`: location-relative ratio to the Hub ensemble,
states/DC 80% and US 20%, admissions 1 / ED 0.5); it was deleted on the user's
instruction to score like FluSight with US and states reported separately.

Assumptions: the CDC report does not state its log offset; we use log(x + 1), as the
FluSight and COVID dashboards do. Applying the FluSight pairwise method to the COVID and
RSV Hubs is our choice. Pairwise rankings use the frozen Hub truth; raw WIS uses the
finalized panel truth. Runs fitted before this change must be refitted.

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
the held-out season. The scenario selects evaluation seasons (`evaluation_seasons`) and the pinned panel
supplies the training calendar; `production` fits every completed season and scores nothing. These retrospective folds can train on seasons
later than the scored season; they are not forward deployment simulations.

## Protocol of the B7 relaunch (7 October 2026)

On 7 October 2026, the individual-model relaunch adopted a prescribed 2025–26
reporting process in every held-out season (`error_reference=2025-2026`,
`evaluation_inputs=prescribed`; then named `revision_reference` and `evaluation_vintaging=1`). Training and evaluation admission and ED histories
are artificially revised throughout their lookback, including histories with
real archived reports. Future prediction labels remain the September research
latest values. Kinsa is unchanged, and FluSurv is omitted. The epidemic folds
remain exchangeable leave-one-season-out folds, evaluating 2023–24, 2024–25 and
2025–26 and training on the other three completed seasons including 2022–23.

Revision donors match only the permitted flu admission/ED signals. By default
one donor release is shared across locations and matched on normalized level
and recent change without calendar. The closest seasonal donor release is
excluded in every recipient season. Adjacent error windows may share reference
weeks: the 2025–26 reporting process is prescribed separately, and only errors,
not donor epidemic trajectories or future labels, are transported. This rule
replaced the absolute-date full-window exclusion on 7 October 2026 because that
exclusion removed winter donors only when evaluating 2025–26. Missing donor residuals
use genuine age-aligned evidence at the same location, then national/native
location pools; they do not default to unrevised later values. The prescribed
revision calibration is separate from the epidemic training-season mask.

The default nowcaster is a small tree trained on synthetic preliminary
histories, with research latest training histories as reconstruction labels.
It corrects two admission and ED weeks. The focused 30-minute screen compares
four weeks and a residual MLP on the three-block sampled MLP, plus removal of
forecaster calendar inputs on sampled and direct-quantile leaders. A is trained behind its
cross-fitted corrector; B retains raw synthetic-history joint reconstruction
training, so its correction-estimator/window comparisons change its scoring
inputs without changing the neural fitting treatment. All forecasters are
refitted. Their modern training and epoch selection use the existing
Q95-normalized native loss plus 0.5 times log(1 + admissions) loss; ED stays
native. Historical ILI initialization and the existing training geography
weights are retained.

## Restructuring log, 8 October 2026

Every submitted model (System2, B7) and every experiment since B3 used one training
route, the former B3 "pilot" route. On the user's instruction ("each thing done once
and well"; delete unused code rather than store it), the package was reduced to that
route and one path per step:

- **Deleted training routes:** plain finalized/vintaged fits, B2 "weekend" revision
  experiments, fixed-checkpoint replay with nowcast inputs (`replay_from`), standalone
  nowcasting, nowcast-to-forecast pipelines and per-signal finalizers, with their
  modules (`experiment/{fitting,weekend,replay,finalization,two_stage,baseline,nowcast_baseline,augmentation}.py`,
  `model/{finalization,context_nowcast,revision_uncertainty,baselinenowcast,pipeline,revision_regression}.py`,
  `dataset/finalization.py`), their scenario fields and the `finalized`/`finalized_available`
  episode modes. Strings naming those fields now raise. The finality-flag and
  estimated-input network channels went with them. Their experiment pages remain as
  dated reports. To reproduce them, use git history (last commit before the restructuring:
  `8616acc`) or the `src/` snapshot stored with each Longleaf experiment
  (`data/experiments/<name>/code`). The B3–B7 changes and scripts that were never committed
  are not kept elsewhere: the local backup archive was deleted on 9 October 2026 at the
  user's request.
- **Renamed:** `experiment/pilot.py` → `experiment/fit.py`; its scoring and ranking moved
  to `evaluation/ranking.py`; the reporting-error scoping (`ScopedErrors`) merged into
  `dataset/reporting_error.py`.
- **Evaluation:** `totals.py`, `plots.py`, `report_layout.py`, `effects.py`, `reports.py`
  replaced by `ranking.py` and `report.py`; the WIS and Hub-relative helpers moved into
  `standard.py`. The B0 80/20 composite ranking and matched-effects tables were removed.
- **Scripts:** about 50 experiment-specific planners, replays, ensemble scorers and report
  scripts (B4–B7, peak study, audits) were replaced by study files, `planner replay`,
  `evaluation.ensembles` and `chromantis.production`.
- **Verification:** re-exporting the 7 October B7 forecasts through the new code reproduced
  the submitted CSV byte for byte (SHA256 `cdee271c…`); replaying one production
  checkpoint per B7 recipe on CPU gave identical forecasts with old and new code; a short
  fit of each B7 recipe gave identical training losses. All tests pass.
- **Renamed (same day, user decision):** `pilot_method` → `training_histories`
  (`two_stage` → `corrected`), `pilot_nowcaster` → `corrector_examples` + `corrector_model`,
  `evaluation_vintaging` → `evaluation_inputs`, `revision_*`/`vintage_seasons` → `error_*`,
  `evaluation_replicates` → `evaluation_draws`, `nowcast_weeks` → `reconstruction_weeks`,
  `nowcast_noise*` → `correction_noise*`. Every report from B3 on now has a model-choices
  A–F table linked to [the explanation](reference/model-choices.md).
- **Review, 9 October 2026 (user agreed):** `training_histories` bundled three choices and was
  split into `history_source`, `history_correction` and `reconstruction_labels`; the forecast
  view became a recipe field (`forecast_view`, default `corrected`), and ranking, reports and
  production use only that view (previously each configuration's best view was picked after
  scoring); protocol tables describe every configuration with its training seasons per fold,
  evaluated seasons and labels; ensembles record their own `forecast_view`, and scoring no
  longer assumes a `raw` forecast file exists (ensembles may hold only `corrected`).
- **Old names dropped, 9 October 2026 (user decision):** instead of translating old field names
  forever, the records still in use were rewritten to the current names and the translation
  code was removed. Rewritten, on Longleaf and in the local copies: the B7 folds
  (`b7-folds-20261007`), B7 production fits and operational forecasts
  (`b7-production-20261007`, `b7-operational-20261007`), and the System2 production fits and
  operational forecasts (`b6-production-system2-20261007`, `b6-operational-20261007/system2`):
  scenario strings in `jobs.csv`, `runs.csv`, run and fold manifests and operational manifests;
  run folders renamed to the current run ids (e.g. B7 production
  `…-8561787f0cec` → `…-e60aa9e03e00`, `…-5af694f72554` → `…-ffdcc78fdeae`,
  `…-d29b75ad47f9` → `…-c2af77a5341a`); ranking folders and view symlinks deleted (regenerated by
  `planner rank`). `model.pt`, correction trees and pinned code snapshots were not touched, so
  checkpoint hashes are unchanged; the metadata inside `model.pt` keeps the old string as
  provenance, so `fit --resume` cannot continue these (all are complete). The pre-migration
  record backups were deleted after the checks, at the user's request. Checks: the planner reads
  all three experiments (B7 folds 16 complete / 4 stopped, as before; B7 production 30; System2 90); all
  120 operational manifests parse and point to existing checkpoints; replaying one checkpoint
  per recipe of both releases gives forecasts identical to the pre-migration replays; re-exporting
  the 7 October forecasts reproduces both submitted files byte for byte (B7 `cdee271c…`,
  System2 `b68b3422…`); `python -m chromantis.production run` completes for B7. Every other
  experiment (B3–B6, older B7 checks) is history: its reports remain, the planner can no longer
  read it, and rerunning means a new experiment. The B4 fallback release was deleted. A System2
  release (`production/releases/system2-20261007.json`, the 80 submitted checkpoints) was added.
  Docs cleanup (9 October 2026, user decision): deleted interim B6 snapshot pages
  (`b6-extension-score-*`, `b6-candidate-ensembles-*`, `b6-broad-*`), settings-only folders
  linked from nowhere (`b6-search`, `b6-extension-check`, `b6-matched`, `b6-specialists`,
  `b7-revisions*`), `b7-plan` (never run), `training-audit` (tables without a page),
  `submission-peer-review`, `b3-vintage-flu-1000-design`, `performance-ideas`,
  `flusight-selection-review` (superseded by the submission choice review), and the root
  notes `implementation-plan.md`, `icare.md`, `todo.md`. Report folders keep their page,
  figures and tables up to 1 MB; larger tables stay in the ranking folder (`report.py`). The
  runs index is curated by hand.
  The B7 fold experiment was then renamed `b7-fast-20261007` → `b7-folds-20261007` (Longleaf
  folder and its records, the report folder `docs/experiments/b7-folds-20261007/`, the study
  files `experiments/b7-folds-20261007.json` and `experiments/b7-folds-ensembles.json`); the planner
  and the ensembles file read it. Its report pages keep the label "B7-fast" in their text.
- **Unchanged:** defaults, the saved correction-tree pickles, the WIS mathematics and the
  mixture rule. Run folders are named by `Scenario.run_id`.
