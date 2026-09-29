# Training and prediction

Build the shared panel, choose `task=nowcast`, `task=forecast`, or `task=pipeline`,
then plan, run and rank with the same manager. Pipeline mode fits independent
nowcasting and forecasting checkpoints and passes sampled reconstructed histories
between them. See [the two-stage workflow and Python handoff](../design/nowcast-forecast.md)
for settings, assumptions, commands and outputs. Both stages use the same network
architecture and scenario space; no separate B0/B1/B2 commands are needed.
Everything below runs from the repository root.

## 1. Acquire raw sources

```bash
python -m tapestry.data --data-root data pull delphi_nhsn delphi_nssp \
    delphi_claims_inpatient delphi_claims_outpatient delphi_nwss delphi_nwss_aux \
    hub_flusight_current hub_covid_current hub_rsv_current pophive_kinsa_ili \
    delphi_fluview_ilinet delphi_fluview_clinical delphi_flusurv
```

Only needed when acquiring or refreshing sources; see
[the source catalog](../data/sources.md). `delphi_nwss` needs the archive
options shown in [the explorer overview](../explorer/overview.md) (sewershed
`*_avg_conc_lin` signals).

## 2. Build the dataset panel

After pulling `delphi_nwss` or `delphi_nwss_aux`, first rebuild the two wastewater
indices per pathogen (`wval_like`, `pct_rank`), a derived raw snapshot
`derived_nwss_state_indices`, from the selected NWSS snapshots:

```bash
python -m tapestry.dataset.build nwss-indices --data-root data   # several minutes, streams the 1.2 GB aux archive
```

It is a separate command rather than a step of `build` because it is slow and its
inputs change only when NWSS is re-pulled; `build` reads whichever snapshot is
selected. Definitions and policy: [wastewater](../data/wastewater.md#production-indices-derived_nwss_state_indices).
Then build the panel:

```bash
python -m tapestry.dataset.build build --data-root data   # about a minute, one process per source
python -m tapestry.dataset.build show
python -m tapestry.dataset.build check    # stored panel == direct extract() at sampled issuances
python -m tapestry.dataset.analyze_dataset   # rewrites docs/data/panel.md (fallback, revisions, covariates, wastewater)
```

Produces one array, `data/processed/panel.npz`: a weekly Saturday calendar
from 2023-09-02 with the retrospective truth panel (targets, state covariates,
national Kinsa) resolved at the end of the build day, plus an exact as-of
store: for every Wednesday issuance, the value visible at its cutoff for every
calendar week, for targets and covariates alike (stored on disk only where it
differs from the truth, about 10 MB). Episodes of any lookback are cut from it
at training time (`tapestry.dataset.episodes`); nothing about the lookback or
how many weeks are as-of is fixed at build time. `--truth-day` pins the truth resolution day. The layout
and every choice are in
[the design doc](../design/restructure-2026-unified.md#3-one-dataset-panel).
Covariate columns are grouped by `COVARIATE_GROUPS` into the source groups a
scenario's `covariate_set` string selects from (`inpatient`, `outpatient`,
`ww_wval_like`, `ww_pct_rank`, `kinsa`, `ilinet`, `clinical_lab`, `flusurv`;
the last three since 2026-09-22, see [FluView and FluSurv-NET](../data/fluview-flusurv.md)).

## 3. Define a scenario

A `Scenario` is a flat, frozen dataclass; its string is `key=value` tokens
joined by `,`, one token per field that differs from `Scenario()`'s default,
in any order:

```python
from tapestry.model.scenario import Scenario

Scenario(width=32, input_mode='vintaged', covariate_set='inpatient+kinsa').scenario_string
# 'width=32,covariate_set=inpatient+kinsa,input_mode=vintaged'
```

- `input_mode='finalized'` + `covariate_set=''` -- plain history-only model
  (formerly B0).
- `input_mode='vintaged'` + `covariate_set=''` -- Wednesday-vintage-aware,
  masked-nowcast model (formerly B1-direct).
- `input_mode='vintaged'` + nonempty `covariate_set` -- adds the named
  covariate groups into the shared context encoder (formerly B2).

A string only ever needs to name what's non-default, tokens can be reordered
freely, and a field added to `Scenario` later never breaks an old saved
string -- it just takes its default until named. This is what makes it safe
to keep growing one big experiment: append `,new_field=value` to any existing
scenario string and it still resolves. See `tests/test_scenario.py` for the
full contract (round-trip, order-independence, typo rejection, run-id stability).

Data-organisation fields (2026-09-22; defaults reproduce earlier runs and are
not in the string):

- `asof_weeks` (vintaged only, default 2): how many most recent context weeks
  take their target value as visible at the issuance; older weeks take final
  truth; `asof_weeks >= lookback` = fully as-of. Vintaged covariates are always
  as-of for every context week.
- `validation_weeks`, `validation_spacing`, `validation_offset` (only with
  `patience > 0`, defaults 3/16/4): the early-stopping weeks hidden in each
  training season.

National covariates (Kinsa) are broadcast to every location, preserving the
national reporting mask. State covariates remain location-specific.

## 4. Plan, run, rank

```bash
python -m tapestry.experiment.planner plan -e my-experiment \
    -s 'width=32' 'width=32,covariate_set=inpatient' --seeds 42 43 44

python -m tapestry.experiment.planner run -e my-experiment
python -m tapestry.experiment.planner status -e my-experiment
python -m tapestry.experiment.planner rank -e my-experiment
```

`plan` appends new scenario/seed combinations to a persistent `jobs.csv` under
`data/experiments/<name>/`, never renumbering existing tasks, and copies `src/`
plus the Slurm launchers into `data/experiments/<name>/code/` (with the git commit
in `code/git.json`); `scripts/jlessler.sbatch` runs that pinned copy, so edits made
after planning do not reach a queued job, and re-running `plan` re-pins the code; run it again
with more scenarios/`--seeds` to extend the same giant experiment later. `-s`
takes several scenario strings after one flag (`-s A B`); repeating the flag
(`-s A -s B`) keeps only the last one. `plan` also records in `experiment.json`
the dataset path, the frozen ensemble support and the sha256 of `panel.npz`
and of the frozen manifest; `run` and the Slurm dispatcher refuse to fit if either file
changed since planning (rebuild -> new experiment name). `run`
fits and evaluates one leave-one-season-out fold at a time (three seasons,
refit after early-stopping selection); pass `-t`/`--task` to run a subset, or
use the Slurm array dispatcher below for a shared GPU cluster. `rank` scores
every completed run in pure Python (`tapestry.evaluation.totals`) with the one
score: per target and season, the mean of per-location WIS ratios to the hub
ensemble (states/DC share 80%, US 20% by default), targets combined as
(2 x admissions + ED) / 9, seasons equal. The weights are rank-time options
(`rank --us-weight 0.2 --admissions-weight 1 --ed-weight 0.5`), recorded in the
ranking's `manifest.json` and part of its folder hash. It writes
`season_scores.csv`, `season_composite_scores.csv`, `run_scores.csv`, and
`configuration_ranking.csv` under `ranking-<hash>/`, plus the figures in
`ranking-<hash>/plots/`. Each configuration is labelled `C<k>` by its ranking
position; figures show the label and the scenario string wrapped at commas.

- `cv-layout-*.png`: each week's role per fold, one file per CV setting.
- `fans-US-hosp.png`, `fans-US-ed.png`, `fans-NC-hosp.png`, `fans-NC-ed.png`:
  rows = disease, columns = hub ensemble then each selected configuration (default:
  the best ranked; `--configs` selects others, any number, each at its lowest
  seed), x = the three held-out seasons, finalized truth and 50%/90% bands +
  median every 4 weeks (`--dates` overrides), y limits shared within a row.
- `dotplot.png`: seaborn PairGrid dot plot, every configuration plus a hub-ensemble
  row ranked by the score; columns WIS ratio all/states/US, coverage, WIS
  decomposition; opaque dot = median over seeds, light dots = seeds.
- `heatmap-<run id>.png` per selected configuration: one panel per target, WIS
  ratio by location x season (mean over seeds), one log colour scale centred at 1.

Finally it writes the report page `docs/results/<experiment>/index.md` (figures
embedded as base64 PNG, so they display in any viewer; a page is a few MB): the
figures in that order, a **Write-up** section for hand-written text, the ranking
table, and **Appendix: scenarios run** (label, full scenario string, run id, seeds,
non-default fields) linking to [the scenario field key](../reference/scenario.md),
which the same call regenerates from `Scenario` (`docs/reference/scenario.md`).
Text between the write-up markers is kept when `rank` regenerates the page;
everything else is overwritten. When ranking on Longleaf, bring the page
back with `rsync -a chadi@longleaf.unc.edu:/proj/jlessler/projects/tapestry-all/tapestry/docs/results/<experiment> docs/results/`
and add it to the `mkdocs.yml` navigation by hand. Re-draw with other choices:

```bash
python -m tapestry.experiment.planner plots -e my-experiment --configs 'width=32' '' \
    --dates 2024-01-06 2025-01-04
```

`plots` redraws the figures only; the report page is rewritten by `rank`.

Pass `--allow-incomplete` to rank before every run has finished. The report page
is written only for the complete ranking (every planned run, default score
weights); a `--seeds` subset, an incomplete ranking or non-default weights get
their `ranking-*` folder and figures, and `rank` prints why the page was not
written. A local `planner run` runs the working tree, not the code snapshot.

There is no `compare` command; `rank`'s location-relative ratio is the only
comparison. Summed WIS columns in the outputs are raw totals, not a score.

## Shared-GPU cluster launch (Longleaf)

```bash
sbatch --job-name=my-experiment --array=0-3 scripts/jlessler.sbatch my-experiment
```

One launcher for every scenario: it reads the scenarios and seeds from the
experiment's `jobs.csv` (written by `plan`), runs the code snapshot `plan` pinned in
`data/experiments/<experiment>/code`, and dispatches through `tapestry.experiment.dispatch`'s shared job queue (one
whole GPU per array element, several fitting processes per GPU, drawing from
one queue across every node in the array -- no static task slices). See
[Longleaf setup](../longleaf-setup.md) for cluster environment setup.

## Cross-validation folds

`tapestry.dataset.cv.fold(panel, scenario, held_out_season, inner)` is the one
place a train/validation/score boundary is defined; `experiment.fitting.fit()`
is its only caller. It masks the panel, not episodes: every week outside the
two training seasons becomes unavailable in targets, covariates and the as-of
arrays, and training episodes are cut from that masked panel, so held-out
weeks never reach inputs, labels, loss scales or covariate standardization.
`inner=True` (scenarios with `patience > 0`) additionally hides the scenario's
validation weeks (default 0-based weeks 4-6, 20-22 and 36-38, counted from the
season's first epiweek, CDC week 31) of each training season for
early stopping; its validation episodes score only those weeks. The
refit uses the full training seasons. Score episodes have origins in the
held-out season (from its first week, at any lookback) and score only labels
inside it. Their context before the season is the real data of the preceding
weeks (the unmasked panel); only weeks before the calendar start (2023-09-02) are
padding (unavailable). Details and history:
[the design doc](../design/restructure-2026-unified.md#4-cross-validation).
