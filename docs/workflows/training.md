# Training and prediction

The single current workflow: build the dataset panel, define a scenario,
plan and run it, then rank. There is one model (`tapestry.model.network.Model`)
and one scenario space (`tapestry.model.scenario.Scenario`) -- no separate
B0/B1/B2 commands. See
[docs/design/restructure-2026-unified.md](../design/restructure-2026-unified.md)
for why. Everything below runs from the repository root; prefix with `uv run`
or activate `.venv` first.

## 1. Acquire raw sources

```bash
python -m tapestry.data --data-root data pull delphi_nhsn delphi_nssp \
    delphi_claims_inpatient delphi_claims_outpatient delphi_nwss delphi_nwss_aux \
    hub_flusight_current hub_covid_current hub_rsv_current pophive_kinsa_ili
```

Only needed when acquiring or refreshing sources; see
[the source catalog](../data/sources.md). `derived_nwss_state_indices` (the
two wastewater indices, `wval_like` and `pct_rank`) is built from
`delphi_nwss`/`delphi_nwss_aux` separately -- see
[wastewater](../data/wastewater.md).

## 2. Build the dataset panel

```bash
python -m tapestry.dataset.build build --data-root data   # about a minute, one process per source
python -m tapestry.dataset.build show
python -m tapestry.dataset.build check    # stored panel == direct extract() at sampled issuances
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
`ww_wval_like`, `ww_pct_rank`, `kinsa`).

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

National covariates (Kinsa) reach only the US row; states cannot see them
(with the default `spatial='none'`, nothing mixes locations).

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
`configuration_ranking.csv` under `ranking-<hash>/`, plus the four figures in
`ranking-<hash>/plots/`: `cv-layout-*.png` (each week's role per fold),
`fans-US.png`/`fans-NC.png` (two best configurations + hub ensemble),
`pairplot.png` (per-seed WIS ratio, coverage, WIS decomposition) and
`heatmap-*.png` (WIS ratio by location x season). Re-draw with other choices:

```bash
python -m tapestry.experiment.planner plots -e my-experiment --configs 'width=32' '' \
    --dates 2024-01-06 2025-01-04
```

Pass `--allow-incomplete` to rank before every run has finished.

There is no `compare` command; `rank`'s location-relative ratio is the only
comparison. Summed WIS columns in the outputs are raw totals, not a score.

## Shared-GPU cluster launch (Longleaf)

```bash
sbatch --job-name=my-experiment --array=0-3 scripts/jlessler.sbatch my-experiment
```

One launcher for every scenario: it pulls the model from `experiment.json`
and dispatches through `tapestry.experiment.dispatch`'s shared job queue (one
whole GPU per array element, several fitting processes per GPU, drawing from
one queue across every node in the array -- no static task slices). See
[Longleaf setup](../longleaf-setup.md) for cluster environment setup.

## Cross-validation folds

`tapestry.dataset.cv.fold(panel, scenario, held_out_season, inner)` is the one
place a train/validation/score boundary is defined; `experiment.planner.fit()`
is its only caller. It masks the panel, not episodes: every week outside the
two training seasons becomes unavailable in targets, covariates and the as-of
arrays, and training episodes are cut from that masked panel, so held-out
weeks never reach inputs, labels, loss scales or covariate standardization.
`inner=True` (scenarios with `patience > 0`) additionally hides the scenario's
validation weeks (default 4-6, 20-22 and 36-38) of each training season for
early stopping; its validation episodes score only those weeks. The
refit uses the full training seasons. Score episodes have origins in the
held-out season (from its first week, at any lookback: earlier context is
padding) and score only labels inside it. Details and history:
[the design doc](../design/restructure-2026-unified.md#4-cross-validation).
