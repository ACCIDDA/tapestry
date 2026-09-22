# Training and prediction

The single current workflow: build the two array datasets, define a scenario,
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

## 2. Build the two training arrays

```bash
python -m tapestry.dataset.build build --data-root data
python -m tapestry.dataset.build show --dataset data/processed/finalized.npz
python -m tapestry.dataset.build show --dataset data/processed/vintaged.npz
```

Produces `data/processed/finalized.npz` (truth-only, no revision structure)
and `data/processed/vintaged.npz` (one entry per historical Wednesday
issuance, Saturday-target, state+national geography). Both share one
covariate name -> column index (`tapestry.dataset.build.STATE_COVARIATE_NAMES`
/ `NATIONAL_COVARIATE_NAMES`), grouped by `COVARIATE_GROUPS` into the source
groups a scenario's `covariate_set` string selects from (`inpatient`,
`outpatient`, `ww_wval_like`, `ww_pct_rank`, `kinsa`).

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
full contract (round-trip, order-independence, typo rejection).

## 4. Plan, run, rank

```bash
python -m tapestry.experiment.planner plan -e my-experiment \
    -s 'width=32' -s 'width=32,covariate_set=inpatient' --seeds 42 43 44

python -m tapestry.experiment.planner run -e my-experiment
python -m tapestry.experiment.planner status -e my-experiment
python -m tapestry.experiment.planner rank -e my-experiment
```

`plan` appends new scenario/seed combinations to a persistent `jobs.csv` under
`data/experiments/<name>/`, never renumbering existing tasks; run it again
with more `-s`/`--seeds` to extend the same giant experiment later. `run`
fits and evaluates one leave-one-season-out fold at a time (three seasons,
refit after early-stopping selection); pass `-t`/`--task` to run a subset, or
use the Slurm array dispatcher below for a shared GPU cluster. `rank` scores
every completed run in pure Python (`tapestry.evaluation.totals`, weighted
interval score against the frozen ensemble denominator) and writes
`season_scores.csv`, `season_composite_scores.csv`, `run_scores.csv`, and
`configuration_ranking.csv` under `ranking-<hash>/`; pass `--allow-incomplete`
to rank before every run has finished.

There is no `compare` command -- the EpiBench/R comparison path it used to
wrap was removed with R itself; `rank`'s WIS-vs-ensemble ratio is the only
comparison.

## Shared-GPU cluster launch (Longleaf)

```bash
sbatch --job-name=my-experiment --array=0-3 scripts/jlessler.sbatch my-experiment
```

One launcher for every scenario: it pulls the model from `experiment.json`
and dispatches through `tapestry.experiment.dispatch`'s shared job queue (one
whole GPU per array element, several fitting processes per GPU, drawing from
one queue across every node in the array -- no static task slices). See
[Longleaf setup](../longleaf-setup.md) for cluster environment setup.

## Defining train/validation/score splits

`tapestry.dataset.splits.season_split(dates, held_out_season)` takes one
calendar label per row of an array's leading time axis and returns a
`Split(train, val, score)` of boolean masks: `val` holds out 3-week windows
every 16 weeks (offset 4) from the two training seasons for early stopping,
`score` is every week inside the held-out season, and `train` is everything
else. This is the one place a train/val/score boundary is defined; change the
split by editing (or replacing, for a one-off experiment) that one function --
`experiment.planner.fit()` is the only caller.
