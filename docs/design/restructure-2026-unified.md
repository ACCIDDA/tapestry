# Unified B0/B1/B2 restructuring (2026-09)

Decisions for collapsing the model/data/scoring stack into one simpler shape.
This document is the spec; implementation follows it exactly rather than
re-deriving choices. Research project — favor deleting code over guarding it.

## 1. Model: one architecture, one scenario, a string that only ever grows

**Finding that drives this**: B2 already *is* B0+B1 combined at the class
level (`B2(B1)` forces `direct=True`, and `B1(direct=True)` internally builds
a `B0`). The real complexity isn't the network — it's three scenario
dataclasses (`TrainingScenario` -> `B1Scenario` -> `B2Scenario`) each with
their own hand-written positional string codec that breaks the moment a field
is added or reordered.

**Decision**: keep the `B0` `nn.Module` (`src/tapestry/models/b0.py`)
essentially as-is — it is the one network. Drop the alternate B1 pipelines
that are not the direct/B2 path (`two_stage`, `parallel_recent`,
`direct_finalflag`, `joint_aux025` and their `SampleHead`/two-stage
encode/decode machinery in `b1.py`). Only the `direct` pipeline (B1 wraps B0)
survives, since that's the one B2 requires and the one going forward. Delete
`B0`/`B1`/`B2` as three classes; replace with a single `Model` class in
`src/tapestry/model/network.py` that is today's `B0` forward pass, plus the
thin wrapping logic B1's `direct` mode adds (feeding Wednesday-vintage recent
weeks + a `known_final` flag channel instead of always-finalized input). Fold
that wrapping into `Model.forward` behind a plain bool arg (`vintaged: bool`),
not a subclass.

**Scenario**: replace `TrainingScenario`/`B1Scenario`/`B2Scenario` with a
single frozen dataclass `Scenario` in `src/tapestry/model/scenario.py`,
union of the fields actually still needed (drop fields that only existed for
removed pipelines: `pipeline`, `supplied_final` stays since direct mode still
uses it). Fields: everything in today's `TrainingScenario` **minus**
`loss_weights` string alias (keep it — cheap), **plus** from B1:
`mask_rate`, `mask_recent`, `mask_gap`, `mask_outage`, **plus** from B2:
`covariate_set` (a `+`-joined subset of `SOURCE_GROUPS`, `''` = no
covariates), `input_mode` (`finalized` | `vintaged` — which prebuilt array,
§2, the scenario trains against).

**String codec — the actual fix for "add to the string and it resolves"**:
replace the positional `prefix + fixed-order fields` scheme with
self-describing `key=value` tokens joined by `,`, order-independent, one
token per **non-default** field only:

```python
def scenario_string(self) -> str:
    return ','.join(f'{f.name}={encode(f.name, v)}'
                     for f in fields(self) if (v := getattr(self, f.name)) != f.default)

@classmethod
def from_string(cls, value: str) -> 'Scenario':
    if not value:
        return cls()
    options = {}
    for token in value.split(','):
        key, _, raw = token.partition('=')
        options[key] = decode(key, raw)   # dataclass default fills anything absent
    return cls(**options)
```

Consequences (all intentional):
- Old strings and new strings both parse: a string only ever needs to name
  what differs from the default `Scenario()`.
- Order doesn't matter, so a human (or a shell history) can append a new
  `key=value` pair anywhere and it still resolves.
- Adding a new dataclass field never breaks existing saved strings — it just
  takes its default until named explicitly.
- No hand-maintained `PREFIX`/`CODES` short-code tables; encode/decode just
  use `str`/`int`/`float`/`bool` conversion by field type, keeping a small
  `CODES` dict only for the handful of enum-like string fields (encoder,
  decoder, spatial, noise, heads, count_transform, ed_transform, us_error,
  head_sharing, fit_partition, loss_weights) so typos still raise instead of
  silently becoming a new field.
- `covariate_set='' ` + `input_mode='finalized'` reproduces plain B0.
  `covariate_set=''` + `input_mode='vintaged'` reproduces B1-direct.
  Nonempty `covariate_set` + `input_mode='vintaged'` reproduces B2. One
  scenario space, no subclassing, no backend dispatch by string prefix.

**Experiment planner**: `src/tapestry/models/manager.py` +
`src/tapestry/models/backends.py` collapse into one
`src/tapestry/experiment/planner.py` with no per-model branching at all (the
three `Backend` classes existed only to special-case B0 vs B1 vs B2; with one
`Scenario` there is one code path). Keep the CLI shape (`plan`, `run`,
`status`, `rank`, `compare`) — it already works well; just delete the
dispatch layer. `dispatch.py` (Slurm queue) and `provenance.py` carry over
unchanged (model-agnostic already).

## 2. Explorer & data layer

**Scope cut** (per direction: only Wednesday-issued forecasts of the
Saturday target, only state + national geography, plus wastewater):
- `model_data`/dataset building only ever produces Wednesday-issuance,
  Saturday-target episodes (`wednesday.py`'s `output_dates` already enforces
  this — no change needed there, just delete the finalized-only "any day"
  path since it's superseded, see §3).
- Drop `RSVNet`, `cdc_nssp_trajectories`'s HSA resolution, and any other
  catalog entries that only exist for sub-state/HSA/county granularity.
  `geography.py`'s `observation_geography` already collapses everything to
  state-or-`US`-or-`None`; keep it as the one place geography is decided.
- Keep exactly two wastewater indices per pathogen, already the ones
  promoted into B2: `wval_like` and `pct_rank` (module
  `analysis/covariates/indices.py`'s other four candidates —
  `robust_z`, `flowpop_wval`, `conc_matched`, `wval_popw` — move to
  `analysis/covariates/exploratory.py` or are deleted; they are not part of
  the production dataset build).
- Keep `explorer/` (download/version/browse) as the generic layer over
  `RawDataRepository` — it already does "display" (server + static export)
  and doesn't need model-specific changes. Trim `catalog.py`'s `_SPECS` to
  the sources actually used by the two training arrays (§3) plus what the
  explorer needs to show provenance: NHSN final/preliminary, NSSP,
  Delphi claims inpatient/outpatient, derived NWSS state indices, PopHive
  Kinsa, Hub current + git-mirror target data. Delete specs for anything else
  (comprehensive/legacy Hub variants no longer scored, county sources).

**Extraction is one function with two axes** (target vs. covariate,
vintaged vs. finalized), in `src/tapestry/dataset/extract.py`:

```python
def extract(name: str, kind: Literal['target', 'covariate'],
            version: Literal['vintaged', 'finalized'],
            as_of: date | None = None) -> pd.DataFrame:
    ...
```

`finalized` always reads the latest snapshot truth (today's
`finalized.py`/`b2.py` "final" path). `vintaged` reads the archive as-of a
cutoff (today's `VintageArchive.resolve`); `as_of=None` means "every
Wednesday issuance," matching what the array builders need. This function is
the single place both array builders (§3) and any ad-hoc notebook/analysis
code call — no more separate bespoke read paths per dataset.

## 3. Two training arrays, one covariate index

Replace `model_data/finalized.py` + `model_data/wednesday.py` +
`model_data/b2.py` (three separate `.npz` builders with three separate
covariate/column conventions) with **one builder module**,
`src/tapestry/dataset/build.py`, producing exactly two artifacts under
`data/processed/`:

- `finalized.npz` — truth-only array, no revision structure.
- `vintaged.npz` — one entry per historical Wednesday issuance, i.e. what
  today's `WednesdayDataset.episodes()` yields, but carrying the same
  covariate panel and index as `finalized.npz` (today `b2.py` only attaches
  to the Wednesday path — keep it that way; `finalized.npz` covariates are
  included too for symmetry/backtesting even though production scenarios
  with `input_mode='finalized'` won't use them).

**Array layout (npz keys), shared by both files**:

```
dates:            datetime64[D], shape [T]      # Saturday target weeks
locations:        str,           shape [L]      # 2-letter state codes + 'US'
targets:          float32,       shape [T, L, 6]
target_names:     str,           shape [6]      # index via {n: i for i, n in enumerate(target_names)}
covariates:       float32,       shape [T, L, K]      # state-resolved covariates, NaN = unavailable
covariate_mask:   bool,          shape [T, L, K]
covariate_names:  str,           shape [K]            # index via {n: i for i, n in enumerate(covariate_names)}
covariates_national: float32,    shape [T, Kn]         # national-only sources (currently just kinsa_ili)
covariate_national_names: str,   shape [Kn]
```

`vintaged.npz` additionally carries an `issuance` axis in front of `T`
(shape `[I, T_i, ...]` is ragged per issuance in general; store instead as a
flat table of episodes exactly like today's `WednesdayDataset.episodes()`,
i.e. add `issuance_dates: datetime64[D], shape [I]` and index `targets`/
`covariates` as `[I, lookback+horizon, L, ...]` — one episode per issuance,
which is what the model actually consumes). Both files use the *same*
`covariate_names`/`covariate_national_names` ordering so a `covariate_set`
string resolves identically regardless of which array a scenario points at.

`COVARIATE_GROUPS` (today's `b2.py`) becomes the single source of truth for
`covariate_names` order and is reused, unchanged, to expand a scenario's
`covariate_set` string into column indices.

**Split definition** — `src/tapestry/dataset/splits.py`:

```python
@dataclass(frozen=True)
class Split:
    train: np.ndarray   # boolean mask over the array's leading time axis
    val: np.ndarray
    score: np.ndarray

def season_split(array, held_out_season: str, val_weeks=3, val_spacing=16, val_offset=4) -> Split:
    ...
```

One function, one place, reused by both arrays and by `rank`/`compare` in
the experiment planner. This directly replaces `season_cv.py`'s
`fold_data`/`masked_episodes`/`validation_split`, generalized to operate on
the array schema above instead of bespoke per-model tensors.

## 4. Scoring: pure Python only

- Delete `src/tapestry/evaluation/epibench.py`,
  `scripts/validate_epibench_evaluation.py`, `scripts/setup_r.R`,
  `tests/test_hub_evaluation.py`'s EpiBench cross-check cases, and the
  `epibenchmark` pip dependency. No R, no subprocess to `Rscript`, no
  `scoringutils`.
- Keep `src/tapestry/evaluation/totals.py`'s WIS math
  (`quantile_scores`, `SCORE_DEFINITION`/`SCORE_VERSION` aggregation) as the
  only scorer, but repoint its inputs at the two arrays from §3 instead of
  Hub-format CSVs/frozen-ensemble parquet: truth comes from
  `finalized.npz`/`vintaged.npz`'s `targets`, forecast samples come from a
  scenario run's `forecasts.npz`, and `Split.score` (§3) selects which weeks
  count. Keep the frozen ensemble-vs-model ratio (`SCORE_VERSION
  location-relative-season-first-us20-v1`) unchanged — it's the actual
  scoring policy, not something in scope to simplify.
- `evaluation/sweep.py`, `compare.py`, `hubs.py`, `nowcast.py`,
  `decisive.py`, `pdf_report.py` are audited by the implementer and trimmed
  to whatever still calls into `totals.py`'s new array-based entry point;
  anything that exists solely to feed EpiBench or the deleted pipelines is
  deleted outright rather than kept dark.

## 5. New module layout

```
src/tapestry/
  data/            # unchanged: raw acquisition/versioning (repository.py, catalog.py trimmed, sources/)
  explorer/        # unchanged: browse/display over data/
  dataset/         # NEW, replaces model_data/
    extract.py     # §2 extract()
    build.py       # §3 builds finalized.npz + vintaged.npz
    splits.py      # §3 Split/season_split
  model/           # NEW, replaces models/ B0/B1/B2 classes + scenarios.py stack
    network.py     # the Model nn.Module (today's B0 + B1-direct wrapping)
    scenario.py    # Scenario dataclass + string codec
    objective.py   # unchanged (fair-CRPS loss weighting)
  experiment/       # NEW, replaces models/manager.py + backends.py
    planner.py     # plan/run/status/rank/compare, no model branching
    dispatch.py    # unchanged Slurm queue, moved
    provenance.py  # unchanged, moved
  evaluation/
    totals.py      # WIS, repointed at dataset/ arrays (§4)
    (sweep.py, compare.py, hubs.py, nowcast.py, decisive.py, pdf_report.py — trimmed per §4)
```

Delete: `models/b0.py` classes folded into `model/network.py`
(`models/architecture.py` merges in too), `models/b1.py`, `models/b2.py`,
`models/scenarios.py`, `models/b1_scenarios.py`, `models/b2_scenarios.py`,
`models/manager.py`, `models/backends.py`, `models/season_cv.py` (replaced
by `dataset/splits.py` + `experiment/planner.py`), `models/bundles.py` if it
only served the removed per-pathogen/per-target independent-fit machinery
for non-`all` `fit_partition` values (keep `fit_partition` support itself —
just confirm `bundles.py`'s `IndependentBundle` still applies to the unified
`Model`), `model_data/finalized.py`, `model_data/wednesday.py`,
`model_data/b2.py`, `model_data/forward.py` (folded into `dataset/build.py`
+ `dataset/extract.py`), `evaluation/epibench.py`.

CLIs: collapse `tapestry-model-data` + the model-training half of
`tapestry-experiments`/`manager.py` into one `tapestry-dataset` (build/
extract/show) and one `tapestry-experiment` (plan/run/status/rank/compare)
entry point; `tapestry-data` (raw acquisition) and `tapestry-explore` are
unchanged.

## 6. What's explicitly out of scope

- No change to the fair-CRPS loss, the FiLM decoder, or the season-CV
  leave-one-out policy itself (3 seasons, refit-after-early-stopping) —
  only where that logic lives.
- No change to `SCORE_DEFINITION`'s weighting (states 80/US 20, admissions
  1.0/ED 0.5, seasons averaged equally).
- `analysis/`, `docs/`, `references/` content is not rewritten, only
  pointers updated where they reference deleted module paths.
