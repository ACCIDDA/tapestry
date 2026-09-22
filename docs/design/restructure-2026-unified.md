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
covariates), `input_mode` (`finalized` | `vintaged` — how episodes are cut from the one
panel, §3).

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
  the sources actually used by the training panel (§3) plus what the
  explorer needs to show provenance: NHSN final/preliminary, NSSP,
  Delphi claims inpatient/outpatient, derived NWSS state indices, PopHive
  Kinsa, Hub current + git-mirror target data. Delete specs for anything else
  (comprehensive/legacy Hub variants no longer scored, county sources).

**Extraction** lives in `src/tapestry/dataset/extract.py` (rewritten
2026-09-22 for speed, same policy). Each archive-backed source (six targets,
four Delphi claims covariates) is read once into a revision table
(`revisions(name)`: tier, release, reference Saturday, location, value) and
resolved as of any cutoff day by `resolve(revisions, day, dates)`; NWSS indices
and Kinsa, which are report-dated files rather than archives, resolve through
`resolve_reports`. `extract(name, as_of)` wraps both for ad-hoc use. The
vintage policy, stated in the module docstring and unchanged from the
row-by-row `VintageArchive` it replaced:

- Three tiers per target: native Hub `as_of` full snapshots, Hub Git-history
  full snapshots (every registered commit is a full snapshot, including empty
  ones, so deletions stay deleted), and Delphi per-observation report-time
  revisions. A location is covered by a Hub tier from the earliest reference
  week in any eligible release of that tier; covered cells take the latest
  eligible snapshot's value (absent = unavailable, no fallback). Native coverage
  precedes Git, which precedes Delphi; outside Hub coverage the latest eligible
  Delphi revision is used (a missing latest revision is unavailable).
- Same-release, same-cell disagreements (including one missing, one present)
  are unavailable, never resolved by file order.
- A release is visible at a cutoff day when released by 23:59:59.999999 UTC
  that day (so a Wednesday issuance sees data released on that Wednesday). The
  cutoff is always given as a calendar day: `cutoff_time` rejects an
  already-converted timestamp, which is how the previous `build_vintaged`
  crashed (`cutoff_time` applied twice).
- Values must be finite and nonnegative; NSSP percentages at most 100 and
  divided by 100. Claims archives are daily; only Saturday reference days are
  kept (as `model_data/b2.py:_claims_frame` did), no weekly averaging; rows with
  a non-native Delphi `fill_method` are dropped.
- NWSS and Kinsa: latest non-missing value reported on or before the cutoff
  day. Kinsa's weekly value is the Saturday-ending mean of seven daily values,
  reported once all seven are visible; its as-of archive begins 2026-04-06, so
  vintaged episodes before then have no Kinsa.

## 3. One dataset panel

**User decision 2026-09-22**: one dataset array, `data/processed/panel.npz`,
replaces `finalized.npz` + `vintaged.npz`, and the lookback is no longer fixed
at build time. Built by `python -m tapestry.dataset.build build` (one process per
source, about a minute; see the decision log for timings).

```
dates:                    datetime64[D] [T]      Saturdays from CALENDAR_START (2023-09-02) to the build day
locations:                str [L]                50 states + DC + 'US'
target_names:             str [C=6]              nhsn_{flu,covid,rsv}_admissions, nssp_{flu,covid,rsv}_proportion
targets:                  float32 [T, L, C]      truth, NaN = unavailable
covariate_names:          str [K]                inpatient/outpatient claims, NWSS wval_like/pct_rank
covariates:               float32 [T, L, K]      truth, NaN = unavailable
covariate_mask:           bool [T, L, K]         = ~isnan(covariates)
covariate_national_names: str [Kn]               kinsa_ili
covariates_national:      float32 [T, Kn]        truth (national only)
issuance_dates:           datetime64[D] [W]      Wednesdays from 2023-08-30 to the build day
asof_targets:             float32 [W, T, L, C]   week t as visible at issuance w's cutoff
asof_covariates:          float32 [W, T, L, K]   week t as visible at issuance w's cutoff
asof_covariates_national: float32 [W, T, Kn]     week t as visible at issuance w's cutoff
metadata:                 JSON                   build constants, truth day, raw snapshot ids, per-source seconds
```

**Exact as-of store (user decision 2026-09-22).** For every Wednesday issuance
w, the panel holds the value visible at its cutoff (end of the issuance day,
UTC) for *every* calendar week from `CALENDAR_START`, for targets and state and
national covariates alike, so a forecast at any issuance can be reconstructed
exactly as it was. Cells after `context_end(w)` (the Saturday four days before
issuance w) are NaN by definition; NaN elsewhere means nothing was visible at
the cutoff. This is the in-memory layout (`build.load`). On disk
(`build.save`), each as-of array is stored only where it differs from the truth
panel: a bool mask `<name>_revised [W, T, ...]` and the differing values
`<name>_values [N]` (NaN = present in truth, not visible at the cutoff), with
`np.savez_compressed`; `load` rebuilds the dense arrays exactly
(`tests/test_dataset.py` round trip). The 2026-09-22 build is 10.0 MB (4.6 MB
for the previous 2/52-week overlay): 0.74 M revised target cells and 4.8 M
revised covariate cells, of which the claims covariates' 2.2 M non-missing
revised values are 8.5 MB. Encoding along the issuance axis (store a cell only
when it changes from the previous issuance) was measured at about 6 MB but
needs a forward fill to decode; the simpler difference-from-truth encoding was
kept.

`python -m tapestry.dataset.build check [--samples 4]` compares the stored panel
(decoded) with a direct `extract.extract(name, day, dates=...)` for every
source at randomly sampled issuances and at the truth day, cell for cell.
Result for the 2026-09-22 build: see the decision log.

**float32, not float16.** The user suggested float16 to save space. float16 has
an 11-bit significand and a maximum of 65504: integers are exact only up to
2048, the spacing is 32 between 32768 and 65504, so US weekly influenza
admission counts (tens of thousands at peak) would be rounded, and larger
counts would overflow. Values stay float32 (integers exact to 2^24), as in the
truth panel.

**`known_final`** is not stored. In a vintaged episode, a context cell within
the last `asof_weeks` weeks is known-final only when nothing was visible at the
cutoff and the truth fallback is available (previous builder's rule, below);
older context weeks are known-final wherever the truth is available. In a
finalized episode every available context cell is known-final.

Build choices:
- **Truth** is resolved at the end of the build day (`--truth-day`, default
  today, recorded in metadata). The previous `vintaged.npz` used the last
  Wednesday instead; one truth day for both modes is simpler and differs only
  by revisions released since that Wednesday.
- **One calendar for both modes, starting 2023-09-02.** Nothing earlier exists
  in the panel; episodes pad earlier context weeks as unavailable. The previous
  vintaged builder resolved context weeks before 2023-09-02 from the archives;
  those weeks belong to seasons outside cross-validation (masked in training
  anyway) except August 2023, which now only affects the context of the first
  2023-24 score origins.
- **How much of a vintaged window is as-of is a scenario choice, not a build
  choice**: `Scenario.asof_weeks` (below). The earlier build constants
  `ASOF_TARGET_WEEKS = 2` and `ASOF_COVARIATE_WEEKS = 52` (and the lookback <= 52
  limit for vintaged covariates) are gone.

**Episodes** (`src/tapestry/dataset/episodes.py`, one builder, one switch):
- `input_mode='finalized'`: one episode per calendar Saturday origin t, context
  weeks t-lookback+1..t, targets t+1..t+4, all truth; origins from the first
  calendar week through the last whose four targets fall in the calendar.
  Every visible context cell is known-final.
- `input_mode='vintaged'`: one episode per Wednesday issuance, origin its
  context end (issuance - 4 days), targets the four following Saturdays
  (reference date = issuance + 3 days, horizon 0 in hub terms). The last
  min(`asof_weeks`, lookback) context weeks take the as-of target value where
  one was visible (known_final False) and fall back to truth where nothing was
  visible yet (known_final True where available); older context weeks and all
  target weeks are truth; covariates (state and national) are as of the
  issuance for every context week, NaN where nothing was visible (no truth
  fallback).
- **`Scenario.asof_weeks`** (int, default 2, 2026-09-22): default 2 reproduces
  the previous vintaged builder (B1): checked cell for cell against the
  previous panel and episode code for lookbacks 4, 12 and 52, both input modes,
  with covariates (decision log). A value >= lookback makes every context week
  as-of. Only meaningful with `input_mode='vintaged'` (a non-default value with
  `finalized` raises, so two identical fits cannot carry two scenario strings).
  **One field, targets only; covariates always fully as-of** (implementation
  choice, the option with the fewest fields that reproduces old behaviour by
  default): the previous builder took every covariate week of a vintaged window
  as of the issuance, so a single `asof_weeks` applied to covariates as well
  would have changed default B1/B2 covariates, and two fields would add a knob
  nobody has asked to vary.
- **The truth fallback is the previous builder's rule, now applied to however
  many weeks `asof_weeks` names.** Where nothing was visible at the cutoff (a
  week not yet reported, or a missing archive), a vintaged context week in the
  as-of window takes the final truth, flagged known-final. With a large
  `asof_weeks` this puts later truth into cells that were empty at the time, so
  a "fully as-of" episode is exact wherever something was visible and
  truth-filled elsewhere. Flagged for the user; not changed because the
  default must reproduce B1.
- Kept only if some context cell and some target cell are available.

`COVARIATE_GROUPS` (in `build.py`) stays the single source of truth for
covariate order; a scenario's `covariate_set` expands through
`covariate_names_for`. **Current behaviour:** a national-only covariate (Kinsa)
is placed in the `US` column only and is unavailable for every state, so with
`spatial='none'` (no cross-location attention) state rows never see Kinsa;
only the US row does.

## 4. Cross-validation

`src/tapestry/dataset/cv.py` (replaces `splits.py`, 2026-09-22). One function,
`cv.fold(panel, scenario, held_out, inner=False)`, called by the planner.
Policy = the pre-refactor `models/season_cv.py` one (`fold_data`,
`masked_episodes`, `validation_split`, `channel_scales`), implemented by
masking the array rather than selecting episodes:

- **Fold**: copy the panel; every week outside the two training seasons (the
  held-out season, and weeks in no CV season) becomes unavailable in targets,
  covariates and the as-of arrays (by reference week). Training
  episodes are cut from the masked panel with origins in training weeks only,
  so held-out weeks vanish from inputs, labels, loss scales, loss weights and
  covariate standardization (computed from the training episodes' visible
  covariate cells).
- **Inner early-stopping fit** (`patience > 0`): additionally hide
  `validation_weeks` consecutive weeks of every `validation_spacing`, starting
  at week `validation_offset` of each training season (defaults 3/16/4: weeks
  4-6, 20-22 and 36-38). Validation episodes are cut from the training panel
  (hidden weeks visible), with origins in the four (= number of horizons)
  training weeks before each hidden week, and score only hidden weeks.
- **CV settings are scenario fields (user decision 2026-09-22)**: training-data
  organisation changes the fit, so it belongs in the scenario and its run id.
  `validation_weeks`, `validation_spacing`, `validation_offset` replaced the
  `cv.py` constants; at their defaults they are absent from the scenario string,
  so existing strings and run ids are unchanged (`tests/test_scenario.py` pins
  three run ids recorded before the change). Non-default values with
  `patience=0` raise (they would not change the fit). The set of seasons
  (`cv.SEASONS`) stays a module constant: it is tied to the frozen ensemble
  support, the export and the run-completeness check, so moving it is not
  simple (not done). `cv.week_roles(dates, scenario, held_out)` is the one
  place a week's role (fit / validation / score / unused) is decided; `fold`
  and the CV-layout figure both use it.
- **Refit** after epoch selection: the fold's training panel without the
  validation mask (full training seasons, as old B0).
- **Score**: origins in the held-out season from the unmasked panel, from its
  first week at any lookback (padding); earlier weeks are allowed as context,
  only labels inside the held-out season count.

`tests/test_dataset.py` checks that no held-out or validation week value
reaches any training input, label, loss scale/weight or covariate scale, and
that score labels stay inside the held-out season.

## 5. Scoring: pure Python only, one score

- No R, no EpiBench, no `scoringutils`: `src/tapestry/evaluation/totals.py`
  (`quantile_scores`) is the only scorer, against the frozen hub-ensemble task
  support in `data/evaluation/b0_hub_comparison_q23` (truth and ensemble
  quantiles frozen there; model forecasts from each run's `forecasts.npz`).
- **User decision 2026-09-22: exactly one score.** Per target and season, the
  mean of per-location WIS ratios (total model WIS / total ensemble WIS on
  identical tasks, all eligible dates and horizons 0-3): states/DC ratios
  average equally and share 1 - w, the US ratio gets w. Within a season,
  targets combine as (2 x admissions + ED) / 9 (weights 1 and .5); seasons
  count equally; configurations are the mean and SD over seeds.
  `SCORE_VERSION = location-relative-season-first-v2`.
- **Score weights are rank-time options (user decision 2026-09-22)**, not
  scenario or experiment settings: `planner rank --us-weight 0.2
  --admissions-weight 1 --ed-weight 0.5`. They are recorded in the ranking
  folder's `manifest.json` and hashed into its name
  (`ranking-<sha256(attempts, weights)[:12]>`), so rankings with different
  weights sit side by side. The plan-time `--us-score-weight` /
  `experiment.json` setting was removed; an old `experiment.json` that still
  carries `us_score_weight` is ignored by `rank`. The weights are independent
  of the loss's US weight (`model.objective.US_WEIGHT`). Scoring only ever
  uses each fold's held-out score episodes (`forecasts.npz`).
- `season_scores.csv` also carries `model_/ensemble_{dispersion,underprediction,
  overprediction}_ratio`: each WIS component divided by the ensemble's total
  WIS at the same location, location-weighted like `wis_ratio`, so the model's
  three sum to `wis_ratio`.
- **Figures** (`evaluation/plots.py`, user request 2026-09-22) are written by
  `rank` into `ranking-*/plots/` and by `planner plots -e NAME [--ranking DIR]
  [--configs A B] [--dates ...]`. Exactly four, every choice in the module
  docstring: (1) CV layout per fold from `cv.week_roles` on the finalized US
  series, one panel per target (all six, not a representative one), one file
  per distinct CV setting; (2) fans for US and NC, rows = targets, columns =
  held-out seasons, up to two configurations (default: the two best ranked,
  each at its lowest seed, seeds not pooled) plus the hub ensemble where frozen
  support exists, default reference dates = the 25th/50th/75th percentile
  positions of each season's score reference dates; (3) a seaborn PairGrid dot plot
  (replaced the pairplot, user request 2026-09-22): one row per configuration
  ranked by mean combined WIS ratio, one column per metric (WIS ratio for all,
  states/DC and US; 50/80/90/95% coverage; the three component ratios), one dot
  per seed and a bar at the mean, ensemble values dashed and nominal coverage
  dotted; (4) per selected configuration, a location x season heatmap of the
  WIS ratio (targets combined with the ranking's target weights over the
  targets available there, mean over seeds, plus the mean of seasons). Every
  other plotting script was deleted (§6).
- The pooled total-WIS ratio (sum of model WIS / sum of ensemble WIS) is no
  longer computed. Summed `model_wis`/`ensemble_wis` columns remain in
  `totals.csv` and `season_scores.csv` as raw totals only, not a ranking.
- `fit` scores against `settings['frozen']` (passed as `--frozen`), not a module
  constant. `plan` records the sha256 of `panel.npz` and of the frozen
  `manifest.json` in `experiment.json`; `run_seed` (local `run` and the Slurm
  dispatcher) refuses to fit when either changed.

## 6. New module layout

```
src/tapestry/
  data/            # unchanged: raw acquisition/versioning (repository.py, catalog.py trimmed, sources/)
  explorer/        # unchanged: browse/display over data/
  dataset/         # NEW, replaces model_data/
    extract.py     # §2 revision tables, resolve() as of a cutoff, extract()
    build.py       # §3 builds panel.npz
    episodes.py    # §3 one episode builder, input_mode switch
    cv.py          # §4 season folds by masking the panel
  model/           # NEW, replaces models/ B0/B1/B2 classes + scenarios.py stack
    network.py     # the Model nn.Module (today's B0 + B1-direct wrapping)
    scenario.py    # Scenario dataclass + string codec
    objective.py   # unchanged (fair-CRPS loss weighting)
  experiment/       # NEW, replaces models/manager.py + backends.py
    planner.py     # plan/run/status/rank/plots (+ internal fit), no model branching
    dispatch.py    # unchanged Slurm queue, moved
    provenance.py  # unchanged, moved
  evaluation/
    totals.py      # WIS and the one location-relative score (§5)
    hubs.py        # export forecasts to hub task tables
    quantiles.py   # the 23-level quantile grid (the one copy; planner imports it)
    plots.py       # the four figures (§5)
```

One path (2026-09-22): `dataset.build build` -> `planner plan` -> `planner run`
or `scripts/jlessler.sbatch` (dispatch) -> `planner rank` (scores and writes the
four figures). Deleted 2026-09-22 as part of that simplification (git history
keeps them): `evaluation/summary.py` (frozen-support competitor selection),
`evaluation/scoring.py` (its `match_forecasts` check moved into `totals.py`, the
rest was unused), `evaluation/configurations.py` and the standalone
`totals score|rank` CLI (the planner is the only entry point), the offline
Hub-mirror extraction in `hubs.py` (`extract_hub`, `GitHubSnapshot`,
`frozen_truth`, `wide_quantiles`; no caller), the duplicated quantile grid in
`planner.py`, the `cv.HORIZON_WEEKS` constant (= `len(episodes.HORIZONS)`), the
`SEASONS` re-export and unused `CALENDAR_START` in `provenance.py`, and all of
`scripts/archive/` (B0/B1/B2-era plotting, audit and launcher scripts that
imported deleted modules). The explorer (`explorer/`, `scripts/explore_covariates.py`,
`scripts/update_published_explorer.sh`) is the raw-data browser, not a results
figure, and stays (§2).

Delete: `models/b0.py` classes folded into `model/network.py`
(`models/architecture.py` merges in too), `models/b1.py`, `models/b2.py`,
`models/scenarios.py`, `models/b1_scenarios.py`, `models/b2_scenarios.py`,
`models/manager.py`, `models/backends.py`, `models/season_cv.py` (replaced
by `dataset/cv.py` + `experiment/planner.py`), `models/bundles.py` if it
only served the removed per-pathogen/per-target independent-fit machinery
for non-`all` `fit_partition` values (keep `fit_partition` support itself —
just confirm `bundles.py`'s `IndependentBundle` still applies to the unified
`Model`), `model_data/finalized.py`, `model_data/wednesday.py`,
`model_data/b2.py`, `model_data/forward.py` (folded into `dataset/build.py`
+ `dataset/extract.py`), `evaluation/epibench.py`.

CLIs: collapse `tapestry-model-data` + the model-training half of
`tapestry-experiments`/`manager.py` into one `tapestry-dataset` (build/
show/check) and one `tapestry-experiment` (plan/run/status/rank/plots)
entry point; `tapestry-data` (raw acquisition) and `tapestry-explore` are
unchanged.

## 7. What's explicitly out of scope

- No change to the fair-CRPS loss, the FiLM decoder, or the season-CV
  leave-one-out policy itself (3 seasons, refit-after-early-stopping) —
  only where that logic lives.
- No change to the score's target and season weighting (admissions 1.0/ED 0.5,
  seasons averaged equally); the US share stays 20% by default but is now an
  experiment setting (§5).
- `analysis/`, `docs/`, `references/` content is not rewritten, only
  pointers updated where they reference deleted module paths.

## 8. Decision log

**2026-09-22 — one dataset panel (user decision).** `finalized.npz` +
`vintaged.npz` replaced by `panel.npz` (§3). Why: two arrays duplicated the
truth, fixed the lookback at build time, and the vintaged build never ran
(`cutoff_time` applied twice crashed it). Regression check against the
row-by-row code: the truth panel equals the reference `finalized.npz` built the
same day cell for cell (targets, covariates, national); the as-of resolution
equals the old `VintageArchive` at all 160 Wednesday cutoffs over the full
calendar for the six targets and both inpatient claims signals, and the NWSS and
Kinsa overlays equal the old `_nwss_panel`/`_national_panel` at every issuance.
Choices flagged: covariate overlay depth D = 52 (not R) to preserve what
vintaged covariates were; one truth day and one calendar for both modes.

**2026-09-22 — cross-validation restored by masking the panel.** The
refactor's `splits.season_split` assigned whole episodes to train/val/score by
origin, so training episodes near season boundaries carried held-out-season
labels (25-33% of the loss through season-equal loss weights) and validation
weeks stayed in training inputs and labels. `dataset/cv.py` restores the
pre-refactor policy (§4) and pads the calendar start so score origins exist from
the first 2023-24 week at any lookback (lookback 12 previously failed scoring
with "Missing frozen tasks": first origin 2023-11-18, frozen tasks from
2023-10-14).

**2026-09-22 — one score (user decision).** Mean of per-location ratios with a
configurable US share (§5); the pooled total-WIS ratio is no longer computed or
ranked. This supersedes the earlier preference for the pooled total-WIS ratio
in B0 selection.

**2026-09-22 — dataset build speed.** cProfile of the old build: reading one
target (`nhsn_flu_admissions`) took 991 s under the profiler, 63% in
`describe()` called per row (dataclass `asdict` per call), 21% in
`observation_geography` per row, plus SQLite per-row Hub validation and per-row
CSV parsing. The finalized build alone took about 31 minutes; the vintaged build
never completed. Now: columnar pyarrow reads, per-row policy functions
evaluated once per unique column combination, vectorized conflict detection
and resolution, one process per source. The full panel (truth and overlay) builds
in about 65 s wall time on 12 cores.

**2026-09-22 — exact as-of store (user decision).** The 2/52-week overlay was
replaced by the value visible at each Wednesday cutoff for every calendar week,
targets and covariates alike (§3), stored as difference-from-truth
(`<name>_revised` + `<name>_values`, `np.savez_compressed`): 10.0 MB (was
4.6 MB), build 74 s wall on 12 cores. float32 kept; float16 cannot represent US
admission counts (§3). Checks: (1) `build check --samples 4` against direct
`extract()` at issuances 2024-06-26, 2025-03-19, 2025-07-30, 2026-03-18 and the
truth day 2026-09-22, all 17 sources: 0 mismatched cells (6 min on 12 cores);
(2) episodes from the new panel with the default `asof_weeks=2` equal the
episodes of the previous panel and code (commit dc93483), field for field
(values, availability, known_final, labels, covariates incl. Kinsa/NWSS/claims),
for lookbacks 4, 12 and 52 in both input modes. `Scenario.asof_weeks` added
(targets only; vintaged covariates always fully as-of); the lookback <= 52 limit
for vintaged covariates is gone.

**2026-09-22 — CV settings in the scenario (user decision).** `validation_weeks`,
`validation_spacing`, `validation_offset` replaced `cv.py` constants (§4); run ids
of existing strings unchanged (pinned in `tests/test_scenario.py`). Seasons stay
a module constant (tied to the frozen support and the export).

**2026-09-22 — score weights at rank time (user decision).** `planner rank
--us-weight --admissions-weight --ed-weight`, recorded in the ranking manifest
and hashed into the folder name; `plan --us-score-weight` removed (§5).

**2026-09-22 — no national-covariate broadcast (user decision).** A
`national_covariates=broadcast` option was proposed and dropped by the user.
Current behaviour stays: Kinsa reaches only the US row (§3).

**2026-09-22 — four figures, one path (user request and priority).**
`evaluation/plots.py` draws the four figures from `rank` (§5); every other
plotting, report and dead evaluation code and `scripts/archive/` were deleted
(§6). matplotlib and seaborn, already installed in the checkout's `.venv` but
undeclared, were added to the `evaluation` extra so `uv sync` on the cluster
keeps them.

- 2026-09-22: `rank` writes `docs/results/<experiment>/index.md` with a fixed
  layout -- figures, then a hand-written write-up kept across regenerations, then
  the ranking table. Figures are embedded as base64 PNG so they show in the page in
  any viewer (user: "not one click away"); cost: pages of a few MB.
