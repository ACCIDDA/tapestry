# Unified B0/B1/B2 restructuring (2026-09)

The model organization in §1 is extended by the implemented
[nowcast/forecast split](nowcast-forecast.md): independent stage weights and a
sampled-history interface, sharing this panel, scenario codec and planner.

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
  Kinsa, Hub current + git-mirror target data; since 2026-09-22 also Delphi
  FluView (ILINet, clinical labs) and FluSurv-NET. Delete specs for anything else
  (comprehensive/legacy Hub variants no longer scored, county sources).

**Extraction** lives in `src/tapestry/dataset/extract.py` (rewritten
2026-09-22 for speed, same policy). Each archive-backed source (six targets,
seven Delphi covariates: four claims, ILINet ILI, clinical-lab percent positive,
FluSurv-NET rate) is read once into a revision table
(`revisions(name)`: tier, release, reference Saturday, location, value) and
resolved as of any cutoff day by `resolve(revisions, day, dates)`; NWSS indices
and Kinsa, which are report-dated files rather than archives, resolve through
`resolve_reports`. `extract(name, as_of)` wraps both for ad-hoc use. The
vintage policy, stated in the module docstring and unchanged from the
row-by-row `VintageArchive` it replaced:

- Three tiers per target: native Hub `as_of` full snapshots, Hub Git-history
  full snapshots (every registered commit is a full snapshot, including empty
  ones, so deletions stay deleted), and Delphi per-observation report-time
  revisions. **Each tier states what it knew at its own latest eligible release,
  and the most recent statement wins per cell** (2026-09-22; before that the
  latest Hub snapshot won for every date once a location's Hub history started,
  which hid data Git and Delphi had already released). A Hub tier covers a
  location from the earliest reference week in any eligible release of that
  tier; inside its coverage its statement is the latest eligible snapshot --
  the value if the cell is in it, *missing* if it is not, both at that
  snapshot's release time -- and outside it the tier says nothing. Delphi
  states each cell's latest eligible revision at that revision's time (a
  missing latest revision is a statement of missing; earlier revisions are not
  used as fallback). Ties go Hub, then Git, then Delphi.
- **A Hub `as_of` that is a week-ending label, not a publication time, moves to
  its Git publication time** (`extract._publication_times`, 2026-09-22).
  FluSight's native `as_of` was the snapshot's own last reference Saturday until
  2025-07-05 and the real Wednesday publication day (week end + 4) afterwards;
  taken literally, the label made a snapshot visible up to six days early --
  leakage in vintaged mode. Such a release moves to the first Git release whose
  snapshot reaches that reference week (the commit that published it), or to
  label + 4 days if the source has no Git history, replacements kept strictly
  increasing so two snapshots never collapse into one release. The COVID and RSV
  hubs always publish week end + 4 and are untouched.
- **The Hubs distribute both target kinds in one file**, so the three NSSP
  channels take the same Hub as their admissions channel (2026-09-22): FluSight
  `target-data/time-series.csv` and the COVID/RSV hubs'
  `target-data/time-series.parquet` carry `wk inc <disease> hosp` *and*
  `wk inc <disease> prop ed visits`, picked apart by the `selection.describe`
  origin column. Hub ED values are proportions (0-1); Delphi's percentages are
  divided by 100, and the two agree to float32 rounding.
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
covariate_names:          str [K]                claims, NWSS wval_like/pct_rank, ILINet, clinical labs, FluSurv (COVARIATE_GROUPS order)
covariates:               float32 [T, L, K]      truth, NaN = unavailable
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

**What the panel does not hold (made explicit 2026-09-23).** The panel is not the
full revision history. It holds (1) the truth, i.e. every week as visible at the
end of the build day, and (2) for each Wednesday issuance, every week as visible
at that Wednesday's cutoff. Releases between two Wednesdays are superseded and
not stored, and nothing before `CALENDAR_START` (2023-09-02) is in it. The full
history stays in the raw snapshots (`data/raw/`), from which the panel is rebuilt;
`metadata.snapshots` records which ones. The explorer's Wednesday/Saturday index is
a separate, independent thinning of the same raw data.

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

**`known_final`** is not stored. In a vintaged episode a context cell within the
last `asof_weeks` weeks is never known-final (it is either what the issuance saw
or unavailable, 2026-09-22); older context weeks are known-final wherever the
truth is available. In a finalized episode every available context cell is
known-final.

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
  one was visible (known_final False) and are **unavailable where nothing was
  visible** (user decision 2026-09-22, replacing the truth fallback); older
  context weeks and all target weeks are truth; covariates (state and national)
  are as of the issuance for every context week, NaN where nothing was visible.
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
- **No truth fallback (user decision 2026-09-22, later).** A target cell in the
  as-of window that was not published at the cutoff is unavailable
  (`available=False`, `known_final=False`), never filled with later truth; the
  model's masking handles it, as it already did for covariates. This replaces
  the earlier decision to keep the fallback and **changes vintaged results
  against old B1**, whose runs saw final truth in cells that were empty at the
  time. Finalized mode is unaffected (it never used the fallback). An episode is
  still kept only if some context cell and some target cell are available, so a
  fully as-of vintaged episode whose whole window was unpublished is dropped (7
  of 160 issuances on the 2026-09-22 panel at `asof_weeks = lookback = 9`; none
  at the default `asof_weeks = 2`). Sizes:
  `python -m tapestry.dataset.analyze_dataset` ([panel analysis](../data/panel.md)).
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
  at week `validation_offset` of each training season (defaults 3/16/4:
  0-based weeks 4-6, 20-22 and 36-38). Week positions count from the season's
  first epiweek (CDC week 31, `cv.season_start`), so every season hides the same
  weeks; 2023-24 begins before the calendar (2023-09-02 is its week 4, 0-based) and
  its missing early positions simply do not exist (fixed 2026-09-22; positions
  were counted from the calendar start before). Validation episodes are cut from the training panel
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
  first week at any lookback; context before the season is the real data of the
  preceding weeks (only weeks before the calendar start are padding), and only
  labels inside the held-out season count.
- **Loss scales** (`planner.unique_truth` -> `objective.loss_scales`) use truth
  only: target cells and known-final context cells of the training episodes, one
  value per date and cell. Before 2026-09-22 the first value seen per date was
  used, which in vintaged mode was usually the earliest as-of value of that week;
  finalized scales are unchanged by the fix.
- **Covariate standardization** (`planner.covariate_scales`) pools the visible
  covariate cells of all training episodes' context windows, so a calendar cell
  is counted once per window containing it (up to `lookback` times; weeks near
  the ends of the training seasons count fewer times). Kept as is (documented
  2026-09-22, not changed).

`tests/test_dataset.py` checks that no held-out or validation week value
reaches any training input, label, loss scale/weight or covariate scale, that
validation episodes (inputs, labels, availability, covariates) never see
held-out or unused weeks, and that score labels stay inside the held-out season.
The validation check was added 2026-09-22 after a review showed that cutting
validation episodes from the unmasked panel still passed; it fails under that
mutation.

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
- **Figures** (`evaluation/plots.py`, user requests 2026-09-22) are written by
  `rank` into `ranking-*/plots/` and by `planner plots -e NAME [--ranking DIR]
  [--configs A B ...] [--dates ...]`; every choice is in the module docstring.
  Labels: each configuration is `C<k>`, k = its ranking position (1 = best);
  figures show the label plus the scenario string wrapped at commas (~40
  characters a line), and the report appendix is the key. Selected
  configurations (fans, heatmaps): the best ranked only by default; `--configs`
  replaces the selection, any number; fans use each one's lowest seed. (1) CV
  layout per fold from `cv.week_roles` on the finalized US series, one panel per
  target, one file per distinct CV setting; (2) fans, one file per location (US,
  NC) x target kind (`fans-US-hosp.png`, `fans-US-ed.png`, ... -- admissions and
  ED proportions have different units): rows = disease, columns = hub ensemble
  then one per selected configuration, x = time across the three held-out
  seasons, finalized truth in every panel, 50%/90% bands and median per
  reference date, y limits shared within a row; default reference dates every
  4th score reference date of each held-out season (`--dates` overrides); the
  ensemble only where frozen support exists; (3) seaborn PairGrid dot plot: one
  row per configuration plus a hub-ensemble row (WIS ratio 1, its coverage and
  its decomposition over itself), ordered by the main score, opaque dot = median
  over seeds, light dots = seeds, columns = WIS ratio all/states-DC/US,
  50/80/90/95% coverage, the three component ratios; red dotted = nominal
  coverage; height scales with rows x label lines, width with metric columns;
  (4) per selected configuration, one heatmap file with one panel per target
  (rows = disease, columns = admissions/ED): WIS ratio by location x season,
  mean over seeds, plus the mean of seasons with support; one log colour scale
  centred at 1 shared by all panels of all heatmap files of the call, grey = no
  frozen support. Every other plotting script was deleted (§6). Heatmap files
  are named `heatmap-C<k>.png` (short label only: scenario strings can exceed
  file-name limits); their titles and report headings carry `C<k>` and the full
  scenario string, like the appendix.
- **Report** (`plots.write_report`, called by `rank` only for the complete
  ranking: every planned run complete and ranked, default score weights; a
  `--seeds` subset, `--allow-incomplete` with missing runs or non-default
  weights still gets its ranking folder and figures, and `rank` prints why no
  page was written. An existing page without both write-up markers raises
  instead of being overwritten):
  `docs/results/<experiment>/index.md` = summary, figures (base64-embedded; CV
  layout, fans, dot plot, heatmaps), the preserved write-up, the ranking table
  (with labels), then "Appendix: scenarios run" (label, full scenario string,
  run id, seeds, non-default fields with their defaults). The appendix links
  `../../reference/scenario.md`, the field key that `write_report` regenerates
  each time via `write_scenario_key` at `<root>/../reference/scenario.md`
  (`docs/reference/scenario.md` for the default root; in the mkdocs nav under
  Reference). The key is built from `Scenario` fields, `scenario.CODES` and the
  one-line `scenario.MEANING` taken from existing docs (pointers to the defining
  document where one line is not enough). Chosen over a separate `planner
  scenario-key` command: no new command, and the key can never be stale relative
  to a report.
- The pooled total-WIS ratio (sum of model WIS / sum of ensemble WIS) is no
  longer computed. Summed `model_wis`/`ensemble_wis` columns remain in
  `totals.csv` and `season_scores.csv` as raw totals only, not a ranking.
- `fit` scores against `settings['frozen']` (passed as `--frozen`), not a module
  constant. `plan` records the sha256 of `panel.npz`, of the frozen
  `manifest.json` and of the one population file (`data/metadata/locations.csv`,
  `planner.LOCATIONS`; user decision 2026-09-22) in `experiment.json`;
  `run_seed` (local `run` and the Slurm dispatcher) refuses to fit when any
  changed. The population file and the frozen support are git-ignored, so not
  synced to the cluster with the code (docs/longleaf-setup.md). The Slurm
  launcher runs the code snapshot `plan` pinned in `<experiment>/code`; a local
  `planner run` runs the working tree.
- **Evaluation** (`training.evaluate`): 256 members by default
  (`--eval-members`), drawn in forward passes of 32 (`EVAL_CHUNK`, bounds memory);
  the 23 quantiles are taken over the draws, and admission quantiles are rounded
  to integers (counts), ED proportions are not. `plan` defaults to seeds 42 43 44.
- **Loss and score weights share numbers, not meaning.** The score defaults
  (`totals.US_SCORE_WEIGHT, ADMISSIONS_WEIGHT, ED_WEIGHT` = .2, 1, .5) are defined
  once; `model.objective` builds its default loss weights (`TARGET_WEIGHTS`,
  `US_WEIGHT`) from them. The loss's channel weights stay a scenario choice
  (`loss_weights`), the score's weights rank-time options.
- **Which targets each season scores** (frozen support, not a choice made here):
  2023-24 only flu admissions; 2024-25 flu and COVID admissions; 2025-26 all six
  targets. The (2 x admissions + ED) / 9 weighting therefore fully applies only
  to 2025-26; the other seasons' composites are the weighted mean of the targets
  they have (renormalized), and seasons still count equally.
- `build check` compares the stored panel with `extract.extract`, which runs the
  same `resolve`/`resolve_reports` functions as the build: it verifies the panel's
  assembly and storage (calendar, as-of layout, encoding), not the vintage policy
  itself (that was checked once against the row-by-row code, decision log).

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
  seasons averaged equally); the US share stays 20% by default and, like the
  target weights, is a `rank`-time option (§5).
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

**2026-09-22 — report figures and appendix (user requests).** Fans: reference
dates every 4 weeks, one file per location x target kind, rows = disease, the
ensemble and each configuration in separate columns with row-shared y limits,
default the best configuration only (`--configs` adds more, lowest seed each).
Heatmaps: one panel per target, shared log colour scale centred at 1. Dot plot:
a hub-ensemble row, median over seeds opaque and seeds light, size adapting to
rows and columns; long scenario strings shown as `C<k>` + string wrapped at
commas, the report appendix being the key. `write_report` appends "Appendix:
scenarios run" and regenerates `docs/reference/scenario.md` from `Scenario`
(`scenario.MEANING` added). Checked on the smoke experiment
(`ranking-c08a5acf2c27`, 4 configurations, seeds 42/43), default and all four
configurations. Choices in §5 and the `plots.py` docstring.

**2026-09-22 — truth fallback kept, size documented (user decision).** In
vintaged mode, target cells in the as-of window with nothing visible at the
Wednesday cutoff keep the final-truth fallback flagged `known_final=True`
(reproduces old B1). `src/tapestry/dataset/analyze_dataset.py` (rerun after
every panel rebuild) writes `docs/data/panel.md`: fallback share by target,
season and weeks before issuance (states+US and US), per issuance, revision size
where an as-of value exists, covariate as-of availability, and US/NC covariate
series. It cuts episodes with the episode builder itself (`lookback = asof_weeks
= 9`), so fallback is exactly `known_final` inside the as-of window. On the
2026-09-22 panel the fallback is mostly all-or-nothing per issuance (missing
archives: no NHSN covid/RSV as-of in 2023-24, NSSP context-end week not
visible at most 2023-25 cutoffs); pooled over seasons, states+US, j = 0/1:
NHSN covid 47/43%, flu 35/27%, RSV 38/34%, NSSP 67/35%.

**2026-09-22 — review clean-up (user approved all fixes).**
(1) Leakage test extended to validation episodes; it now fails if validation
episodes are cut from the unmasked panel. (2) `plan` snapshots code only after
the settings check, so a rejected re-plan keeps the pinned code. (3)
`write_report` raises when an existing page lacks a write-up marker. (4) `fit`
raises on a nonfinite validation loss, on a component with no weighted
validation labels, and never refits with 0 epochs. (5) Validation week positions
count from the season's first epiweek; with the defaults, 2023-24's hidden weeks
moved from 2023-09-30/10-07/10-14, 2024-01-20..02-03 and 2024-05-11..05-25 to
2023-09-02..09-16, 2023-12-23..2024-01-06 and 2024-04-13..04-27, the same season
weeks as 2024-25's (2024-08-31..09-14, 2024-12-21..2025-01-04, 2025-04-12..04-26);
fits with `patience > 0` change for folds training on 2023-24.
(6) `rank` writes the report only for the complete default ranking. (7) Heatmap
files `heatmap-C<k>.png`, headings `C<k>` + scenario string. Loss scales now use
truth only (vintaged scales change; finalized unchanged). One population file:
`data/metadata/b1_locations.csv` (byte-identical, sha256 80aaa247...) moved out,
`locations.csv` kept, its sha256 pinned at plan time (experiments planned before
lack the hash and must be re-planned to run). `status` prints the resume
commands. Deleted dead code: `totals.case_totals`, `cv.ROLES`,
`planner.ARRAY_CHUNK`, `evaluate(name=)`, `Scenario.flags`,
`extract.COVARIATE_SOURCES`, `extract.saturday`, `quantiles.select_quantiles`
(the saved levels are exactly `LEVELS`; `export` now checks that instead),
`covariate_mask` (removed from the panel format; panel rebuilt with the same
truth day, every other array identical; `build check` below), the snapshot
copies of `scripts/*.sbatch` (Slurm runs the working-tree launchers), and the
empty `src/tapestry/models` and `model_data` directories. The six-channel count
comes from `build.CHANNELS` in `objective.py` and `planner.py`.
Checks: `build check --samples 4` on the rebuilt panel (issuances 2024-06-26,
2025-03-19, 2025-07-30, 2026-03-18 and truth day 2026-09-22): exact, 0 mismatched
cells; the rebuilt panel equals the previous one array for array. pytest: 68 passed
(the known `test_hub_history.py` explorer failure remains). Smoke (`epochs=2`,
`epochs=4,patience=2`, seeds 42/43, 32 evaluation members): `rank --seeds 42` and
`rank --us-weight 0.5` wrote rankings and figures but no page, the complete default
`rank` wrote it; a page missing a marker raised.

### Wastewater indices regenerable again, legacy docs, explorer as-of test — 2026-09-22

**User requests.** (1) The derived wastewater indices must be rebuildable by one
command in the current workflow; (2) the failing explorer test must be fixed at
its root cause; (3) every pre-restructure page moves into a "Legacy (v0)" section.

(1) `src/tapestry/dataset/nwss.py` restores the deleted
`model_data/b2.py:build_nwss_covariates` (removed in `4a0b172`) unchanged in policy,
as `python -m tapestry.dataset.build nwss-indices --data-root data`. Decisions:
a separate subcommand, not part of `build build`, because it streams the 1.2 GB
auxiliary archive (11 minutes, 12 GB peak RSS here) while its inputs change only
when NWSS is re-pulled; the two index formulas (`score_wval`, `score_pct_rank`)
and the 26-week/3-site thresholds live in that production module and are imported
by `analysis/covariates/indices.py`, so `analysis/` is not a build dependency and
the formulas cannot drift. Reproduction from the same input snapshots: 880,352 rows
against the stored 880,384, every shared row bit-identical; the 32 stored rows not
reproduced are one US row per report time for reference week 2020-02-29 with
`n_sites = 1`, excluded by the committed >= 3 site rule (so the stored snapshot came
from a variant without that rule nationally). That week precedes the calendar start,
and the rebuilt `panel.npz` is identical to the previous one array for array (only
the recorded snapshot id and timings differ); `build check --samples 4`: exact.
`analyze_dataset` gained §5: both indices per pathogen for US, NC, CA, NY, TX
(final line, as-of dots) and state coverage per week.

(2) `tests/test_hub_history.py` asserted that the explorer shows a Thursday
revision on Thursday. `ExplorerIndex.data` rounds an as-of date back to the
preceding Wednesday or Saturday end of day (`_cutoff`), which is the documented
API rule (explorer/overview.md: "The API applies the same rule", never forward),
so the expectation was wrong, not the code: the test now asserts the Thursday view
equals the Wednesday one and that Saturday sees the Friday whole-file deletion.
The `extract.resolve` assertions on the exact days are unchanged and still cover
the revision and the deletion.

(3) `docs/legacy-v0/` holds the pre-2026-09-22 B0/B0.1/B1/B2 design, data, results
and workflow pages (the old `docs/archive/workflows-2026-09` became
`legacy-v0/workflows`), keeping their relative layout so links between them survive;
`docs/legacy-v0/index.md` explains the boundary. `design/architecture.md` stays in
the main nav: it describes the network, masks and CRPS objective that `Model` still
implements. All relative links were rewritten mechanically (resolve each link at the
page's old location, repoint to the target's new one) and `mkdocs build --strict`
passes, which it did not before this change (30 warnings, mostly links broken by the
earlier archive move).

### Hub vintages stop shadowing; Hub ED targets wired; no truth fallback — 2026-09-22 (later)

**User premise.** For every Hub round, the versioned truth modelers had at that
moment is in the Hub's Git repository; wherever a round ran, a vintage must exist
from Git, and a truth fallback there is our bug, not missing data. Where no round
ran (the 2025 federal-shutdown Wednesdays 2025-10-08..11-12) there is genuinely
nothing, and no ensemble to score either.

**What our acquisition actually holds** (verified against the mirrors
`data/mirrors/*.git` and the selected snapshots; nothing was missing, so nothing
was acquired):

- `hub_flusight_current`: `target-data/time-series.csv` (native `as_of`, 96
  snapshots 2023-09-23..2026-07-08, both targets) and Git history of
  `target-data/target-hospital-admissions.csv` + `target-ed-visits-prop.csv`
  (132 releases 2023-10-03..2026-07-09). The dated
  `auxiliary-data/target-data-archive/*_<Saturday>.csv` copies are the same
  snapshots under their week-ending label and add nothing.
- `hub_covid_current`: `target-data/time-series.parquet` (87 `as_of`, 2024-11-20
  onwards for admissions, 2025-06-18 for ED) + 93 Git releases of
  `covid-hospital-admissions.csv` (2024-11-18 onwards). Nothing earlier exists:
  the repository itself starts 2024-11.
- `hub_rsv_current`: `target-data/time-series.parquet` only (87 `as_of`,
  2024-11-27 admissions, 2025-06-13 ED). Its **0 Git target releases are correct,
  not a missing acquisition**: the repository starts 2025-08-14 and has never
  contained a non-`as_of` target file (`HISTORY_TARGETS['hub_rsv_current'] = {}`).
  `hub_rsvnet` is RSV-NET rates, a different target, dropped in §2.

So the pre-2024-11 NHSN covid/RSV gap and the pre-2025-06 NSSP Hub gap are
genuine (no hub, no target file), while three real defects hid data we hold:

1. **A stale Hub snapshot shadowed Git and Delphi.** Once a location's Hub
   history started, the latest Hub snapshot won for every later date and dates
   absent from it became unavailable. Fixed by resolving each cell to the most
   recent release across tiers (§2). Effects: flu truth ran to 2026-07-04 (the
   last FluSight snapshot) and now runs to 2026-09-05; the flu truth panel gains
   468 cells; 434 older flu cells and 43 covid / 5 RSV recent cells now take a
   newer Delphi revision instead of an older Hub snapshot (differences of a few
   admissions, NHSN revisions, no definition jump). NSSP truth changes only by
   float32 rounding (max relative 1.2e-07), so Hub proportions and Delphi
   percentages/100 agree.
2. **FluSight's 2023-24..2025-06 `as_of` was the week-ending Saturday**, not the
   publication day (lag 0 until 2025-07-05, then the real Wednesday, lag 4), so a
   snapshot appeared up to six days before it existed — leakage in vintaged mode.
   Each such release now moves to the first Git release reaching that week (§2).
   Verified against Git: week ending 2023-09-30 moves from the label to
   2023-10-06T22:36Z, the commit that added those 52 rows, so the 2023-10-04
   issuance now sees only through 2023-09-23; 2023-10-07 moves to 2023-10-11.
3. **The three NSSP channels ignored the Hub ED target files.** They are in the
   same Hub files as admissions and are now wired (§2).

**No truth fallback (user decision).** Vintaged cells that were not published at
the cutoff are unavailable instead of final truth flagged `known_final=True`
(§3). Downstream nothing assumed the fallback: `cv.fold` masks by reference week,
`planner.unique_truth`/`objective.loss_scales` read truth from `known_final`
cells only (now fewer), and the plots and `analyze_dataset` read availability.
Finalized mode is untouched. `analyze_dataset` now reports the *unpublished*
share over the same denominator, so it stays comparable.

**Coverage, states+US, pooled over seasons, j = 0/1 (old fallback % -> new
unpublished %):** NHSN flu 35.3/27.0 -> 25.2/20.6, covid 46.9/43.3 ->
45.5/42.9, RSV 38.1/33.8 -> 37.3/33.8; NSSP flu 67.0/35.1 -> 66.0/34.3, covid
67.0/35.1 -> 65.2/33.5, RSV 66.8/34.8 -> 65.2/33.5. The remaining NSSP j = 0
share is concentrated in 2023-25, where no Hub ED file existed and Delphi had not
released the context-end week by Wednesday. Whole-issuance gaps in 2025-26 are
now exactly the six shutdown Wednesdays 2025-10-08..11-12 (five for NHSN flu,
which came back on 11-12; NSSP flu also lacks 2025-10-01, before FluSight had an
ED target file). Earlier whole-issuance gaps are all explained: NHSN flu
2024-05-15..2024-11-13 is the voluntary-reporting pause, NHSN covid/RSV before
2024-11-20 and NSSP before Delphi's archive have no source at all.

Rebuilt panel: `build check` exact, 69 tests pass, and a two-epoch CPU smoke
experiment (one finalized, one `input_mode=vintaged`, seed 42) ran through
plan/run/rank with all figures.

### FluView and FluSurv-NET from Delphi V5 — 2026-09-22

**User request.** Add the new Delphi V5 FluView ILINet, FluView clinical and
FluSurv sources to the explorer, the dataset and the documentation. Delphi
announced the V3 → V5 move in an email to the user (2026-09-19): V3
`pub_fluview` → `fluview_ilinet`, `pub_fluview_clinical` →
`fluview_resp_lab_clinical` + `fluview_resp_lab_ph`, `pub_flusurv` → `flusurv`.

**Acquisition.** Four catalog specs on the existing `delphi_v5` fetcher (no new
code path): `delphi_fluview_ilinet`, `delphi_fluview_clinical`,
`delphi_fluview_ph`, `delphi_flusurv`, full archives, all published signals at
state/nation/regional support. The public-health-lab source is included
although the user named three sources, because Delphi's email splits the V3
clinical endpoint into both; it was meant to be acquisition and explorer only
(its state rows are season totals). **Update, same day (user decision):** the
public-health labs are dropped from the explorer too, so the spec, its raw
snapshot and its selection labels were deleted; `SelectedData` still ignores a
leftover `delphi_fluview_ph` snapshot on other data roots. The selection filter that hid V3-era
`delphi_fluview_clinical`/`delphi_flusurv` migrations was removed: those keys
now name the V5 datasets. Only the V3 `delphi_fluview` key stays excluded.

**Explorer.** Selection policy v6: families *FluView* (ILINet under
influenza-like illness, laboratory signals under influenza; signal keys carry
the component prefix because both lab sources publish `positive_b` and
`total_specimens`) and *FluSurv-NET* (rates labelled by stratum).

**Dataset.** `CLAIMS_SOURCES` became `DELPHI_COVARIATES` (one Delphi signal per
covariate, no Hub tier) with three additions: `ilinet_ili` (unweighted ILI;
weighted ILI is suppressed at state level), `clinical_lab_flu_pct_positive`,
`flusurv_flu_rate` (`rate_overall`). New `COVARIATE_GROUPS`: `ilinet`,
`clinical_lab`, `flusurv`. Two extraction rules were needed and apply to Delphi
archives generally: rows with `age_group` other than `all` are dropped (ILINet
visit counts are age-stratified; without the filter, strata sharing a release
would collide and become unavailable), and `DELPHI_FILLS` admits
`nyc_plus_ny_minus_nyc` for ILINet only, because statewide New York exists only
as Delphi's pool of CDC's two New York jurisdictions. Panel metadata version 4;
stored covariates now follow `COVARIATE_GROUPS` order (consumers select by
name, so existing scenarios are unaffected).

**Measured.** FluView is released on the Friday after the week ends, so j = 0
is never visible at a Wednesday cutoff (99.3% / 99.4% not visible) and j = 1
nearly always is (ILINet 6.5%, clinical labs 14.0% not visible). Delphi's
FluSurv archive has no vintage from 2020-11-06 to 2025-11-03 nor from then to
2026-02-03: as-of FluSurv exists at 46 of 160 issuances, the first 2025-11-05.
FluView has no vintage from 2025-09-26 to 2025-11-14, the same shutdown window
as the NHSN/NSSP gaps above (a publication pause is inferred, not verified).
Rebuilt panel: `build check` exact; target unpublished shares identical to the
previous entry. No scenario uses the new groups yet. Details:
[FluView and FluSurv-NET](../data/fluview-flusurv.md).


**2026-09-25 — B0 forensic scoring and normalization audit.** Rescoring saved
B0.1 target-cap-100 and pathogen-cap-300 forecasts (three seeds each) through
the current scorer reproduces 0.883377 and 0.894767 on 56,662 identical frozen
tasks per run. Independent pinball WIS agrees. No historical configuration
wins all supported target–season comparisons (or all six target-specific
season averages); “unicorn” was an overstatement of aggregate performance.
The old finalized panel matches its original run hash; shared target values
are unchanged within floating precision across the CV seasons.

The current training model-options builder omitted historical fitted target
input scales and ED logit centers, leaving defaults 1/0. Experiment
`b0-normalization-audit` restores those statistics only in its pinned snapshot,
for the latest leader/control and seeds 42–44; production code is unchanged.
Its finalized-available rule unions observed fitting-context cells by date,
without held-out/validation values. Matched performance results are pending.
Other verified changes include 2,048 versus 256 evaluation draws and independent
component loss normalization (current pathogen mass 1/3, historical 1; the
audited season shares remain equal). See the [audit report](../results/b0-audit/index.md)
for assumptions, evidence, plots and exact plan/launch/status/rank commands.


**2026-09-25 — normalization-only results complete.** All six runs of
`b0-normalization-audit` finished. Against identical benchmark support and paired
seeds, restoring B0 input normalization changes the latest no-covariate control
from 1.124000 to 1.082407 (2/3 seeds improve), and the best covariate formulation
from 1.036003 to 1.047787 (1/3 improves). Every normalized seed remains above
ensemble parity. COVID ED improves markedly but remains poor; other targets
show mixed changes. The normalization omission is real but does not explain
most of the B0 performance gap in this intervention. Production code remains
unchanged. These three-seed means are descriptive; other B0/current differences
are not isolated. The [audit report](../results/b0-audit/index.md#matched-normalization-only-experiment)
contains paired CSVs, the graph, protocol assumptions and manager commands.


**2026-09-25 — B0 reproduction and controlled comparisons complete.** All 18
hardware-matched folds reproduce 25,833,600 quantile values byte-for-byte, with
zero maximum difference and identical stopping epochs. Dates, locations, truth,
masks and quantile levels also match. Historical target/pathogen combined WIS
is recovered at 0.883377/0.894767. Existing jobs finished; no re-planning or new
training launches were needed. Production training code remains unchanged.

On L40s with pathogen cap300 and seeds 42–44, old training scores 0.888892,
current unscaled/finalized training 0.974340, restored normalization/finalized
0.929660, and restored normalization/Wednesday availability 1.198981. Paired
mean deltas are +0.085449, −0.044680 and +0.269321. Normalization improves two
seeds; availability worsens all three and all nine target–season means. The
first intervention includes the normalization omission; after restoring it,
a +0.040768 gap to old training remains. Remaining loss/stopping/random-draw
differences are not individually identified.

These are conditional sequential comparisons on the same frozen support, not
an order-independent decomposition of the latest covariate screen. Availability
changes both fitting and forecasting and the fitted normalization statistics.
Archive absence does not establish nonpublication. Exact retrospective
repeatability does not establish real-time superiority: finalized revisions,
exploratory selected seasons, a frozen benchmark and only three seeds remain
material assumptions. No score-math error was found; no historical formulation
wins every target–season. Full evidence, graphs and manager commands are in the
[completed reproduction report](../results/b0-reproduction/index.md).


**2026-09-25 — remaining B0 fitting differences, loss multiplier held fixed.**
User requested experiments on other fitting changes and interpretation under
2025–26-like future availability. A 2×2 comparison crosses B0 validation calendar
(first observed season week) and B0 validation latent draws (separate CPU generator,
component seed + 2000). Reuse the completed normalized/full-finalized current-code
baseline; new `b0-fitting-split`, `b0-fitting-draws`, `b0-fitting-split-draws`
experiments have three seeds each, 2,048 forecast draws, and L40 hardware. Slurm
jobs 2488021/2488022/2488023. New snapshots derive from the saved normalized
baseline's source. No production fitting change or existing re-plan.

Preflight found no examples lacking all outcomes for their pathogen in either
validation-calendar policy: 54 pathogen/fold/phase/calendar cases, zero removable
episodes. The historical filtering rule is thus a no-op for these finalized
fits; no redundant filter arms were launched. Fixed loss mass remains 1/3 in
all cases. Checks matched B0 validation dates/draws and verified split exclusion
and unchanged dataset/benchmark/population settings. Paired effects at each
factor level and interactions will be reported after completion.

The latest own-target input was present on 100% of 2025–26 scored admissions
cases, but only 94.6% flu ED, 98.6% COVID ED and 95.1% RSV ED. These separate
scored-support denominators do not imply all channels/history weeks were present
at once. Full-calendar fitting retention is lower. Assuming similar future
availability makes complete-history fitting worth evaluating, but does not
establish optimal fitting or the availability of finalized revisions at issuance.
Earlier availability interventions jointly changed training and forecasting;
this new fitting experiment does not silently claim to separate those stages.
See [experiment design, evidence and commands](../results/b0-fitting-controls/index.md).


**2026-09-25 — Christmas availability interpretation corrected.** User noted
changed Hub holiday deadlines. COVID/RSV document December 29 deadline/data
release for reference December 27; FluSight extended its accepted window.
The fixed-Wednesday audit incorrectly treated December 24 input absence as
evidence of absence at the Hub deadline. Actual December 29 Git states contain
December 20 ED values for all modeled locations except Missouri in FluSight and
COVID. RSV's canonical time-series update lagged, but its raw NSSP file in the
December 29 Git state already held December 20 RSV values for 51 modeled
locations (all except Missouri); this directly confirms availability by the
extended deadline for all three pathogens. Earlier percentages and the 0.930 → 1.199 intervention
measure the fixed-Wednesday policy, not a complete operational Hub-deadline
reconstruction. Do not attribute that full score effect to real-time missingness.
Saved experiments and finalized-input fitting controls remain unchanged; an
operational comparison must explicitly model per-round deadlines/timezones in a
new data policy. See [holiday correction](../results/b0-fitting-controls/hub-ed-check.md).


**2026-09-25 — remaining fitting controls complete.** All nine new L40 runs
(jobs 2488021–2488023) completed; the comparison reuses three normalized/finalized
baseline runs, totaling 36 evaluated folds. Current baseline 0.929660, B0
validation-calendar-only 0.910424, B0 validation-randomness-only 0.929660, both
0.911037. The calendar improves combined WIS in all three seeds and closes 47%
of the numerical gap to original B0/L40 (0.888892), leaving 0.021532. Validation
randomness has no scored effect under the current calendar and a small effect
under the B0 calendar. This is not a claim of identical forecasts for targets
outside scoring support. Episode filtering is a verified no-op; loss multiplier
was unchanged and untested.

On held-out 2025–26, the calendar changes mean WIS 0.906485 → 0.884676, but seed
42 supplies the improvement (−0.090255); seeds 43/44 worsen (+0.003031/+0.021795).
The original B0/L40 2025–26 mean is 0.900254. Thus copying the entire original
fitting implementation is not established as best for the most recent season.
Three seeds, finalized inputs and exploratory season/model selection limit the
conclusion. Holiday-aware input availability remains a separate unresolved
operational comparison. No production changes or further training launches.
[Completed report and graphs](../results/b0-fitting-controls/index.md).


**2026-09-25 — B0 report consolidated.** The [complete B0 investigation](../results/b0-reproduction/index.md) now contains the scoring audit, all 18 exact fold comparisons, normalization and original-data controls, completed fitting factorial, holiday/Git correction, defined terms, graphs, assumptions, evidence links and exact manager commands. Former audit, fitting and Git-check pages are pointers; evidence artifacts retain their existing paths. This documentation consolidation changes no forecasts, scores or experiment snapshots and launches no jobs.


**2026-09-25 — B0-to-B1 narrative revision.** At the user's request, the consolidated report now moves from small implementation effects through validation weeks, normalization and input availability to the B1 design and later forecasting setup. Definitions are introduced in context. The glossary, command appendix and detailed reproduction mechanics were removed from the report; measurements and scientific caveats remain. This is a narrative revision only.


**2026-09-25 — B1-to-B0 comparison clarified.** The report now starts with a table and follows measured reversals toward B0. It explicitly separates the later-model normalization baseline from the original-data controls; no additive attribution is claimed across that break. The consecutive controlled sequence is 1.198981 → 0.929660 → 0.910424 → 0.911037, followed by the bundled original-B0 reference 0.888892. Commands and reproduction mechanics remain outside the narrative.


**2026-09-25 — Consecutive comparison completed, replacing disconnected baselines.** All 30 fixed-draw scoring runs (90 folds) completed on L40. The nine useful table rows run from a newly matched B1-derived direct baseline 1.098300 through normalization 1.121097, no artificial masking 1.194839, no finality flag 1.270849, complete forecast inputs 1.070263, complete training histories 0.933389, B0 validation weeks 0.914759, B0 validation draws 0.915398, and B0 error mass 0.888892. Additional reduction/layout controls produce identical fitted models and forecasts in all nine seed/season cases, so they are omitted from the report. Restoring error mass closes the original-B0 gap; the earlier CPU minibatch rounding discrepancy does not affect these completed GPU fits. Original 18-fold reproduction remains verified.

The rewritten fitter had not reset evaluation randomness as B0 did. Every chain stage was therefore reforecast from its saved model with seed + 1000 and the common full-finalized origin calendar; restricted stages mask inputs on that calendar. Production code is unchanged. Corrected 2025–26 availability unions checked Hub Git canonical/direct files, raw NSSP CSV/Parquet and dated Delphi reports, including holiday deadlines. The common earliest Hub cutoff has the same visibility masks as the individual cutoffs. Earlier seasons retain the original Wednesday archive; all supplied numerical values remain finalized. Missing scored latest inputs are exclusively Missouri ED (COVID 21, flu 28, RSV 29), with no admissions gaps. Complete training histories improve 2025–26 1.338363→0.906519 across all targets/seeds; B0 calendar gives 0.885678, but its aggregate gain is seed-dependent. B0 error mass worsens that season to 0.900254. The [single report](../results/b0-reproduction/index.md) starts with the consecutive table, explains the validation dates relative to peaks, and includes graphs and every missing week/location. Redundant experiments and reproduction mechanics are excluded from the narrative.


**2026-09-27 — Direct covariate research protocol.** Added separate complete-finalized
fitting and deadline-input validation/evaluation policies, opt-in B0 target input
normalization and its fixed validation calendar, and a reset evaluation seed.
The source/architecture study retains the smaller component objective, fixed
validation draws, finality flag, and target missingness in 50% of fitting examples.
User clarified ILI-only, not ED input removal, and authorized expected source
availability based on 2025–26 when archives are incomplete. Actual vintages remain
preferred; lag-gated finalized substitutes, explicit withdrawals and geographic
support are recorded in the data evidence. National Kinsa remains broadcast to
all locations. New national/gated-pool messages complement existing attention;
scoped target/pathogen/joint tokens now carry covariates and mask unavailable
senders. All models are freshly fitted; old rankings are not matched controls.
See [study design and launch record](b2-direct-research.md).

**2026-09-28 — Direct covariate study completed.** All 468 runs (156 configurations,
three seeds, 1,404 outer folds) completed with 256 evaluation draws. The final
ranking `ranking-ac92a3c7dbcc` selects target multiscale/all sources/pooled
(WIS ratio 1.059). Its matched covariate improvement is 9.0% overall, concentrated
in 2024–25; only one seed improves in 2025–26. Simple pooling and national
broadcasts are more consistent than scoped target attention. No single source
helps every backbone, and leave-one-out attribution was not run inside the
winning pooled architecture. Nominal 95% coverage is 79.3%. Results retain the
authorized availability assumptions, including lag-gated finalized proxies
where historical archives are missing. No prospective superiority is claimed.
The [report and fixed-seed fan plots](../results/b2-direct-research-v1/index.md)
use all completed seeds for ranking and seed 42's saved 256-draw forecasts for
illustration; no refitting or recalibration was performed.

**2026-09-28 — Covariate availability and revision audit.** Added a 2025–26 table for all 12 context lags, using actual archived reports at holiday-adjusted Hub deadlines and no operational proxies. Counts use native location × issuance opportunities, with national Kinsa counted once. Revision summaries compare finite same-location/week pairs to the frozen final panel, and disclose claims cells with reported values but missing final counterparts. Archive gaps are not interpreted as proven upstream unavailability. See [tables and downloadable cells](../data/availability/index.md).

**2026-09-28 — Availability audit correction.** Raw claims archives contain conflicting finite values sharing a report date, reference date and location. The shared extraction conflict policy turns these into missing for both deadline and final arrays. Availability reporting now separates finite publisher reports from unambiguous pipeline inputs; claims revision estimates remain explicitly conditional on the retained subset. ILI gaps were checked directly in the raw archive and with a read-only Delphi query. Daily claims are sampled on Saturday, daily Kinsa is weekly averaged, and the remaining modeled signals are weekly. No frozen model data or fitted results were changed and no within-date ordering was assumed. See [diagnosis](../data/availability/conflict-audit.md).

**2026-09-28 — Full missingness tables.** Expanded corrected source availability to every one of the 14 panel covariates and all 12 history lags. Reports distinguish unique missing states, distinct submission weeks across states, per-state missing weeks, DC and national gaps, and structurally unsupported geographies. Exact missing submission and observation dates are exported. Revision tables retain the explicit unambiguous-pair restriction. See [full tables](../data/availability/full-availability-summary.csv).

**2026-09-28 — Season availability timelines.** Added full observation-week × submission-date matrices for all 14 covariates and six target series, extending submission rounds through the frozen panel end. Colors distinguish complete native support, 1–2, 3–10, >10 missing locations, no archived observations, and future weeks. Claims use raw finite-report presence; target series use the joint Hub/Delphi deadline reconstruction. Native national coverage is shown separately. Retrospective final-panel coverage is not presented as proof of historical dashboard availability. Static plots and an interactive missing-location explorer are linked from the availability report.

**2026-09-28 — Distinguish availability evidence.** Added separate unknown-timing, explicit-missing-then-later-observed, explicit-missing-with-no-later-report, and never-observed categories. A later archive entry alone is not treated as proof of delayed first publication. Crosses mark explicit missing or never observed, distinguished by color; neither is described as permanent absence. Gray periods require explicit next-season availability assumptions. State selection and per-cell evidence accompany all 20 series.

**2026-09-28 — Add 2022–23 training data.** Rebuilt the default panel from May 14, 2022 with all six target channels and 14 covariate series. Finite CDC finalized NHSN counts take precedence for retrospective admissions; archive values remain where CDC final has no finite value. Historical as-of arrays are not truth-filled. Existing-period target, covariate and vintage values are unchanged. The added season has complete flu/COVID admissions, all three ED outcomes from October 1, and no RSV admissions. Training includes 2022–23; evaluation remains the three existing seasons. The base dataset is extended; completed experiment pins and its separate operational/deadline panels are unchanged. See [coverage and verification](../data/index.md#current-dataset).

**2026-09-28 — Consolidated Data documentation.** Dataset coverage and extension, availability evidence and heatmaps, full lag/state tables, revisions, source archive dates, and availability provenance now live under the single Data navigation section. The landing page distinguishes the expanded base dataset from the frozen B2 audit. Source and acquisition documentation and the general explorer are organized beneath it. Old experiment-data pages point to the canonical pages, and old HTML staircase explorer routes redirect. Report generators write to the new locations.
