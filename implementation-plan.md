# Multi-source mask-native forecasting — implementation plan

Companion to [Architecture candidates](architecture-candidates.md), which decides *what* to build and why.
This one decides *how*, in enough detail to start writing code on Monday: schemas, tensor shapes,
loss definitions, hyperparameter defaults, test invariants, gates, and a dated schedule.

Written 2026-09-04. Target: FluSight 2026-2027, hub expected to open ~early November 2026.
Working assumption: **9 weeks of calendar time**, one developer, one GPU.

**Acquisition update, 2026-09-11:** the active Delphi sources are V5 NHSN,
NSSP, NWSS, and inpatient/outpatient claims. Older
Delphi API code has been removed. Historical ILI/ILI+/FluSurv experiments below
are conditional on future V5 adapters; the current source catalog is documented
in [data sources](docs/data/sources.md). CDC and Hubverse feeds remain available.

---

## 0. How to read this document

Every design decision carries a provenance tag naming where it comes from. The user-facing
requirement for this plan was to make attribution explicit, so nothing is asserted without a source
or an explicit `[NEW]`.

| Tag | Source |
|---|---|
| `[FLU]` | Flusion — Ray, Wang, Wolfinger, Reich, *Epidemics* 50:100810 (arXiv:2407.19054) |
| `[IP]` | InfluPaint — Lemaitre & Lessler (arXiv:2604.24913) |
| `[GC]` | GenCast — Price et al., *Nature* 637:84-90 (arXiv:2312.15796) |
| `[FGN]` | FGN — Alet et al. (arXiv:2506.10772) |
| `[CSDI]` | CSDI — Tashiro et al., NeurIPS 2021 (arXiv:2107.03502) |
| `[RF]` | Flow matching / rectified flow — Lipman et al. (2210.02747), Liu et al. (2209.03003) |
| `[HUB]` | cdcepi/FluSight-forecast-hub spec + hubverse |
| `[EVAL]` | Bracher et al. 2021 (WIS), Cramer et al. PNAS 2022 (pairwise rWIS), scoringutils |
| `[NEW]` | Original to this plan — not taken from any of the above |
| `[REJ]` | Something a source did that this plan deliberately does **not** do |

Where a source's own ablation contradicts its headline method, this plan follows the ablation.
That happens three times (`[FLU]` reporting adjustments, `[FLU]` ensembling, `[IP]` augmentation)
and each is called out below.

---

## 1. Decision summary

### 1.1 The scheduling conclusion that reorders the design doc

The design doc's recommended plan (§8) is: data layer → B → C → A → GBQR → ensemble. Its own
complexity estimates are B at 2-3 weeks and A at 4-8 weeks. With 9 weeks to the hub opening and the
data layer consuming the first two, **A cannot be on the critical path for the season opener**.

Revised ordering, and the single most consequential change this plan makes to the design doc:

1. **Weeks 1-2 — Layer 0 (vintage data) and Layer 1 (evaluation harness).** Both are prerequisites
   for every model and neither is model-specific. The harness is built *before* the first model, not
   after, so that no model is ever evaluated by ad-hoc code.
2. **Week 3 — the floor and the free baseline.** GBQR `[FLU]` reproduction and foundation-model
   zero-shot `[C]`. Both are cheap and both bound expectations for B.
3. **Weeks 3-6 — B, then B's ablations.** The ablations are the deliverable, not the architecture.
   Both reference papers found their gains in ablation (`[FLU]` §ablation table; `[IP]` levers
   table), and B's minutes-scale backtest is the only thing that makes 50+ ablations affordable.
4. **Week 7 — freeze the submission stack.** Whatever has cleared the gates.
5. **Weeks 8-9 — operations.** Dry runs, hub onboarding, monitoring, buffer.
6. **Weeks 5 onward, in parallel, off the critical path — A.** Gate for a mid-season swap in
   mid-December. A is the scientific instrument and the publication; it is not the November
   submission.

### 1.2 What gets submitted in November

The quantile average of whatever clears gate G3 (§14), from: B (seed-ensembled), GBQR `[FLU]`, C.
If nothing clears, GBQR alone. This is a deliberate floor: `[FLU]` is a published, reproducible,
top-of-leaderboard method, and shipping it is never a failure state.

Note the tension the design doc already flags: `[FLU]`'s own ablation found ensembling contributed
almost nothing (0.625 → 0.622). The ensemble is therefore included on the evidence of the
*retrospective backtest*, not on the general prior that ensembles help `[REJ: FLU's ensembling
emphasis]`.

---

## 2. Provenance ledger

The complete list of what is taken from where, what is refused, and why. This is the table the
design doc implies but does not write down.

### 2.1 Adopted

| Element | From | Exactly what is taken | Why |
|---|---|---|---|
| Joint training across signals | `[FLU]` | One model, all signals, signal identity as a feature; long-history ILINet/FluSurv windows train the same weights with NHSN masked | Largest single lever in the Flusion ablation: GBQR-only-NHSN 0.857 vs GBQR 0.625 |
| Joint training across locations | `[FLU]` | One parameter set for all 53 locations, location as an embedding | Second-largest lever: by-location 0.780 vs 0.625 |
| Rate per 100k → power transform → per-(location, channel) centre/scale | `[FLU]` | Whole preprocessing chain; 95th-percentile scaling | Removes the scale heterogeneity that otherwise defeats joint training |
| Target = change from last observed value | `[FLU]` | Output parameterization | Makes the model's job a delta, not a level; `[FLU]`'s GBQR-no-level variant confirms level features are not load-bearing |
| Season week + weeks-from-Christmas features | `[FLU]` | Both, as statics | Both in `[FLU]`'s top five feature importances |
| Season exclusions 2008/09, 2009/10, 2020/21, 2021/22 | `[FLU]` | Verbatim | Pandemic H1N1 and suppressed-flu seasons |
| Power transform | `[FLU]` + `[IP]` | Exponent as a hyperparameter over {0.25, 0.5} | `[FLU]` fourth root, `[IP]` sqrt; both found transform > no transform (`[FLU]` 0.642 → 0.625; `[IP]` sqrt > linear) |
| Simulated trajectories as training data | `[IP]` | Scenario Modeling Hub rounds mixed into the corpus with covariate channels masked | `[IP]`'s largest lever (30/70 surveillance/simulation best) |
| Signal dropout as the *only* augmentation | `[IP]` | Channel- and block-level dropout | `[IP]` found every other augmentation harmful; this one is not one they tested and is required for operational robustness |
| Mask-native inputs | `[CSDI]` | Values never fed without their mask bit; masks are inputs, not exceptions | Makes "covariate absent" ordinary at train and inference time |
| Conditioning by concatenation at *training* time | `[CSDI]` | Context/target split sampled per example; loss on targets only | `[IP]`'s own conclusion: "training directly for the forecast task may improve performance" |
| Conditioning by concatenating past states | `[GC]` | Past window concatenated to the model input rather than injected by cross-attention | Simplest thing that worked at scale |
| Autoregressive rollout for long horizons | `[GC]` | Block-at-a-time rollout for peak targets from B | The only way B reaches end-of-season |
| Shared low-dimensional noise vector | `[FGN]` | z ~ N(0, I_32), one draw per trajectory, shared across all locations, horizons and pathogens, injected through conditional layer norm | Produces jointly coherent samples from a marginal loss — exactly the hub's 100-sample requirement |
| CRPS training objective | `[FGN]` | Almost-fair CRPS on marginals | WIS is discretized CRPS `[EVAL]`; this trains the scored metric |
| Rectified flow instead of DDPM | `[RF]` | Linear interpolant, velocity prediction, ODE sampling in 20-40 steps | Removes `[IP]`'s documented T=200-vs-500 sensitivity and cuts sampling cost by ~an order of magnitude |
| Vintage-aware training | `[NEW]` | Every example assembled as-of a reference date, targets finalized | `[FLU]` names this as an unclosed limitation; nothing in the literature does it for this target |
| Cross-location attention | `[NEW]`, motivated by `[FLU]` | Self-attention over 53 location tokens | `[FLU]` states verbatim that predictions were not informed by contemporaneous observations in other locations |
| Sample-native outputs | `[HUB]` | One draw satisfies primary, ED, sample, rate-change, and peak targets | Hub requires 100 temporally-connected samples, not resampled from quantiles |
| Pairwise-tournament relative WIS | `[EVAL]` | Cramer et al. procedure, scoringutils implementation | The metric the hub reports |
| Revision sensitivity analysis | `[FLU]` | Exclude location-date pairs revised by ≥10 admissions, report both | Their own robustness check; ours will matter more since we model revisions |

### 2.2 Refused

| Refused | From | Why |
|---|---|---|
| Reporting/backfill adjustments as preprocessing | `[FLU]` `[REJ]` | Their ablation found them counterproductive (no-reporting-adj 0.600 beat GBQR 0.625). Replaced by vintage-aware training, which learns the revision process instead of correcting for it by hand |
| Ensembling as a headline contributor | `[FLU]` `[REJ]` | Their ablation: 0.625 → 0.622. Kept, but demoted to "verify empirically", never assumed |
| Unconditional training + inpainting-only conditioning | `[IP]` `[REJ]` | `[IP]` measured training loss to be positively but *non-orderingly* related to forecast skill. Their stated future work is conditional training. We do that |
| Poisson resampling, temporal padding, intensity scaling | `[IP]` `[REJ]` | All three degraded performance in their ablation |
| Elaborate inpainting schedules (RePaint/CoPaint variants) | `[IP]` `[REJ]` | Their ablation: negligible effect. Not worth engineering time |
| U-Net variant search | `[IP]` `[REJ]` | Their ablation: negligible |
| DDPM with T=500 | `[IP]` `[REJ]` | 20-40 min for 512 trajectories. Rectified flow gets equivalent quality at 20-40 ODE steps |
| Graph-transformer encoder-processor-decoder | `[GC]` `[REJ]` | Designed for ~1M mesh nodes at 0.25°. We have 53 locations. Plain self-attention is sufficient and 3 orders of magnitude cheaper |
| Diffusion head as the primary forecaster | `[GC]` `[REJ]` | `[FGN]` — the same group, same backbone — beat it with a CRPS-trained marginal head at a fraction of the cost. That is the direct evidence for B before A |
| Large-scale fine-tuning of a foundation model | `[C]` `[REJ]` | ~30 seasons of data against 100M+ parameters. LoRA with hard early stopping only |
| History imputation for missing signals | Meyer et al., Benefield et al. `[REJ]` | Both imputed history rather than training jointly; both ranked below `[FLU]`. We train jointly |

### 2.3 Original to this plan

Marked `[NEW]` throughout. The substantive ones, so they can be defended or dropped as a set:

1. **The as-of leakage invariant as an executable test** (§4.4). Vintage correctness asserted by
   unit test, not by discipline.
2. **As-of-fitted transform constants** (§4.6). `[FLU]` scales by the 95th percentile; if that
   percentile is computed over the full history it leaks the future into every training example.
3. **Target maturity policy** (§4.7). "Finalized" is undefined near the end of the corpus; this is
   made explicit and enforced.
4. **The revision atlas as a week-1 deliverable** (§4.11). Measures how much horizon -1/0 values
   actually move before a line of model code is written. If revisions are small, the nowcasting
   claim is weak and we should know that immediately.
5. **Resolution of the NB-layer double-counting problem** (§6.6). The design doc specifies both a
   CRPS loss against observations and an NB observation layer. Trained against observations, the
   head already contains observation noise; stacking NB on top over-disperses. Resolved below.
6. **Loss-scale question made explicit** (§6.3). CRPS on the transformed scale is not CRPS on the
   count scale, and WIS is scored on counts.
7. **Seed ensembling as the primary variance reducer** (§6.4). At 0.3M parameters and ~900 joint
   training examples, seed variance is likely to exceed most architectural effects.
8. **Hierarchical coherence handling for the US location** (§6.5).
9. **Gate structure with named fallbacks** (§14).

---

## 3. Repository and module layout

Building on what exists (`src/influpaintx/data`, `scripts/pull_covariates.py`, `tests/`). Raw
acquisition is done; everything below `vintage/` is new.

```
src/influpaintx/
  data/            # EXISTS. Raw acquisition, snapshots, checksums, native geography.
  vintage/
    schema.py      # canonical long-format record + validators
    store.py       # parquet/DuckDB writer, issue-partitioned
    materialize.py # as_of(date) -> wide arrays; THE leakage boundary
    geography.py   # location harmonization to the hub's 53 + FluSurv sites
    calendar.py    # MMWR epiweeks, season indexing, reference-date arithmetic
  tensors/
    transforms.py  # rate, power, as-of centre/scale; forward and inverse
    dataset.py     # window/canvas example construction, masks, vintage age
    dropout.py     # signal dropout augmentation
    corpora.py     # long-history windows, SMH simulation pool, mixing
  models/
    common/        # residual MLP, attention block, FiLM, embeddings, EMA
    b_crps/        # windowed CRPS forecaster
    a_flow/        # flow-matching block/canvas model
    c_foundation/  # foundation-model adapters
    gbqr/          # Flusion-style LightGBM quantile regression floor
  losses/
    crps.py        # fair / almost-fair CRPS
    flow.py        # rectified-flow velocity loss
  sampling/
    draw.py        # ensemble draw, rollout, transform inversion
    calibrate.py   # post-hoc quantile-width correction, NB fit
  eval/
    wis.py         # WIS, coverage, interval score components
    tournament.py  # pairwise relative WIS
    backtest.py    # the driver: reference dates -> model outputs -> scores
    diagnostics.py # spread-skill, PIT/rank histograms, joint-coherence checks
  submit/
    hubwrite.py    # hubverse model-output writer (quantile + sample types)
    validate.py    # local validation before PR
configs/           # YAML; one file per experiment, inherits from base
scripts/
tests/
  test_leakage.py  # the invariant in §4.4
  test_roundtrip.py
  test_hub_format.py
  test_scoring.py  # harness vs. published hub scores
```

Config: plain dataclasses + YAML overrides. No Hydra `[NEW]` — the experiment count is ~50-100, all
one-dimensional sweeps off a base config, and Hydra's composition machinery costs more than it
returns at that scale. Experiment tracking: a flat `runs/` directory of JSON result records plus a
DuckDB view over them; sufficient for 100 runs and no service dependency during a season.

---

## 4. Layer 0 — the vintage data store

The design doc's §2 in implementable form. This is the largest block of work and the one that
every model depends on. Two weeks.

### 4.1 Canonical schema

One long-format table. Every source normalizes into it; nothing downstream knows about source
idiosyncrasies.

```
observations
  source        str      # 'nhsn', 'nssp', 'nwss', 'ilinet', 'iliplus', 'nrevss', 'flusurv'
  signal        str      # 'admissions_flu', 'ed_prop_covid', 'wastewater_iav', ...
  geo_type      str      # 'state', 'nation', 'hhs_region', 'flusurv_site'
  geo_value     str      # canonical code within geo_type
  time_type     str      # 'epiweek'
  time_value    int      # YYYYWW, MMWR
  issue         int      # YYYYWW of the vintage in which this value was current
  value         float    # raw, untransformed, in the source's native units
  unit          str      # 'count', 'rate_per_100k', 'proportion', 'copies_per_ml_norm'
  extra         struct   # source-specific: pct_hospitals_reporting, sample_size, ...
```

Primary key `(source, signal, geo_type, geo_value, time_value, issue)`. Append-only. A value that
does not change between issues is *not* re-stored — store only issue-diffs, and materialize by
as-of query `[NEW]`. This keeps the store ~2 orders of magnitude smaller than a full issue cross
product and makes revision analysis a direct query rather than a diff job.

### 4.2 Sources and their vintage mechanism

| Source | Signals | Vintage mechanism | Depth | Risk |
|---|---|---|---|---|
| NHSN weekly hospital respiratory | admissions flu / COVID / RSV, pct hospitals reporting | Hub `target-data` git history + CDC dataset snapshots (already captured) | 2022/23– | Reporting mandate has changed more than once; completeness varies by jurisdiction |
| NSSP ED visits | proportion of ED visits, flu / COVID / RSV | Delphi V5 `snapshot_date` / `report_time` | ~2022– | ~3 usable seasons |
| NWSS wastewater | influenza A, SARS-CoV-2, RSV | Delphi V5 `snapshot_date` | ~2023– | ~2-3 usable seasons, site composition changes |
| ILINet | weighted ILI, unweighted ILI | Deferred until V5 acquisition is added | 1997– | State-level from 2010; national/HHS earlier |
| ILI+ | ILI × positivity | Derived; vintage = max of components' issues | 1997– | Derived signals need their own vintage rule (rule below) |
| NREVSS | positivity by pathogen | Direct CDC NREVSS snapshots/posted vintages | 1997– | Lab panel definitions changed ~2015 |
| FluSurv-NET | hospitalization rate | Deferred until V5 acquisition is added | 2010– | Site-based geography, not states (§4.4) |

Derived-signal vintage rule `[NEW]`: a derived value's issue is the **maximum** issue of its inputs
(it did not exist until its last input arrived), and its value at issue *i* is computed from inputs
as-of *i*. Naive derivation from finalized components is a leak and is caught by the §4.4 test.

### 4.3 The as-of materializer

One function, and it is the only place in the codebase permitted to read the `issue` column.

```python
def as_of(ref: EpiWeek, signals: list[SignalSpec], geo: GeoSet) -> VintageFrame:
    """Everything that was knowable at reference week `ref`, and nothing else.

    Returns value, mask, and vintage-age arrays. A cell is masked 0 when no issue <= ref
    carries a value for it. `age` is (ref - first_issue_carrying_this_cell) in weeks;
    undefined (and never read) where mask == 0.
    """
```

Contract, enforced by test:

- No row with `issue > ref` may influence the output, at any depth, including through derived
  signals, transform constants, imputations, or feature engineering.
- The output carries `(value, mask, age)` for every requested cell. `age` is the revision-maturity
  feature; it is what lets the model learn that a 1-week-old NHSN value is provisional and a
  20-week-old one is not `[NEW]`, and it is the mechanism by which horizons -1 and 0 become a
  learned backfill model rather than a preprocessing correction `[REJ: FLU reporting adjustments]`.

### 4.4 The leakage invariant (executable) `[NEW]`

```python
def test_no_future_issues():
    for ref in sample_reference_weeks(n=200):
        with issue_tripwire(max_issue=ref):   # store raises on any read of issue > ref
            frame = as_of(ref, ALL_SIGNALS, HUB53)
        assert frame.max_touched_issue <= ref
```

`issue_tripwire` is a store-level context manager that raises on any query whose predicate could
admit `issue > ref`. This is a stronger test than comparing outputs, because it catches leaks
through code paths that happen not to change the numbers on the sampled weeks. Run in CI on every
commit. **Nothing merges that fails this test.**

Rationale for the emphasis: vintage-awareness is the design's single novel claim against `[FLU]`,
and a silent leak would produce an excellent backtest and a bad season. The failure is
undetectable by inspection of results.

### 4.5 Geography and calendar

- Canonical location set: the hub's 53 (50 states, DC, PR, US) via `auxiliary-data/locations.csv`
  for FIPS, abbreviation, and population `[HUB]`.
- FluSurv-NET sites are **not** states and must not be mapped to them. They enter as their own
  `geo_type` with their own tokens, exactly as `[FLU]` treated FluSurv as a separate source with
  its own locations. Attempting a site→state mapping invents data.
- HHS regions: used only for early ILINet seasons where state-level ILI does not exist. Those
  windows train the shared weights with state channels masked — the same mechanism as any other
  missing covariate `[CSDI]`.
- Calendar: MMWR epiweeks throughout. Season *s* spans EW31 of year *s* to EW30 of *s+1*;
  `season_week ∈ [1, 52|53]`. 53-week seasons are handled by dropping EW53 from the canvas for
  model A `[NEW]` and by leaving the week in place for model B (which has no fixed canvas — one of
  B's structural advantages, design doc §6).
- Reference date arithmetic `[HUB]`: reference_date is the Saturday following the Wednesday
  deadline; `target_end_date = reference_date + 7·horizon`; horizon -1 is submitted but not scored.
  All of this lives in `calendar.py` and nowhere else.

### 4.6 Transforms, fitted as-of `[NEW]`

Chain, per `[FLU]`:

```
count → rate per 100k (population from locations.csv, fixed vintage)
      → power transform x^p, p ∈ {0.25 [FLU], 0.5 [IP]}   (hyperparameter)
      → (x - c_{l,ch}) / s_{l,ch}                          (per location × channel)
```

with `s_{l,ch}` the 95th percentile of the transformed series `[FLU]` and `c` its median.

The `[NEW]` part: **`c` and `s` are computed from data available as of the reference date**, not
from the full history. Otherwise every training example is scaled by a constant that encodes the
season's eventual peak, which is a direct leak of the target. Implementation: cache
`(c, s)` per `(location, channel, ref_year)` and recompute annually rather than weekly, both for
cost and to avoid a scale that jitters week to week. For the first ~2 seasons of a new signal,
where the percentile is unstable, fall back to a pooled cross-location estimate.

Inverse transform must be exact and tested round-trip (`test_roundtrip.py`), including the
clipping behaviour at zero, which is where the power transform is least well behaved.

### 4.7 Target definition and maturity `[NEW]`

The design doc says targets are "finalized values". Near the end of the corpus, no such value
exists. Made explicit:

- `y*_{l,t}` = the value of the signal at time *t* in the **latest snapshot available at corpus
  build time**.
- `maturity(t) = latest_snapshot_week − t`. An example's target cell is included in the loss only
  if `maturity ≥ M_min`, default `M_min = 8` weeks.
- Cells with `maturity < M_min` are masked out of the loss, not imputed. They are the most recent
  weeks, which are also the most informative about the current season — so `M_min` is a real
  bias/recency trade-off and is an ablation knob, not a constant.
- The revision atlas (§4.11) determines whether `M_min = 8` is too small (NHSN still moving at 8
  weeks) or wastefully large.

### 4.8 Extra training corpora

**Long-history multitask windows `[FLU]`.** ILINet from 1997, FluSurv-NET from 2010. Each window is
an ordinary training example with the NHSN channels masked. No special code path — this is the
payoff of mask-native inputs `[CSDI]`. Reproduces `[FLU]`'s largest lever.

**Simulation pool `[IP]`.** Flu Scenario Modeling Hub rounds, plus COVID and RSV hubs. Each
trajectory is a full season with only the hospitalization channel unmasked, so for B it yields ~35
windows per trajectory and for A one canvas.

Mixing weight `w_sim` is a hyperparameter. `[IP]` found 70% simulation optimal for a full-season
unconditional generative model. **Expect a much lower optimum for B** `[NEW]`: simulations carry no
covariate structure, and B's entire thesis is the contemporaneous covariate relationship. A
simulation-heavy mix trains B mostly on examples where every covariate is masked. Sweep
`w_sim ∈ {0, 0.1, 0.2, 0.3, 0.5, 0.7}`.

**Exclusions `[FLU]`.** 2008/09, 2009/10, 2020/21, 2021/22.

**Corpus size, stated plainly `[NEW]`.** ~24 usable ILI-era seasons × ~35 reference weeks ≈ **840
joint examples** (each example is a 53-location joint state), of which only ~4 seasons carry NHSN.
This number, not taste, is why the parameter counts in the design doc are 0.3M and 1M rather than
30M, and why §6.4's regularization and seed ensembling are not optional.

### 4.9 Augmentation policy

Exactly one augmentation `[IP]`:

- **Channel dropout**: zero an entire channel across all locations and weeks, `p = 0.3`.
- **Block dropout**: zero a random `(location, channel)` block of random length, `p = 0.15`.
- Both set `mask = 0`, so the model sees them as ordinary missingness, identical in form to a real
  outage `[CSDI]`.

Nothing else. `[IP]` tested Poisson resampling, temporal padding, and intensity scaling and all
three hurt `[REJ]`.

Purpose is operational, not statistical: NSSP and NWSS have ~3 usable seasons and can disappear
mid-season. The inference-time stress test (forced removal of NWSS and NSSP, §12 ablation 5) is the
acceptance criterion, and it is a *requirement*, not an experiment — a model that degrades
catastrophically when NWSS vanishes cannot be submitted.

### 4.10 Acceptance criteria for Layer 0

- `test_leakage.py` green on 200 sampled reference weeks, all sources.
- `test_roundtrip.py`: transform ∘ inverse = identity to 1e-9 on the full corpus, including zeros.
- Materializing any single reference date for all 53 locations and all channels: **< 2 seconds**
  warm. This is what makes a 3-season backtest (~105 reference dates × N models) feasible in
  minutes rather than hours, and it is therefore a hard requirement, not a nicety.
- A spot check against a hub `target-data` snapshot at 5 historical dates, byte-level on the NHSN
  admissions column.

### 4.11 Week-1 deliverable: the revision atlas `[NEW]`

Before any model is written, produce `reports/revision-atlas.md` answering:

1. For NHSN flu admissions, the distribution of `(final − provisional)` at vintage age 1, 2, 4, 8,
   13, 26 weeks, by location and by season.
2. What fraction of location-weeks are revised by ≥10 admissions — the exact quantity `[FLU]` used
   for its sensitivity analysis, so our numbers are comparable to theirs.
3. Whether revision magnitude correlates with `pct_hospitals_reporting`. If yes, that feature is
   load-bearing and must never be dropped; if no, it is decoration.
4. The same for NSSP and NWSS.

**Why this comes first:** the nowcast/backfill head is one of the three claims that distinguish this
work from `[FLU]`. If NHSN horizon -1 values move by a median of 0.5 admissions, the claim is
cosmetic and the effort should move to cross-location structure instead. Two days of analysis
protects several weeks of modelling. This analysis does not exist in either reference paper.

---

## 5. Layer 1 — the evaluation harness

Built **before** the first model `[NEW]`. The design doc places evaluation in §8.1 after the
models; that ordering reliably produces models evaluated by throwaway scripts.

### 5.1 Backtest driver

```python
def backtest(model_fn, seasons, *, n_samples=2000, out_dir) -> ScoreFrame:
    for ref_date in reference_dates(seasons):          # Saturdays, ~35/season
        frame  = as_of(ref_date - 1wk, ALL_SIGNALS, HUB53)   # Wednesday knowledge
        draws  = model_fn(frame, ref_date, n=n_samples)
        write_hub_output(draws, ref_date, out_dir)     # quantile + sample types
    return score(out_dir, truth=finalized_targets(seasons))
```

Critical detail `[NEW]`: the as-of week is the **Wednesday before** the reference Saturday, because
that is when data is released and forecasts are made `[HUB]`. Using the reference Saturday itself
grants the model three extra days of reporting it will not have in production. This is the single
easiest way to produce an optimistic backtest.

### 5.2 Metrics

- **WIS** `[EVAL]`, 23 quantile levels, on integer counts, decomposed into dispersion,
  over-prediction, and under-prediction. The decomposition is what tells us whether a calibration
  problem is width or bias.
- **Relative WIS by pairwise tournament** `[EVAL]` (Cramer et al.), the hub's headline metric.
- **MAE** of the predictive median.
- **Coverage** at 50% and 95%, plus one-sided quantile coverage differentials across all 23 levels.
  Non-negotiable given `[IP]`'s realized 14.8% / 52.6% against nominal 50/90.
- **Sensitivity analysis** `[FLU]`: all of the above recomputed excluding location-date pairs whose
  latest available value was subsequently revised by ≥10 admissions.

Implementation: Python, validated against `scoringutils` `[EVAL]` on a fixed fixture, with the
comparison itself a test (`test_scoring.py`). A pure-Python scorer keeps the backtest in one
process; the R cross-check keeps it honest.

### 5.3 Protocol and splits

**Leave-one-season-out (LOSO) for model selection** over 2023-24, 2024-25, 2025-26, with training on
all prior seasons plus the other held-out seasons' *earlier* data only — i.e. no future season ever
informs a held-out season, and within a season the backtest is strictly walk-forward.

Hyperparameters are chosen by mean LOSO relative WIS. Reported numbers are LOSO numbers. The final
submission model is refit on all seasons with the LOSO-selected configuration and **no further
selection** `[NEW]`. The temptation to peek at 2025-26 (the most recent and most representative
season) is exactly the temptation LOSO exists to remove.

Seed handling: every configuration is run with 5 seeds and reported as mean ± SE across seeds. An
ablation whose effect is smaller than the seed SE is reported as null, not as a small effect
`[NEW]`. At ~840 training examples this will disqualify a meaningful number of otherwise
interesting-looking results, which is the point.

### 5.4 Comparators

| Comparator | Source | Role |
|---|---|---|
| FluSight-baseline | hub | Absolute floor; also the harness validation target |
| Trend baseline | `[NEW]` | Sanity: last value + linear extrapolation with empirical quantiles |
| GBQR, Flusion-style | reichlab/flusion, reimplemented on **our** covariate table | The real floor (§9) |
| Flusion, as published | reichlab/flusion | Reference point |
| InfluPaint, published weights | ACCIDDA/Influpaint | Reference point for A |
| FluSight-ensemble | hub | The thing to beat |

### 5.5 Harness acceptance criterion

Recompute WIS for FluSight-baseline over 2024-25 from hub model output and hub target data, and
match the hub's published scores to within floating-point tolerance. Until this passes, no model
result is trustworthy. This is gate G1.

---

## 6. Model B — windowed stochastic forecaster trained on CRPS

The primary target model. Design doc §4, specified.

### 6.1 Example construction

For reference date *d*, one example is the full 53-location joint state:

```
W  : [L=53, P=8,  C=15, 3]   window: (value, mask, vintage_age) per week per channel
S  : [L=53, F=~12]           statics
Y  : [L=53, H=8,  C_out=6]   targets, transformed scale
YM : [L=53, H=8,  C_out=6]   target mask (missing, immature, or dropped)
```

- **P = 8** past weeks, ablated over {4, 8, 12}.
- **H = 8** horizons, `h ∈ {−2, −1, 0, 1, 2, 3, 4, 5}`. The hub needs −1..3 `[HUB]`; the extra
  negative horizons give the backfill head more supervision `[NEW]` and the extra positive ones
  reduce the number of rollout steps needed for peak targets.
- **C_out = 6**: hospitalizations × {flu, COVID, RSV} and ED proportion × {flu, COVID, RSV}. The
  non-flu channels are multitask supervision, not submission targets `[FLU: joint training]`.
- **Statics F** `[FLU]`: season week (sin/cos), weeks-from-Christmas, weeks since season onset,
  cumulative-to-date per channel, log population, location embedding index, source-regime
  indicators. The design doc calls these load-bearing and it is right: an 8-week window cannot
  distinguish pre-peak from post-peak.

### 6.2 Architecture

```
per-location encoder        (shared weights, applied over L)
  flatten(W_l) ++ S_l                       -> [L, 8*15*3 + 12] = [L, 372]
  Linear -> 128, then 3 × ResBlock(128)      (LayerNorm, GELU, dropout 0.1)
                                             -> h ∈ [L, 128]

cross-location block                                             [NEW; motivated by FLU's gap]
  h += location_embedding + log_pop_proj
  2 × TransformerEncoderLayer(d=128, heads=4, ff=256, pre-norm)
                                             -> g ∈ [L, 128]

stochastic decoder                                               [FGN]
  z ~ N(0, I_32)                             ONE draw per trajectory, shared across L, H, C_out
  for each of 2 decoder blocks:
      γ, β = Linear(z)                       # FiLM on the layer-norm output
      u = γ ⊙ LN(u) + β; u = u + MLP(u)
  Linear -> H * C_out
                                             -> Δ ∈ [L, 8, 6]

output                                                            [FLU]
  ŷ_{l,h,c} = last_observed_transformed_{l,c} + Δ_{l,h,c}
```

~0.3M parameters. No attention over time — at P=8 a flattened MLP is sufficient and cheaper, and
`[GC]`'s conditioning-by-concatenation is the precedent for feeding past states directly rather
than attending over them.

**The single most important architectural detail** `[FGN]`: `z` is drawn **once per trajectory** and
shared across all locations, horizons, pathogens, and output channels. This is what makes samples
jointly coherent even though the loss only sees marginals, and it is what satisfies the hub's
requirement that the 100 samples be temporally connected rather than resampled from quantiles
`[HUB]`. Drawing z per location would produce correct marginals and incoherent trajectories, and
would fail the hub's sample target while looking fine on WIS.

Optional, ablated: geographic adjacency as an additive attention bias `[NEW]`. Cheap; may help at
53 tokens or may be noise.

### 6.3 Loss

Almost-fair CRPS over M draws of z `[FGN]`:

```
CRPS_α(x_{1..M}, y) = (1/M) Σ_i |x_i − y|
                    − (α / (2 M (M−1))) Σ_{i≠j} |x_i − x_j|
```

with `α = 0.95`. The mechanism that matters: the **negative** spread term is what rewards the model
for producing spread. The naive (biased) estimator with `1/(2M²)` systematically penalizes spread
and is the standard route to z-collapse. `α` slightly below 1 keeps gradients well-behaved without
materially breaking propriety. Default `M = 8`, ablated over {4, 8, 16}.

Aggregation:

```
L = Σ_{l,h,c} w_l · w_h · w_c · YM_{l,h,c} · CRPS_α(·) / Σ YM
```

- `w_l ∝ pop_l^β`, `β ∈ {0, 0.5, 1}` `[NEW]`. **Why this matters**: WIS is scored on counts and is
  dominated by California, Texas, New York, Florida. CRPS on the per-location-standardized scale
  weights Wyoming equally with California. Per-location standardization is what makes joint
  training work `[FLU]` and must stay; the correction belongs in the loss weights, not the
  transform.
- `w_h`: flat by default. Only horizons −1..3 are submitted, and only 0..3 are scored `[HUB]`, so a
  variant down-weighting h ∈ {4, 5} is worth one ablation.
- `w_c`: flu channels 1.0, COVID/RSV channels ~0.3, since they are regularizers rather than targets.

**Loss scale — an open decision** `[NEW]`. `CRPS` on the transformed scale ≠ `CRPS` on the count
scale, and the scored metric is on counts. Three options, implemented behind `loss_scale`:

| Option | Behaviour |
|---|---|
| `transformed` | Stable, well-conditioned, mismatched to WIS |
| `count` | Matches WIS exactly, but gradients are dominated by a handful of large states and large weeks |
| `both` | Transformed loss plus a small-weight count-scale term |

Default `transformed` with `β = 1` population weighting, on the argument that population weighting
recovers most of the count-scale emphasis without the conditioning problem. This is an argument,
not a result; it is ablation 6 and the answer should replace this paragraph.

### 6.4 Optimization and regularization

| Knob | Default | Note |
|---|---|---|
| Optimizer | AdamW, lr 3e-4, wd 1e-4 | |
| Schedule | 500-step warmup, cosine decay | |
| Batch | 32 reference dates (each = 53 locations) | |
| Grad clip | 1.0 | |
| Dropout | 0.1 | |
| EMA | 0.999, EMA weights used for evaluation | |
| Epochs | LOSO-determined, expect 200-400 passes | ~840 examples: an epoch is seconds |
| Seeds | 5 per configuration | §5.3 |

**Seed ensembling `[NEW]`.** The production model is a pool of 5-10 independently seeded B models,
combined by **pooling their samples** (not by averaging quantiles), each contributing 100/K
trajectories. At 0.3M parameters this costs minutes and is likely to be worth more than most of the
architectural ablations. Sample pooling preserves joint coherence within each trajectory, which
quantile averaging would destroy — this is why the combination rule is pooling and not averaging
`[HUB]`.

### 6.5 Sampling and hub outputs

```
draw N_q = 2000 z             -> quantiles for the primary and ED targets
subsample 100 of those draws  -> the sample target, temporally connected by construction [HUB]
```

`N_q = 2000` rather than 100 `[NEW]`: the hub requires the 0.01 and 0.99 quantiles, and with 100
samples those are the two extreme order statistics — pure noise. 2000 draws cost one extra forward
pass of a 0.3M-parameter model.

Post-processing chain:

1. Invert transform to rates, multiply by population, get counts.
2. Round to integers `[HUB]`. Rounding is monotone, so quantile monotonicity survives; assert it
   anyway.
3. Clip at 0.
4. **US location** `[NEW]`: `us_mode ∈ {predicted, summed, blended}`. NHSN national is approximately
   the sum of jurisdictions, so predicting US directly can be incoherent with the state forecasts.
   Diagnose first (sum-of-state-samples vs. predicted-US distribution), then choose. `summed` is
   free and exactly coherent; `predicted` may be better calibrated. Default `blended` pending the
   diagnostic.
5. **Rate-change categories, peak week, peak intensity**: computed from the same draws. Thresholds
   are read from the season's `tasks.json` `[HUB]`, never hardcoded — the population-dependent
   category boundaries have changed between seasons.
6. Puerto Rico `[HUB]`: forecast always, include in training with weight 0.5, flag in monitoring.
   It has been excluded from some official evaluations for data instability; forecasting it costs
   nothing and omitting it forfeits a submission requirement.

### 6.6 The negative-binomial layer — resolving a contradiction `[NEW]`

The design doc specifies both (a) CRPS trained against observed values and (b) an NB observation
layer converting latent rates to counts. **These double-count observation noise.** A CRPS-trained
head fit against *observed* data already learns the total predictive spread, observation noise
included; multiplying by an NB layer on top over-disperses.

Resolution:

- **For B**: train CRPS directly against observed (matured) counts on the transformed scale. The
  head's spread *is* the predictive spread. The NB layer is **not** applied by default.
- The genuine problem NB was introduced to solve — `[IP]`'s overconfidence in small states — is
  addressed instead by a **post-hoc per-stratum quantile-width correction** `[NEW]`: fit
  multiplicative width factors `κ(h, pop_stratum)` on held-out seasons by minimizing WIS, with 3
  population strata and 8 horizons = 24 parameters, heavily regularized toward 1.
- NB remains implemented and is ablation 8. If the width correction's fitted `κ` is near 1
  everywhere, calibration is fine and neither is needed.
- **For A**, where the flow model produces a latent field, NB is the natural count layer and stays.

### 6.7 z-collapse diagnostics — a required gate, not an optional check

The design doc names z-collapse as B's principal risk and is right to: `[FGN]` obtained coherent
joints emergently from 40 years of ERA5; we have ~840 joint examples. Every B run emits:

1. **Degeneracy**: `Var_z[ŷ]` at fixed input. If it approaches 0, z is ignored and the model is a
   point forecaster wearing a costume.
2. **Spread-skill ratio** per horizon: `sqrt((M+1)/M) · mean(ensemble sd) / RMSE(ensemble mean)`.
   Target ≈ 1. Below 1 = overconfident (`[IP]`'s failure mode), above 1 = over-dispersed.
3. **Rank histograms / PIT** of observations within the sample ensemble, per horizon and per
   population stratum. U-shaped = overconfident.
4. **Joint checks** `[NEW]`, since the loss does not constrain these and therefore nothing else will
   catch them:
   - correlation matrix of sample residuals across locations vs. the empirical residual correlation;
   - PIT of multi-week *differences* (h=3 minus h=1), which is where trajectory incoherence shows;
   - sum-of-state-samples vs. predicted-US distribution.

If collapse appears: increase M, inject z at more depths, verify the fair estimator's sign, and only
then consider a spread-encouraging auxiliary term — which breaks propriety and is a last resort.

### 6.8 Peak targets by rollout

Autoregressive block rollout `[GC]`: predict H weeks, append to the window, redraw z, repeat ~30
steps to end of season. Peak week and peak intensity are read off each trajectory.

Honest expectation, per the design doc: error compounds and nothing in the CRPS loss constrains
trajectory shape at that range. Since CDC ranks peak targets third and fourth among optional
secondary targets `[HUB]`, this costs little competitively. Rollout is implemented, evaluated, and
submitted only if it beats a climatological peak-week distribution `[NEW]`. If it does not, the
peak targets wait for A.

---

## 7. Model C — foundation-model baseline

One day of work, per the design doc. Its value is as a floor and as an ensemble member with
uncorrelated errors, not as a contender.

- Candidates: Chronos, TimesFM, Moirai families. **Verify interfaces before committing** — these
  change every few months, and the design doc's claim that all three accept covariates is
  optimistic: the Chronos family is univariate, and covariate handling there is external (e.g. a
  covariate regressor on residuals). TimesFM and Moirai have first-party covariate paths. Confirm
  against current releases, do not trust this paragraph.
- Input: per-(location, signal) transformed rate series; covariates via whichever interface exists.
- Zero-shot first, then LoRA with hard early stopping on LOSO. 100M+ parameters against ~840 joint
  examples: over-fitting is the default outcome, and the zero-shot prior is the asset being risked.
- Known structural limits, from the design doc: no vintage concept (provisional values taken at
  face value — precisely the real-time failure mode), quantile grids that typically omit 0.01/0.99
  so WIS tails come from extrapolation, and no coherent samples, so **C cannot satisfy the
  100-sample target** and enters the ensemble for the quantile targets only `[HUB]`.

---

## 8. Model A — full-season masked conditional generative model

Off the critical path. Built on B's backbone, staged in three steps so each has a working
intermediate `[NEW]` (the design doc's §8.4 proposes the same staging; this fixes the details).

### 8.1 Stage 1 — flow-matching head on B's backbone, H-week block

Replace the FiLM decoder with a rectified-flow decoder over the same `[L, H, C_out]` block `[RF]`:

```
x1 = target block (transformed);  x0 ~ N(0, I)
xt = (1−t) x0 + t x1,             t ~ LogitNormal(0, 1)      [RF; SD3-style time sampling]
loss = || v_θ(xt, t, cond) − (x1 − x0) ||²
sample: Euler or midpoint ODE, 20-40 steps
```

`cond` = B's cross-location output `g`, unchanged. This makes ablation 1 (§12) a **controlled**
comparison: identical backbone, identical data, identical splits, only the head differs. That is
the cleanest available way to settle the diffusion-vs-CRPS question empirically rather than by
argument, and it is why A is built on B's backbone rather than as an independent model.

Flow matching rather than DDPM `[RF]` `[REJ: IP's DDPM]`: `[IP]` documented T=500 > T=200 and spent
20-40 minutes per 512 trajectories. Rectified flow's ODE sampling in 20-40 steps removes that
sensitivity and most of that cost.

### 8.2 Stage 2 — full-season canvas

`X ∈ R^{C × L × T}`, T = 52, L = 53 `[IP]`.

- Tokens: (location, week) cells; channels in the feature dimension, concatenated with mask bits and
  vintage age `[CSDI]`.
- Network: factorized attention alternating time-within-location and location-within-week; d_model
  96, 6 blocks, ~1M parameters. Not a U-Net `[REJ: IP's U-Net]` — `[IP]`'s own ablation found the
  U-Net variant made negligible difference, so the architecture choice is free and factorized
  attention composes with the mask machinery more naturally.
- Static features: sinusoidal flow-time embedding, season week, log population, location embedding.
- **Training-time conditioning** `[CSDI]`, the central change from `[IP]`: sample a season, a
  reference week *d*, and a vintage; context = cells observed as of *d*; targets = all remaining
  cells, **including the provisional last weeks whose targets are the finalized values**. Loss on
  target cells only. That last clause is what turns backfill into a learned operation.
- Sampling: 100+ trajectories via `vmap`, then the NB count layer (§6.6 — here it belongs).

### 8.3 Stage 3 — arbitrary-mask evaluation

Reconstruction under masks not seen at training (whole missing states, missing weeks, missing
signals), which `[IP]` demonstrated qualitatively and which the theoretical inpainting work
(RePaint, CoPaint, Rout et al.) motivates. This is a paper figure, not a submission requirement.

### 8.4 What A must clear to be submitted

- Beat B on LOSO relative WIS for the primary target, **or**
- Beat the climatological peak-week distribution by a margin that justifies the secondary-target
  submissions, **and**
- 50%/95% coverage within ±10 points of nominal on held-out seasons — the explicit response to
  `[IP]`'s 14.8%/52.6%. Calibration is A's gating risk and must be verified, not assumed.

---

## 9. The GBQR floor

A `[FLU]`-style LightGBM quantile regression, **fed our covariate table**, not theirs `[NEW]`.

This is the crucial experimental-design point: comparing our neural models to published Flusion
confounds the model with the data. Feeding GBQR the identical vintage-aware, multi-signal,
cross-location covariate table isolates the architecture. If B cannot beat GBQR on the same table,
the neural machinery is not earning its place — and the design doc says as much ("suspect the
implementation, not the idea"), which is only actionable if the comparison is controlled.

Configuration from `[FLU]`: quantile objective, ~100 bags, feature set built from the same
transformed/scaled channels plus season week and weeks-from-Christmas. Two variants, GBQR and
GBQR-no-level, per their ablation.

GBQR also **hedges the whole project**: it is fast, robust, published, and top-of-leaderboard. If
the schedule slips, it ships.

---

## 10. Ensembling

Quantile average of the survivors, with equal weights unless LOSO says otherwise `[FLU]` `[EVAL]`.

Caveat carried forward from the ledger: `[FLU]`'s own ablation found ensembling worth 0.003 rWIS
over its best component. The ensemble is included because two structurally different good models
usually outrank either, **and it is verified on LOSO before it is submitted**, never assumed.

Sample targets cannot be quantile-averaged and keep coherence `[HUB]`. For the 100-sample
submission, take samples from the single best sample-native model (B or A), not from the ensemble.

---

## 11. Weekly operations

Wednesday, data released midday; submission due Wednesday; reference date is the following Saturday
`[HUB]`.

```
14:00 ET  pull all sources           scripts/pull_covariates.py (exists)
14:20     integrity checks           checksums, row-count deltas, staleness alarms
14:30     materialize as-of frame    fails loudly rather than silently dropping a source
14:40     forecast: B-ensemble, GBQR, C
14:50     combine, post-process, write hubverse output
15:00     validate                   local hubValidations + our own format tests
15:10     diagnostics                spread-skill, PIT, coherence, vs. last week's forecast
15:20     human review               a one-page HTML report, then PR to the hub
```

Automated except the final PR `[NEW]`. Submitting to a public hub is outward-facing and irreversible
in practice; a human looks at the plot every week.

Alarms that block submission: any source stale > 2 weeks without a mask being set; any forecast
whose median moves > 3× the historical week-over-week change; any quantile crossing; any location
missing.

---

## 12. Ablation program

Ordered by expected effect, following the design doc §8.2, with cost and decision rule added
`[NEW]`. Each entry is 5 seeds × 3 LOSO folds.

| # | Ablation | Source of the hypothesis | Cost | Decision rule |
|---|---|---|---|---|
| 1 | B's CRPS head vs. A's flow head, same backbone | `[FGN]` vs `[GC]`/`[IP]` | 1 day + GPU | Settles the head question; whichever wins is the primary |
| 2 | Long-history multitask windows on/off | `[FLU]` largest lever | 2 h | If off ≈ on, our corpus story is wrong and needs diagnosis before anything else |
| 3 | Vintage-aware vs. finalized-data training | `[NEW]`; `[FLU]`'s stated gap | 3 h | The project's novelty claim. If null, say so in the paper |
| 4 | Simulation mixing weight, 6 values | `[IP]` largest lever | 6 h | Expect a much lower optimum than `[IP]`'s 70% for B (§4.8) |
| 5 | Signal dropout rate; **forced NWSS/NSSP removal at inference** | `[IP]`-adjacent; operational | 4 h | The removal test is a **requirement**: catastrophic degradation blocks submission |
| 6 | Power exponent {0.25, 0.5}; population weighting β {0, 0.5, 1}; loss scale | `[FLU]`/`[IP]`; `[NEW]` | 6 h | Replaces the §6.3 argument with a result |
| 7 | Cross-location attention on/off; adjacency bias | `[NEW]`; `[FLU]`'s gap | 2 h | Directly tests the second novelty claim |
| 8 | NB layer on/off; post-hoc width correction on/off | `[IP]` calibration failure | 3 h | §6.6. Expect the width correction to win on B |
| 9 | P ∈ {4, 8, 12}; M ∈ {4, 8, 16}; seed-ensemble size | `[NEW]` | 4 h | Routine |
| 10 | `us_mode` ∈ {predicted, summed, blended} | `[NEW]` | 1 h | Coherence vs. calibration |

Reporting discipline `[NEW]`: any effect smaller than the across-seed SE is reported as null. At
~840 examples this will disqualify several plausible-looking results, and reporting them as real
would be the most likely way this project produces a finding that does not replicate.

---

## 13. Risk register

| Risk | Trigger to watch | Mitigation | Fallback |
|---|---|---|---|
| Vintage leakage | `test_leakage.py` | The tripwire test in CI (§4.4) | — the test is the mitigation |
| z-collapse `[FGN]` risk at our data scale | `Var_z[ŷ] → 0`; flat rank histograms | §6.7 diagnostics on every run; M ≥ 8; multi-depth injection | Fall back to a parametric quantile head; lose the sample target, keep the primary |
| Weak joints despite good marginals | Multi-week difference PIT; cross-location correlation error | Same diagnostics | Submit quantiles from B and samples from A |
| A's calibration `[IP]`: 14.8%/52.6% | Coverage on LOSO | Conditional training + NB + width correction | A does not ship; B does |
| NSSP/NWSS have ~3 seasons | Ablation 5 removal test | Signal dropout, weight decay | Model is required to survive their removal |
| Thin COVID/RSV simulation pools | Multitask channels degrade flu | `w_c` down-weighting; drop the channel | Flu-only training |
| NHSN reporting completeness / mandate changes | `pct_hospitals_reporting` drift | The feature itself is the defence; revision atlas quantifies it | — |
| Puerto Rico instability | Per-location WIS outliers | Train weight 0.5, forecast always | — |
| Schedule slip past 24 Oct | Gate G3 | Ship GBQR + C | GBQR alone |
| Foundation-model API drift | Import fails | Pin versions; verify interfaces before committing (§7) | Drop C; it is the least load-bearing component |

---

## 14. Schedule and gates

Nine weeks, 2026-09-07 to 2026-11-08.

| Week | Dates | Deliverable |
|---|---|---|
| 1 | Sep 7-13 | Vintage store, as-of materializer, leakage tripwire green, **revision atlas** (§4.11) |
| 2 | Sep 14-20 | Transforms (as-of), Dataset/window construction, corpora + mixing, **evaluation harness validated against hub scores** |
| 3 | Sep 21-27 | GBQR floor on our table; C zero-shot; B v0 end-to-end on one reference date |
| 4 | Sep 28-Oct 4 | B v1; first full LOSO backtest; diagnostics wired in |
| 5 | Oct 5-11 | Ablations 2-5. **A stage 1 begins in parallel** |
| 6 | Oct 12-18 | Ablations 6-10; calibration repair; seed ensembling |
| 7 | Oct 19-25 | Ensemble selection on LOSO; **freeze**; submission pipeline built |
| 8 | Oct 26-Nov 1 | Dry-run submissions on historical dates; hub onboarding; monitoring report |
| 9 | Nov 2-8 | Buffer; first live submission when the hub opens |
| 10+ | Nov-Dec | A stages 2-3; mid-December swap gate |

### Gates

- **G1 — Sep 20.** Leakage suite green; revision atlas delivered; harness reproduces published
  FluSight-baseline WIS. *Fail → cut scope to a GBQR-only submission path and spend the remaining
  time on the data layer, which is the reusable asset.*
- **G2 — Oct 4.** B produces hub-valid output end-to-end for a historical reference date, all
  output types. *Fail → B is descoped to quantile targets only.*
- **G3 — Oct 18.** B (seed-ensembled) beats the GBQR floor on LOSO relative WIS by ≥3%, with the
  margin exceeding the across-seed SE. *Fail → submit GBQR + C; continue B as research.*
- **G4 — Oct 25.** Stack frozen. Bugfixes only after this date. No architecture changes during the
  season.
- **G5 — Dec 15.** A clears §8.4. *Pass → mid-season swap for secondary targets. Fail → A is the
  paper, not the submission.*

---

## 15. Open questions to settle before writing code

Ordered by how much rework the wrong answer causes.

1. **Does the revision signal justify the nowcast head?** Answered by the revision atlas (§4.11) in
   week 1. If NHSN horizon −1 revisions are negligible, ablation 3 will be null and effort should
   move to cross-location structure.
2. **What exactly counts as "finalized"?** `M_min = 8` weeks (§4.7) is a guess until the atlas
   exists.
3. **`loss_scale` and population weighting** (§6.3). The default is an argument, not a result.
4. **`us_mode`** (§6.5). Needs the coherence diagnostic first.
5. **Hub specifics to verify against the 2026-27 `tasks.json`, not assumed from 2025-26** `[HUB]`:
   the compound task-ID set defining what a sample is jointly coherent over; the rate-change
   category thresholds; whether the location set is still 53; whether horizon −1 remains unscored.
6. **Foundation-model interfaces** (§7). Verify before committing a day to C.
7. **FluSurv-NET geography** (§4.5). Confirm that treating sites as their own geo_type matches how
   `[FLU]` used them, since we are reproducing their largest lever and a mismatch here would
   silently weaken it.

---

## 16. Appendix — default configuration

```yaml
data:
  power_exponent: 0.25          # [FLU]; ablate 0.5 [IP]
  scale_stat: p95               # [FLU]
  scale_fit: as_of              # [NEW] — refit annually, never full-history
  season_start_week: 31
  exclude_seasons: [2008/09, 2009/10, 2020/21, 2021/22]   # [FLU]
  target_maturity_min_weeks: 8  # [NEW]
  channel_dropout_p: 0.30       # [IP]-adjacent
  block_dropout_p: 0.15
  sim_mix_weight: 0.20          # [IP] found 0.70 for A; expect lower for B

model_b:
  window_P: 8
  horizons: [-2, -1, 0, 1, 2, 3, 4, 5]
  c_out: 6                      # {hosp, ed_prop} x {flu, covid, rsv}
  d_model: 128
  encoder_blocks: 3
  cross_location_layers: 2      # [NEW], closes [FLU]'s stated gap
  cross_location_heads: 4
  z_dim: 32                     # [FGN]
  z_shared_across: [location, horizon, channel]   # [FGN] — the coherence mechanism
  film_blocks: 2
  dropout: 0.1
  params_approx: 300_000

loss:
  type: almost_fair_crps        # [FGN]
  alpha: 0.95
  M: 8
  scale: transformed            # open question, §6.3
  pop_weight_beta: 1.0
  channel_weights: {flu: 1.0, covid: 0.3, rsv: 0.3}

optim:
  lr: 3.0e-4
  weight_decay: 1.0e-4
  warmup_steps: 500
  batch_reference_dates: 32
  grad_clip: 1.0
  ema_decay: 0.999
  seeds: 5

sampling:
  n_quantile_draws: 2000        # [NEW] — 100 is too few for the 0.01/0.99 levels
  n_submitted_samples: 100      # [HUB]
  nb_layer: false               # [NEW], §6.6 — width correction instead
  width_correction: true
  us_mode: blended

model_a:
  head: rectified_flow          # [RF], replaces [IP]'s DDPM
  time_sampling: logit_normal
  ode_steps: 32
  canvas_T: 52
  d_model: 96
  blocks: 6
  attention: factorized_time_location
  conditioning: train_time_concat   # [CSDI], replaces [IP]'s sampling-time inpainting
  nb_layer: true
```

### Tensor shapes, for reference

| Object | Shape | Where |
|---|---|---|
| B window | `[53, 8, 15, 3]` | §6.1 |
| B statics | `[53, ~12]` | §6.1 |
| B latent | `[53, 128]` | §6.2 |
| B noise | `[32]` — one per trajectory | §6.2 |
| B output | `[53, 8, 6]` | §6.2 |
| A canvas | `[15, 53, 52]` | §8.2 |
| Hub quantile output | 53 × 5 × 23 rows per reference date | `[HUB]` |
| Hub sample output | 53 × 5 × 100 rows per reference date | `[HUB]` |
