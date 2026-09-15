# Tapestry — Architecture

Research and implementation plan · 14 September 2026 · Proposed work, not an implemented or evaluated model.

**Acquisition update, 11 September 2026:** Delphi pulls now use only V5 NHSN,
NSSP, NWSS, and inpatient/outpatient claims. The dated
local audit below records previously downloaded data; its ILINet/FluView/FluSurv
entries are no longer active acquisition sources. Transfer experiments that need
those feeds remain conditional on a future V5 addition. See the
[current catalog](../data/sources.md).

**Design recommendation:** build a small conditional sample generator with a shared forecasting function across surveillance signals, explicit availability and revision information, and one spatial attention block. Train its influenza hospitalization predictions with fair CRPS in count units. Select it using actual FluSight WIS on forecasts reconstructed from historical information states. Start without simulations or an extra observation-noise layer; add complexity through controlled experiments.

The design and numerical defaults below are **proposals**. Paper findings are attributed separately; weather results are not evidence that this model will beat an epidemic forecasting baseline.

## My staged plan

The work proceeds from a small, finalized-data pilot to the full vintage-aware, multi-channel architecture. The six-channel set is NHSN admissions and NSSP ED proportions for influenza, COVID-19, and RSV.

| Stage | Data | Purpose |
|---|---|---|
| **B0** | Non-vintaged data; six channels (NHSN + NSSP) | Establish the baseline and decide most of the architecture. |
| **B1** | Vintaged data; six channels (NHSN + NSSP) | Add release-time information and measure the value of revision-aware training. |
| **B2** | Non-vintaged data; more channels (NHSN + NSSP + wastewater) | Test whether wastewater adds signal before introducing vintage complexity. |
| **B3** | Vintaged data; more channels (NHSN + NSSP + wastewater) | Combine the selected architecture with the broader vintage-aware input panel. |

B0 is the decision point for the model architecture. B1–B3 extend the same design while isolating the effects of vintages and additional channels.

## 1. Objective and scientific questions

The primary deliverable is a predictive distribution for weekly laboratory-confirmed influenza admissions at each required jurisdiction, at horizons 0–3. An internal hindcast at horizon −1 learns revision correction. Longer direct horizons support development diagnostics. ED visits, COVID-19, RSV, and full-season peak predictions are staged extensions.

The central hypothesis is that a compact model can improve short-horizon WIS by combining four ingredients: learning recurring dynamics from older surveillance systems; conditioning on contemporaneous signals and locations; reproducing what was available at each issuance; and directly training a distribution with a proper score. Each ingredient must survive an ablation.

Specific questions:

1. Does historical ILI/ILI+/FluSurv forecasting improve NHSN predictions after controlling for the data supplied to the baseline?
2. Does cross-signal conditioning add value beyond sharing training examples across signals?
3. Does training on provisional inputs improve forecasts against later evaluation truth?
4. Can functional noise yield useful temporal and spatial dependence with relatively few epidemic seasons?
5. Does a generative neural model improve on the hub ensemble and on simple same-data baselines?

There is no defensible prospective rank prediction. The model remains a research candidate if it fails the comparison in question 5.

## 2. Concepts and the forecast contract

### 3.1 A conditional sample generator

Instead of outputting a mean and a standard deviation, the network takes the available history and a random vector, then returns a possible future. Repeating the forward pass with different random vectors produces a predictive distribution:

```text
Y^(m) = G_theta(context_as_of_cutoff, requested_targets, z^(m))
z^(m) ~ independent Normal(0, I_32)
```

This is an **implicit distribution**: sampling is straightforward, while a closed-form probability density is generally unavailable. Samples can still be trained and evaluated with CRPS.

One sample is a complete array of requested locations, horizons, and outcomes. Its identity must survive aggregation, export, and any later rollout. Sampling each cell independently would define a different joint distribution.

### 3.2 Marginals, dependence, and coherence

A marginal asks, “How many admissions could New York have next week?” A joint asks, “Which New York, New Jersey, and subsequent-week outcomes occur together?”

Two models can produce identical quantiles for every state and week while disagreeing completely about whether waves rise together. Their marginal CRPS and WIS would be identical. Shared noise provides a useful architectural constraint, but cannot make the marginal objective identify the correct dependence. Assess joint behavior with aggregates, changes, and explicit multivariate scores.

Separate three meanings of coherence: preserved member identity; empirically realistic dependence; and exact aggregation identities. The first is an implementation property. The second needs evidence. The third needs a verified geographic/accounting definition and explicit enforcement.

### 3.3 Revisions and nowcasts

An **event week** is when admissions occurred. A **release time** is when a particular estimate became available. A **vintage** is the value of the same event at a particular release. A recent reported value can be input while its later revision is the supervised outcome.

For example, an issuance may see a preliminary count of 80 for the prior week. A later target vintage may contain 110. Training that pair teaches revision correction. Replacing the input 80 with 110 would remove precisely the uncertainty faced in operation.

### 3.4 Dates, horizons, and outputs

Use an explicit cutoff timestamp `d` in UTC, derived from the local submission deadline, and an epidemiological reference Saturday `r`. Never use event date alone to decide availability. CDC epiweeks run Sunday–Saturday and require proper handling of 53-week years.

The currently published [FluSight hub specification](https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/main/README.md) is still labeled **2025–2026** when checked on 5 September 2026. It specifies horizons 0–3, optional −1, integer admission quantiles, and 23 quantile levels. Samples are optional; if supplied, exactly 100 temporally connected samples per task are required for horizons 0–3. Reference date is the Saturday following the Wednesday deadline. Use this as the provisional contract, and pin the new season's configuration when published.

Proposed internal horizon list: **`[-1, 0, 1, 2, 3, 4, 5, 6]`**, eight outputs, with `target_end_date = r + 7h days`. This resolves the ambiguity in “H=8.” The last four horizons are developmental, not established submission requirements.

The geographic request list is explicit: 50 states, DC, PR, and US where required and supportable. The explorer currently displays 51 state/DC entries; that is not evidence that the PR and US output paths have been implemented.

## 4. Training data: what exists and what must be built

### 4.1 Local evidence

The existing code implements acquisition, immutable raw snapshots, geographic metadata, and an explorer. It has no training materializer or model package. Reuse `RawDataRepository` and source manifests; build model data from raw records. The explorer stores a latest-vintage view and is unsuitable as a historical training database.

The following is an inventory observation from the local 4 September snapshots, audited on 5 September. Date envelopes describe indexed events across each dataset, not continuous coverage of every pathogen, state, or historical release. The machine-readable audit is `references/local-data-coverage-2026-09-05.json`.

| Data family / local dataset keys | Observed event-history envelope | Role and constraints |
|---|---|---|
| NHSN: `delphi_nhsn`, `cdc_nhsn_final`, `hub_flusight_current` | Delphi and direct CDC envelopes: 2020-08-08–2026-08-29; current FluSight indexed envelope: 2022-02-05–2026-07-04 | Primary labels and inputs; audit influenza separately from COVID/RSV, reporting regime changes, and first reliable date. Use archived releases for context and a declared evaluation-vintage policy for labels. |
| `cdc_nhsn_initial_release` | 2024-11-09–2026-08-29 | Frozen first-release product; valuable paired-revision evidence, but not a full sequence of all revisions. |
| `delphi_fluview` | 1997-09-29–2026-08-17 in explorer | Long-history ILI auxiliary forecasting. The inspected 1997 event is retained in a 2013 release: event history is longer than real-time vintage history. |
| `delphi_fluview_clinical` | 2016-10-03–2026-08-17 in explorer | Influenza positivity and construction of ILI+ where aligned support and releases exist. Older ILI+ is not automatically available from this feed. |
| `delphi_flusurv` | Raw all-age non-null records span epiweeks 200340–202631; retained release dates span 2012-11-02–2026-08-14 | Historical catchment/network rate task. 236,348 non-null all-age rows include repeated revisions; they are not independent examples. This raw source is not represented in the state explorer audit. Verify incident versus cumulative rate semantics before use. |
| NSSP: `delphi_nssp`, `cdc_nssp_daily`, `cdc_nssp_trajectories` | Weekly Delphi envelope: 2022-10-01–2026-08-29; daily CDC begins 2022-09-25 | Candidate leading covariate and later ED target. Check smoothing, denominators, publication lag, and historical report coverage. |
| NREVSS: comprehensive plus COVID/RSV vintage feeds | Comprehensive begins 2019-07-06; COVID vintage events 2020-03-14; RSV 2020-04-11; through 2026-08-29 | Virological covariates; several products have HHS/national support rather than native state measurements. |
| NWSS: direct pathogen feeds, `cdc_nwss_wval`, `delphi_nwss` | Direct raw envelopes begin 2020-01-14 (COVID), 2021-09-15 (flu), 2022-02-27 (RSV); through 2026-09-01 | Candidate covariates after an as-of site aggregation audit. Delphi sewershed archives are present, but their usable historical release coverage was not exhaustively measured here. |
| COVID/RSV forecast hubs and RSV-NET | Current hub envelopes begin 2022-10-01; RSV-NET indexed envelope begins 2014-10-11 | Secondary pathogen tasks and evaluation archives. Filter truth records from model forecasts and preserve each target's definition. |
| Scenario Modeling Hub simulations | **No dedicated simulation adapter in the 25-dataset catalog** | Additional acquisition and provenance work. Forecast-hub archives are not automatically SMH training trajectories. |
| Weather, mobility, general-population vaccination, school schedules | **No dedicated measured feeds in the current catalog** | Do not make them prerequisites. Calendar features are immediately derivable; add external feeds only with a defined vintage protocol and validation experiment. |

### 4.2 Dataset tiers

**Tier 1 — historical dynamics:** ILI, ILI+ where constructible, FluSurv rates, and reliable NHSN history. Each source is supervised in its own units/support. This is the transfer corpus, not a procedure for inventing missing historical hospital counts.

**Tier 2 — aligned real-time panels:** NHSN and same-cutoff covariates, initially NSSP and virology. An example may have many channels missing. Source absence is represented as part of the input.

**Tier 3 — optional synthetic trajectories:** add only after a strong real-data result. Accept joint simulation draws with simulator version, calibration-data cutoff, scenario, geography, pathogen, units, round date, and member ID. Distinguish latent incidence, hospital prevalence, and incident admissions. If only marginal quantiles are available, do not fabricate a joint trajectory by independently sampling them.

**Tier 4 — secondary pathogens:** introduce COVID/RSV tasks after the influenza pipeline is working. Their seasonality and reporting histories differ. Share the forecasting function with pathogen/source embeddings, and retain an influenza-only comparator for negative transfer.

### 4.3 Signal definitions and geographic support

The eventual context registry can include 15 signal families: three admission series, three ED proportions, three wastewater summaries, ILI, ILI+, three pathogen-positivity series, and FluSurv rate. Quality fields, ages, masks, and calendar variables are additional features. This is a registry target, not a claim that all 15 form a complete historical panel.

ILI+ is the product of aligned ILI fraction and influenza positivity fraction. A 2% ILI proportion times 10% positivity is 0.002 as a fraction, or 0.2% of visits. Store units explicitly. State ILI multiplied by regional positivity is a mixed-support derived feature; preserve that metadata. Require both components to be available at the issuance cutoff.

Use native HHS/national observations as parent context, with a support-type indicator. They must not become 51 independent state supervision targets. FluSurv catchments remain catchment tasks; do not divide or expand them using statewide population. Treat changing catchment membership as metadata and exclude invalid comparisons.

For wastewater, aggregate compatible site metrics only: harmonize assay/units, select site records by release time, and carry site count and population-coverage diagnostics. Population-weighted summaries require valid, nonduplicated catchment weights; otherwise use a prespecified robust site summary and label its support. Do not use the explorer's unweighted state mean as a validated training feature. Avoid revised WVAL normalization in historical contexts without a recoverable vintage.

For ED proportions, prefer an official weekly target. If aggregating daily data, use summed numerators divided by summed denominators. A simple mean of daily percentages is a different statistic. Overlapping smoothed daily percentages are not independent weekly observations.

## 5. As-of materialization and leakage controls

### 5.1 Normalized record contract

```text
dataset_key, signal_id, pathogen, geo_id, geo_type, spatial_support
event_start, event_end, release_time, retrieved_at
value, unit, numerator, denominator, population
quality_fields, fill_method, source_record_key
snapshot_id, source_commit, raw_file_sha256
```

Deduplicate multiple representations of the same publisher observation rather than treating direct CDC, Delphi, and hub copies as independent signals. Keep a provenance crosswalk and resolve disagreements visibly. Population metadata must be versioned or pinned to the relevant hub convention.

At cutoff `d`, select the latest eligible release for each natural observation key. A recent download can supply an older vintage only when the publisher archive establishes that older release. A snapshot-only product downloaded in September 2026 cannot be used as though that exact version existed in January 2024.

Track both event age and release age. “Weeks since first report” is known only when the first report is actually recoverable; left-censored archives require an unknown-age indicator. Missing release times must not silently be set equal to event dates. Source-specific conservative lag rules can support a separately labeled sensitivity analysis, not proof of real-time availability.

### 5.2 Three distinct masks

1. **Availability mask `m`:** whether an eligible observed value exists at cutoff. Zero is a valid observed value when `m=1`.
2. **Artificial conditioning-dropout mask `k`:** which eligible values are hidden during training to mimic outages. Effective input mask is `m*k`.
3. **Target mask `q`:** which outcomes have valid supervised labels of the right definition and vintage. `q` is not the complement of the input mask; provisional inputs and finalized labels may coexist for the same event.

Also store a geography-padding mask and categorical missingness reasons: not yet released, structurally unsupported, suppressed, and simulated dropout. Never calculate loss on padded cells or nonexistent historical NHSN outcomes.

### 5.3 Labels versus the information used to train a historical model

For forecast issuance `d`, every fitted weight, scaler, simulation calibration, and calibrator must use information available by `d`. For an older training origin `d_i`, its input comes from releases no later than `d_i`, while its supervised label can use a later release **only if that label was already available by the model-fit cutoff**.

Keep three timestamps separate: example-origin cutoff, model-fit cutoff, and evaluation-truth cutoff. Evaluation may legitimately use revisions arriving after issuance. Those revisions may not influence the model that purportedly issued that forecast.

Use the official hub evaluation truth policy if specified. Additionally evaluate at fixed maturities, initially 4 and 8 weeks after the target event, and on the frozen latest snapshot. Call these “evaluation vintages,” since “final” need not mean permanently immutable. Estimate revision stabilization only using the development/training data.

### 5.4 Gates before any model comparison

The materializer must emit a coverage matrix by source, native geography, season, event date, and available release date. It must pass these checks:

- Appending a future release cannot change an already materialized earlier context, its features, or its scaler.
- Derived covariates use the latest common eligible information boundary; ILI+ cannot borrow a future positivity revision.
- Scalers, source-selection rules, and seasonal summaries use permitted data only. No full-season maximum or retrospectively known peak/onset feature enters an early-season example.
- Training and validation labels do not cross a held-out target boundary. Purge using each example's explicit target-date set; a fixed lookback gap alone is insufficient.
- Network/catchment/HHS broadcasts do not multiply the effective number of target observations.
- Missing, suppressed, invalid, and zero observations remain distinguishable.
- Target files are separated from model-output files in hub archives; no other model's future forecast is accidentally used as truth.

The audit is a prerequisite, not an assertion that every downloaded “versioned” source already passes.

## 6. Preprocessing and the actual supervised task

### 6.1 Input transforms

For count sources with the appropriate population, define rate `r = 100000*y/population`. For rates and proportions, preserve their native denominator. Candidate power transforms are fourth root and square root. Wastewater uses a compatible `log1p`-style or source-specific transformation; do not apply population conversion to an already normalized concentration.

For nonnegative rate-like sources, a concrete starting transform is:

```text
v = r^alpha                     alpha = 1/4 initially; compare 1/2
s = max(training-only Q95(v), source-specific positive floor)
u = v/s - training-only mean(v/s)
```

Estimate statistics within the fitting fold. Sparse/new locations shrink to a source-level scale. Keep the original rate and count inverse transformations. A model can learn on transformed inputs while being scored on original count outputs.

For bounded proportions, use an explicit bounded inverse such as logistic with a small, recorded boundary treatment; verify near-zero predictions. Do not clip targets into invented positive values without preserving their original values for evaluation.

### 6.2 Focal-signal transfer: making the old data contribute

The original six-output sketch does not specify how an ILI-only window produces any loss. Masking all six hospitalization/ED outputs would create an example with no learning signal. Resolve this with a **shared source-query decoder**:

```text
training query = (native geography, focal signal, horizon)
input = focal signal's observed history + eligible contextual signals + metadata
output = a sample of that focal signal's future/revised value
```

Use a common history encoder and common decoder across focal signals, with small source/pathogen embeddings and unit-specific inverse transforms. An old ILI episode predicts future ILI. A FluSurv episode predicts the native catchment rate. A current NHSN episode predicts admissions. Shared weights receive gradients from all three tasks.

The focal history occupies a **generic slot**, so the same learned dynamic mapping sees rising, falling, and turning trajectories across sources. Additional channels occupy typed context slots. Source embeddings allow differences without assigning each source a completely separate model. A source-specific inverse transform prevents unit confusion.

Construct training episodes at one cutoff and geographic support level. A state episode can contain state queries plus regional context; a catchment episode contains catchment queries. Heterogeneous support tokens are a later architecture option. Regional/network labels receive one supervised contribution per actual observation.

Run two independent switches: historical source-query examples on/off, and contemporary contextual signals on/off. This distinguishes transfer of dynamics from use of leading covariates.

### 6.3 Residual targets and summaries

The decoder predicts change from the last eligible transformed focal observation, including a feature for its staleness. This is a parameterization convenience, not an assumption that the latest observation is final. A horizon −1 query may revise that anchor.

If no focal history exists, use a learned source/calendar prior with an explicit missing-anchor indicator. Train some examples in this mode and separately report performance; its existence does not establish reliable zero-history forecasting.

Features beyond the default 12-week window include calendar sine/cosine, weeks relative to Christmas, log population where meaningful, source/geography identifiers, recent slopes, valid-observation counts, and causal season-to-date summaries with coverage. An onset feature must be generated by a fixed online rule and can be unknown. Include a longer trailing summary for off-season pathogens; do not force every pathogen to follow a flu-season reset.

## 7. Architecture

All candidates use the same records, split, output queries, transforms, loss, and evaluation. The following sizes are **engineering budgets**, not measured parameter counts or runtime claims.

| Candidate | Encoder and spatial treatment | Why test it | Proposed budget / order |
|---|---|---|---|
| **Minimal shared MLP** | Flatten each 12-week focal/context panel; two residual MLP blocks; shared decoder; no information exchange between locations | Establish whether the transfer task and stochastic loss work with minimal structure | 0.15–0.4M parameters; build first |
| **MLP + spatial attention — recommended** | MLP encoder plus one 4-head attention block over native location tokens; shared conditional decoder | Allows contemporaneous cross-location information while remaining small | 0.3–0.8M; main candidate |
| **Temporal mixer + spatial attention** | Replace flattening with 2–4 temporal/feature mixing blocks; same spatial block and decoder | Better temporal parameter sharing when expanding the lookback to 26 or 52 weeks | 0.4–1.2M; test if longer context helps |
| **Typed source/geography tokens** | Shared per-series encoder; attention over typed `(source, native geography)` tokens and target queries | Handles irregular source coverage and native catchments without fixed broadcasts | 0.5–1.5M; defer until support handling limits spatial attention |

The MLP plus spatial-attention design is the initial candidate. Do not search all widths, depths, masks, and data mixtures at once.

### 7.1 B1 data flow and tensor contract

```text
Raw snapshots + release histories
            |
As-of materializer --> values, masks, ages, support, quality, calendar
            |
Shared focal/context encoder (12 weeks per native location)
            |
One spatial attention block (same issuance, padding masked)
            |
Target queries: source + horizon + anchor + geographic metadata
            |
Shared stochastic decoder <--- z[32], reused across locations/horizons
            |
Source-specific inverse transforms
            |
Full sample array [member, location, horizon, target]
            |
Count-scale CRPS during fitting; WIS/coverage/joint diagnostics in validation
```

For a minibatch of `N` episodes, use context `[N,L,P,C,F]`, with `P=12`, a registry-defined `C`, and explicit fields `F` for normalized value, availability, event/release ages and quality. Store categorical fields separately for embeddings if cleaner. Targets and their mask have `[N,L,H,S]`. Generated draws have `[M,N,L,H,S]`. `S` enumerates requested source tasks; unavailable tasks are masked, not filled with zeros.

To make the generic focal-history slot compatible with multiple requested sources, also construct focal histories `[N,L,S,P,F_focal]`. Apply one shared focal encoder to each source slice and a context encoder to the common panel. Fuse their embeddings, run the same spatial block separately for each focal-source panel, then decode the horizon queries. Cache these deterministic embeddings across all `M` draws. Thus the encoder weights are shared across sources, their actual histories remain distinct, and the same latent draw spans the final `S` outputs. This is source conditioning, not `S` independent models.

At state inference, `L` follows the requested geography registry rather than a hard-coded canvas. Static source/catchment training episodes may use smaller `L`, padded with a proper attention mask. No location learns from a future-time token. Attention within the observed history can be noncausal because every included observation is already available at the issuance boundary.

Starting dimensions: `d_model=128`, two residual local encoder blocks with hidden width 256, one spatial block with four heads and feed-forward width 256, and two shared decoder residual blocks. Include location/support embeddings but retain metadata-based fallbacks for unseen catchments.

### 7.2 Functional stochasticity

For decoder hidden state `h`, layer `k` uses:

```text
h_mod = (1 + gamma_k(z)) * LayerNorm(h) + beta_k(z)
gamma_k, beta_k = small learned linear maps or MLPs of a shared z embedding
```

The modulation maps are shared over locations and horizon/source queries. Hidden context differs across queries, so the same `z` can produce different local effects. Inject noise in both decoder blocks, not only into a final scalar output. Initialize modulation with small nonzero weights so the model can learn to use noise without destabilizing the initial forecast.

Compute the deterministic context encoder once, then expand the stochastic decoder over `M` draws. This amortizes expensive context processing and is a design simplification of full-network functional modulation. Compare injecting noise before the spatial block if decoder-only perturbation proves too restrictive.

Start with one global 32-vector. If regional idiosyncratic variation is insufficient, compare global + regional + local latent components with small dimensions and the same total training budget. This extension relaxes the original dependence bias and must improve joint diagnostics, not only interval width.

### 7.3 Output support and observation noise

Admissions predictions must be nonnegative. Use a smooth positive inverse parameterization around the residual anchor, and evaluate rounding only at export. Keep scoring and positivity transforms numerically stable near zero; large-state inverse fourth powers require monitoring and gradient clipping.

**Do not append independent negative-binomial noise by default.** A generator trained against observed admissions already learns variation in those observations. An extra noise layer can count that uncertainty twice and alter dependence.

If a later experiment explicitly represents an unobserved latent admission intensity, define a complete observation model such as `Y|lambda ~ NB(mean=lambda, dispersion=k)` and fit the predictive observation distribution consistently. Establish how gradients pass through or integrate out discrete observations. That is a separate model candidate, not a guaranteed fix for undercoverage.

## 8. Loss functions and optimization

### 8.1 Fair CRPS

For one valid target `y` and `M≥2` independent draws from the same fitted generator:

```text
fCRPS = (1/M) sum_m |x_m - y|
        - [1 / (2 M (M-1))] sum_{m != n} |x_m - x_n|
```

The first term rewards closeness to the observation. The second balances that with predictive spread; under the scoring rule, the best distribution in expectation is the true conditional marginal, subject to model capacity and estimation. It is not a reward for unlimited noise. The independence requirement is between sample members, not between locations within a member. See [Ferro](https://doi.org/10.1002/qj.2270).

Proposed initial `M=8`, then compare `M=2` and `M=16` at matched optimization budget. This is a gradient-variance/compute choice. Ordinary empirical CRPS uses `M²` in the denominator; it is correct for the empirical distribution itself, but estimates a different object from the fair underlying-distribution score.

Use raw count-space CRPS for primary admission outputs, after inverse preprocessing. A fourth-root-space score is not count-space CRPS, and multiplying it by population cannot repair the nonlinear difference. Auxiliary sources have normalized source-specific losses because their units differ.

### 8.2 Loss groups and weights

Compute each group as a valid-target masked average. Skip a missing group, log its absence, and never divide by zero. Do not simply average all observed cells together: the plentiful older ILI rows could dominate the hospitalization objective.

```text
L = L_flu_admissions_h0_to_h3_counts / A
    + 0.25 * L_flu_hminus1_counts / A
    + 0.25 * L_flu_h4_to_h6_counts / A
    + 0.20 * L_historical_focal_sources_standardized
    + 0.10 * L_other_pathogens_and_ED_standardized   # extension, initially 0
    + lambda_joint * L_joint                       # initially 0
```

`A` is one positive, training-fold-only scale constant for the entire primary target group; it stabilizes magnitudes without changing relative state weights. It is not a different denominator for every location. The decimal weights are starting proposals. Tune only a small auxiliary-weight grid, then freeze it before the final evaluation.

The primary average uses the location/horizon eligibility and weights prescribed by the evaluation contract. Also report state-equal normalized skill and US-only skill so count-scale gains are interpretable. Absolute WIS and tournament relative WIS are distinct summaries; a CRPS objective cannot exactly optimize the latter's model-pool-dependent ratios.

### 8.3 Joint-score experiment

First evaluate joint behavior without a joint loss. If it fails, compare an auxiliary score on prespecified small groups: a state's four-week vector, neighboring-state pairs, and regional sums where compatible truth exists.

Use a masked energy score and/or a variogram score on training-standardized variables. For example, variogram order `p=0.5` compares observed `|y_i−y_j|^p` to the predicted expectation of that difference, squared and averaged with fixed pair weights. It tests a different aspect than marginal WIS. See [Scheuerer & Hamill](https://doi.org/10.1175/MWR-D-14-00269.1) and [Pacchiardi et al.](https://jmlr.org/papers/v25/23-0038.html).

Use a larger consistent member count for validation. A squared Monte Carlo expectation in a variogram loss has finite-sample bias; if used for training, use an appropriate independent-pair/U-statistic estimate or document the plug-in approximation and its sensitivity to `M`. Do not describe an arbitrary finite-member approximation as exactly proper. Tune `lambda_joint` on development data only, with primary WIS as the main gate.

### 8.4 Training schedule and defaults

| Setting | Initial proposal | Planned check |
|---|---|---|
| Framework | PyTorch in a separate model extra/environment | Keep existing standard-library acquisition install lightweight; pin versions at implementation |
| Optimizer | AdamW, learning rate `3e-4`, weight decay `1e-3` | Small grid `{1e-4,3e-4}` × `{1e-4,1e-3}` only after correctness |
| Batch | 16 complete native-geography episodes | Adjust to measured device memory; do not call 53 correlated states 53 independent seasons |
| Draws | 8 per episode/query panel | Compare 2 and 16 |
| Regularization | Hidden dropout 0.1, channel outage probability 0.2, clipped gradient norm 1 | Validate forced outages and stochastic spread |
| Optimization budget | Pilot 10,000 updates; cap 30,000 until a measured learning curve justifies more | Evaluate every 500 updates on an inner chronological stop set; patience 8 evaluations |
| Precision | Float32 for inverse transforms and scores | Mixed precision only after checking overflow and tail stability |
| Seeds | One seed for screening; three for shortlisted models | Store seeds, RNG/device state, and deterministic settings |

Stage A optionally pretrains the shared source-query function on eligible historical signals, including genuine finalized-history examples explicitly tagged as such. Stage B trains aligned real-data contexts with primary hospitalization loss and continued auxiliary replay. Compare this with joint training from initialization; pretraining is not assumed better.

Initially sample about 60% primary NHSN episodes and 40% historical auxiliary episodes. Sample source, then season, then issuance uniformly within the source pool, with a cap on repeated versions of the same event. This avoids letting row count determine effective weights. Log realized draws and unique season/episode counts. Reduce auxiliary replay if it harms influenza validation.

Outage augmentation hides whole channels, recent release blocks, or regional groups, and recomputes all derived inputs that would disclose the hidden source. Include training with no auxiliary covariates. Use real missingness first; tune artificial probabilities from development outages. Keep a minimum valid context unless intentionally training the no-history fallback.

## 9. Synthetic data and unusual seasons

Use no synthetic data in the first benchmark. Once the real-data model is stable, compare synthetic fractions `{0, 0.1, 0.3, 0.7}` in the episode sampler, clearly distinguished from loss weights. Also test synthetic pretraining followed by real-only fine-tuning. Use one accepted architecture for this sweep.

A simulated realization and all overlapping windows from it belong to one partition. Group related trajectories by simulator, calibration run, parameter draw, and scenario round. Hold out a simulator or scenario family for synthetic diagnostics. Operational performance is always assessed on real surveillance observations.

For a historical backtest, simulations calibrated using a held-out season cannot train that season's model. Accept simulations genuinely available by the model-fit date, or regenerated with strictly eligible calibration data and explicit “reconstructed” provenance. Large synthetic volume is not a large number of independent observed epidemics.

Do not automatically discard 2008–10 or 2020–22 from the general respiratory corpus. Compare a prespecified conventional-flu training subset against inclusion with pandemic/reporting-regime indicators. For the primary ordinary-season task, retain the simpler exclusion policy if development WIS supports it. For COVID/RSV or off-season evaluation, assess those regimes explicitly rather than importing a flu-specific rule blindly.

## 10. Validation design

### 10.1 Chronological development and final evaluation

Use rolling issuance forecasts inside season-based partitions. Never randomly split overlapping windows.

| Role | Proposed season/data boundary | Permitted decisions |
|---|---|---|
| Inner stopping/pilot | Historical auxiliary rolling folds plus reliable 2022–23 NHSN windows; train before each stop block | Debugging, numerical choices, early stopping; report limited early target coverage |
| Development fold D1 | Predict 2023–24 with fitting information available before each issuance; initially use a pre-season frozen-fit version | Architecture screening and feature ablations |
| Development fold D2 | Predict 2024–25; earlier seasons may now train it, using historical availability constraints | Selection, robustness checks, calibration-method choice |
| Final retrospective holdout | **2025–26**; freeze all choices before running its forecast scores | One evaluation of the selected protocol plus prespecified comparators |
| Prospective confirmation | 2026–27 after configuration is published | Weekly stored forecasts before outcomes are known; strongest future evidence |

Use a documented season boundary, initially CDC epiweek 31 through the next year's week 30, to assign training windows. Score official challenge reference dates per season separately from a prespecified year-round evaluation. For each fold, remove training labels that land inside the held-out target period, even if their input windows begin earlier.

The final retrospective season is only “untouched” if it has not already been used for project-specific model choices. The existing note discusses published 2025 results, so record all prior exposure. If 2025–26 has already informed model selection, label it a historical evaluation and reserve 2026–27 as the true prospective test. Looking at date coverage here is not a model-skill evaluation, but the exposure log should say so.

Use two fitting modes with clear names:

- **Frozen-season fit:** parameters/scalers fitted before the season; contexts update each issuance. This isolates model generalization and reduces development cost.
- **Rolling operational fit:** weekly or monthly refitting on labels actually available at that fit time, including accrued current-season observations. Fix the cadence in development and execute it unchanged in the final evaluation.

A test protocol may learn from labels that become available earlier in that test season if that online update rule was fixed in advance. It may not inspect later outcomes to change hyperparameters, select checkpoints, or change the refit rule. Use inner stopping folds or a predetermined update budget, not the final test score.

### 10.2 Required comparators

1. Hub baseline and a local last-value/trend probabilistic baseline, with archived eligibility recorded.
2. B0, B1 without auxiliary-history transfer, and B1 without current auxiliary covariates.
3. Available real-time FluSight ensemble and original model submissions on common targets. Report missing submissions and operational fallback coverage.
4. InfluPaint and a matched conditional-flow head as later generative comparisons. Refit using eligible data; existing pretrained weights cannot be assumed free of held-out-season exposure.

The archived real-time ensemble is an external operational benchmark with its own information sources, not a controlled same-data experiment. The Flusion-style same-data GBQR comparison was dropped on 2026-09-14, so no controlled same-data comparator remains: any margin over the ensemble mixes architecture with the design's finalized-data and retrospective advantages and must be reported that way.

### 10.3 Metrics

The primary metric is WIS on integer admission quantiles at the official grid:

```text
0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
0.95, 0.975, 0.99
```

For a central `(1−alpha)` interval `[l,u]`, the interval score is its width plus `2/alpha` times the miss below/above the interval. With `K=11` central intervals, conventional WIS combines median absolute error with weight `1/2` and interval scores with weights `alpha/2`, then divides by `K+1/2`. Verify against the scoring implementation used by the hub. See [Bracher et al.](https://arxiv.org/abs/2005.12881).

Report paired mean WIS differences/ratios to the hub baseline and ensemble, and tournament relative WIS when comparing incomplete submission archives; freeze the comparison pool. Include MAE, 50/80/90/95% coverage, interval width, quantile reliability, submission completeness, and by-horizon revision/nowcast performance. Fair sample CRPS is a useful diagnostic, not a replacement for actual exported-quantile WIS.

Stratify by season, horizon, location size, epidemic growth/decline, peak proximity, low-count weeks, reporting completeness, and covariate coverage. Retrospectively defined peak proximity is an evaluation stratum only. Also report revision-size sensitivity rather than dropping difficult revised outcomes from the main score.

### 10.4 Dependence, uncertainty, and stress tests

Joint checks include within-state week-to-week changes, four-week totals, neighboring-state differences, region/national aggregates with aligned definitions, and distributions of maximum incidence over the **predicted block**. Add energy and variogram scores. Compare against a member-permuted version of the same predictions: permutations retain marginals while disrupting dependence, providing a useful diagnostic control.

Shared sample indices do not make national admissions equal the sum of states. First test that the observed US target equals a specified component sum across vintages. If it does, a sample-sum output can enforce it. If definitions/reporting differ, keep the native national task separate and model or report discrepancies. Do not silently include/exclude PR to force agreement, and never sum marginal quantiles as though they were aggregate quantiles.

For uncertainty in skill differences, use paired block resampling of issuance dates within season, retaining all locations/horizons in each block. Initial block length is four weeks, with 2/8-week sensitivity. Report season-level results separately: two development seasons do not justify narrow claims about future seasons. Overlapping horizons and common epidemic waves rule out treating every state-week as independent.

Stress cases: remove NSSP; remove NWSS; remove all auxiliary signals; delay NHSN by one/two releases; mask one geographic region; suppress low counts; introduce a missing new-source history; shift signal amplitude within training-supported bounds for diagnostics; and evaluate out-of-season COVID/RSV. Separate operationally plausible missingness tests from hypothetical distribution-shift tests.

### 10.5 Selection and acceptance

Prespecify mean primary WIS across D1/D2 with equal season weight, subject to submission completeness and no severe calibration failure, as the main selection criterion. Use a small number of finalists and identical scored tasks. A provisional adoption threshold is at least **3% lower pooled development WIS** than the FluSight ensemble on identical frozen tasks, with improvement in both seasons; this is a project decision threshold, not a statistical guarantee, and with no same-data floor it does not isolate the architecture. If results are within paired uncertainty, prefer the smaller/faster model or validate a simple ensemble.

Freeze architecture, feature registry, data eligibility, hyperparameters, training/refit cadence, calibrator, random-seed policy, and export procedure before the final holdout. Publish its outcome even if unfavorable. Do not retune against 2025–26 and continue calling it a final holdout.

## 11. Ablation sequence and bounded experiment budget

| Order | Experiment | What it resolves |
|---|---|---|
| 0 | Baselines and B0 on verified NHSN contexts | Data alignment, sample loss, and useful stochastic spread |
| 1 | Shared historical source-query examples on/off | Whether Flusion-style transfer survives in this implementation |
| 2 | Historical-as-of versus finalized-input training | Value of revision-aware fitting; finalized-input arm is explicitly an oracle/sensitivity comparison where appropriate |
| 3 | NSSP, virology, then wastewater group additions | Benefit of contemporary signals conditional on real availability |
| 4 | B0 versus B1 | Value of spatial exchange |
| 5 | CRPS draw count, transform, and auxiliary-weight choices | Numerical and objective robustness |
| 6 | Outage augmentation on/off; forced inference outages | Operational resilience |
| 7 | B2 longer temporal window if seasonal context remains insufficient | Value of temporal encoder capacity |
| 8 | Synthetic fraction or pretraining | Whether simulation helps B at all |
| 9 | Global latent versus extra local noise; optional joint loss | Dependence and local residual variation |
| 10 | One seed versus three; calibration; sample/quantile ensemble | Final reliability and cost |
| 11 | B1 shared representation with conditional-flow head | Whether direct score training or generative objective drives differences |

Start with roughly 12–20 single-seed configurations, then 3–5 finalists across three seeds and both development folds. This is a planning envelope; overlapping sequential decisions should reduce redundant fits. Record every run, including failures and excluded comparisons. Do not promise dozens of full vintage backtests in minutes without measurement.

## 12. Sampling, calibration, and submission

At validation/inference, generate **2,048 full trajectories** initially. Compare 512/2,048/8,192 on a fixed development subset to establish tail-quantile Monte Carlo stability. The 1% quantile is poorly estimated by a 100-member pool; the submission sample count need not limit internal generation.

Compute quantiles from the large sample pool, round nonnegative admission quantiles using a fixed rule, and verify monotonicity and the hub schema. For optional trajectory output, select 100 complete member IDs once for the whole forecast, keeping the same IDs across horizons and locations. Do not independently subsample each task. Record the seed and member/model identifiers.

A multi-seed forecast is a mixture: sample a model, then its latent, or use a documented stratified allocation of whole members. If estimating a fair CRPS of the mixture, draw model identities independently; fixed equal-per-seed allocations do not satisfy the simplest IID derivation. Exact submitted quantile WIS does not have that issue.

First diagnose undercoverage by horizon and subgroup. If calibration helps, fit a parsimonious horizon-specific sample transform on **out-of-sample development predictions**, with shrinkage across horizons and a fixed calibration-update rule. Apply it consistently to complete samples and recompute quantiles/aggregates. A location-specific spread adjustment may affect joint structure or sum identities, so rerun joint diagnostics and any reconciliation.

Keep raw and calibrated forecasts. A conformal or quantile correction would need its own dependence/coverage assumptions; do not claim finite-sample coverage guarantees for correlated epidemic time series by default.

For ensembling B with any quantile-only model, distinguish quantile averaging from a mixture of predictive distributions. Quantile averaging can be a validated primary-target output but does not uniquely define sample trajectories. Either omit the optional sample target for that quantile-only ensemble or define and validate a genuine joint mixture with its quantiles recomputed from the same distribution.

## 13. Peak targets and longer rollouts

The initial eight-output model supports a short-block maximum, not a seasonal peak forecast. It cannot report a full-season peak simply by taking the maximum over weeks −1 through 6.

A later rollout model must specify how unobserved future covariates are generated or masked, how predicted values enter subsequent contexts, and how release-age/missingness features evolve. Never feed realized future NSSP, wastewater, or admissions to a rollout. Train on its own predicted contexts for short rollouts before evaluating long ones.

Use direct blocks for short-range targets. For a seasonal extension, either advance one new week with a transition-trained model or advance nonoverlapping future blocks with consistent indexing. Retain the sampled model per trajectory; draw new process noise at the chosen transition frequency. Do not splice independently generated overlapping horizons.

Combine the observed season prefix, uncertainty in recent provisional weeks, and the sampled future suffix before extracting peak week/intensity. Establish the season endpoint and tie-breaking rule. Compare against a full-season conditional model or a simple seasonal baseline, and gate submission separately from hospitalization WIS. GenCast motivates the transition factorization, not the validity of this particular epidemic rollout.

## 14. Implementation deliverables and milestones

All paths below are **proposed** additions. Existing acquisition files remain the source of truth.

| Milestone | Concrete deliverables | Acceptance gate | Planning estimate |
|---|---|---|---|
| M0: experiment contract | `configs/b/data.yaml`, `splits.yaml`, target/date registry, source exposure log | Explicit target units, horizons, geographic support, and holdout rules | 1–2 working days |
| M1: model data | `src/tapestry/model_data/{normalize,vintages,features,windows,splits}.py`; normalized columnar store; coverage and leakage report | Fixed-cutoff invariance and valid source-query episodes; counts by season/source/support | 4–7 days, longer if archive gaps emerge |
| M2: baselines | `src/tapestry/baselines/`; vintage-aware hub-baseline and trend forecasts | Exported baseline WIS reproduces an independent scorer; same tasks and covariates documented | 2–4 days |
| M3: stochastic B0/B1 | `src/tapestry/models/{encoders,spatial,stochastic,decoder}.py`; `losses/`; train/predict scripts | Gradients through scores; correct masks/units; nonzero spread; finite samples; measured runtime | 3–5 days |
| M4: controlled evaluation | `src/tapestry/evaluation/`; D1/D2 forecasts and ablation report | Leakage audit passes; same-data comparisons, calibration, dependence, and outage results | 4–7 days plus measured compute |
| M5: freeze and holdout | Frozen manifest; complete 2025–26 forecast/score artifact | No model selection on holdout; full failure/coverage accounting | 2–3 days plus compute |
| M6: operating path | Versioned inference bundle; exporter; retry/fallback policy; dry-run report | Reproducible issuance from pinned inputs; valid schema, member identity, and deadline margin | 2–3 days |

Expect roughly **four to six working weeks** for a defensible first system if vintage coverage is adequate. This is an engineering estimate, not a promise. A scientific study with simulation, pathogen transfer, and seasonal rollouts is additional work. September/October work should prioritize the primary target and a reliable fallback before optional extensions.

Use one available accelerator for development; measure whether CPU inference is practical. Pilot reporting must include parameter count, peak memory, time per training update, total fit time, time for 2,048 draws, full-season backtest time, and data-materialization time. Cap architecture expansion using those measurements.

Necessary model tests should verify scientific failure modes rather than mirror layer definitions: future-vintage invariance, horizon/calendar alignment, inverse transform/units, masked-loss exclusion, fair-CRPS agreement with a trusted calculation, nonzero latent gradients, member-identity preservation, and geography aggregation only under a valid identity. Run existing data tests when integration changes their paths.

Each run records code/config hashes, source snapshot/file hashes, release and fit cutoffs, split ID, preprocessing parameters, target-vintage policy, package versions, RNG seeds, checkpoint, member count, calibration, wall time, and per-target predictions. Save forecasts before evaluation so a later scoring correction cannot silently regenerate them.

## 15. Failure decisions and the first build

| Observed failure | Response within the plan |
|---|---|
| Historical vintages missing for a signal | Mask it in the strict operational backtest; use a separately labeled finalized-data transfer/sensitivity study if justified |
| ILI/FluSurv transfer harms NHSN | Lower auxiliary weight, reduce source-specific shortcuts, or retain only beneficial source tasks; preserve negative result |
| Model ignores latent noise | Inspect sample spread and score terms, modulation initialization, and gradients; compare more draws or broader modulation before adding ad hoc noise |
| Good marginal WIS, poor trajectory changes/aggregates | Add validated joint objective or revise latent structure; withhold joint-derived targets until their own gates pass |
| Large states improve while small states become miscalibrated | Inspect source scale, subgroup coverage, and calibration; report both absolute and normalized skill |
| The design does not beat the ensemble on identical tasks | Keep it as an experimental or separately validated ensemble component; with no GBQR floor the submission fallback is the hub baseline |
| A required feed is late at an issuance | Apply trained missingness policy; if minimum target history is absent, use a tested baseline fallback and log it |

**First implementation:** deliver B0 with the six NHSN/NSSP channels, an eight-week context, explicit horizons, count-scale fair CRPS, one global latent, and baseline comparisons. Use the B0 results to choose the architecture, then add vintage handling in B1, wastewater in B2, and the combined vintage-plus-wastewater setup in B3. Freeze each protocol before the next stage and retain every prospective forecast for the following season.

## 16. Reading order and evidence files

Read Flusion §5/§7 for the transfer task and baseline, FGN §2 for the stochastic mechanism, and Pacchiardi et al. for the broader scoring-rule framework. Then use Ferro and Bracher to implement/check the score, CSDI for mask distinctions, and Scheuerer–Hamill for joint diagnostics. InfluPaint motivates the later data-mixture experiment. FiLM and TSMixer supply architectural detail. GenCast, flow matching, rectified flow, DDPM, RePaint, and CoPaint provide comparison/lineage context.

The archive contains source PDFs where downloads succeeded, extracted text for local search, a download script, a checksummed manifest, a reading index, a bibliography, the checked hub README, and the local coverage audit. The preprint PDFs are the archived versions; journal publication year/title can differ. The manifest, rather than a bare arXiv identifier, establishes which bytes were read.

## 17. Active pilot scope — 13 September 2026

The user now requests an initial finalized-data experiment beginning September
2023, without wastewater: NHSN admissions and NSSP ED proportions for flu,
COVID-19, and RSV. History length is configurable, initially **8 weeks**.
The [six-channel pilot](../data/build-b-finalized.md) implements materialization
and window/season/location queries. This overrides the earlier first-build
12-week and historical auxiliary-source scope for this experiment. The neural
model and fitting pipeline are not yet implemented. Historical release handling,
revision nowcasts, and operational evaluation remain later work; the earlier
fold assignments must be revised before fitting on September 2023 onward.

**Working B0 skeleton:** [Run the pilot](../workflows/training.md) documents the small
shared MLP, stochastic decoder, masked fair-CRPS fit, and sample/quantile prediction
commands now implemented. B1 spatial attention and evaluation remain future work.

## References

The project-root `references/` directory contains the downloaded papers, extracted text, checksummed manifest, and reading index. The inventory below records the sources used by this page.

| Paper | Role | Source | Archive |
|---|---|---|---|
| Flusion — Ray et al. (2024) | Transfer across signals, locations, transforms, and baseline | [paper](https://arxiv.org/abs/2407.19054v1) | `ray2024_flusion.pdf` |
| Generative diffusion models for spatiotemporal influenza forecasting — Lemaitre & Lessler (2026) | Simulation mixing and calibration context | [paper](https://arxiv.org/abs/2604.24913v1) | `lemaitre2026_influpaint.pdf` |
| FGN — Alet et al. (2025) | Shared noise and fair CRPS | [paper](https://arxiv.org/abs/2506.10772v1) | `alet2025_fgn.pdf` |
| GenCast — Price et al. (2023) | Conditional transitions and rollout context | [paper](https://arxiv.org/abs/2312.15796v2) | `price2023_gencast.pdf` |
| CSDI — Tashiro et al. (2021) | Conditioning and target masks | [paper](https://arxiv.org/abs/2107.03502v2) | `tashiro2021_csdi.pdf` |
| FiLM — Perez et al. (2018) | Feature-wise affine modulation | [paper](https://arxiv.org/abs/1709.07871v2) | `perez2018_film.pdf` |
| Pacchiardi et al. (2024) | Generative forecasting via scoring-rule minimization | [paper](https://jmlr.org/papers/v25/23-0038.html) | `pacchiardi2024_scoring_rules.pdf` |
| Ferro (2014) | Fair finite-ensemble scores | [paper](https://doi.org/10.1002/qj.2270) | `ferro2014_fair_scores.pdf` |
| Bracher et al. (2021) | WIS, quantile loss, and calibration | [paper](https://arxiv.org/abs/2005.12881v3) | `bracher2021_interval_scores.pdf` |
| Scheuerer & Hamill (2015) | Dependence-sensitive diagnostics | [paper](https://doi.org/10.1175/MWR-D-14-00269.1) | No local PDF |
| Lakshminarayanan et al. (2017) | Deep ensemble uncertainty | [paper](https://arxiv.org/abs/1612.01474v3) | `lakshminarayanan2017_deep_ensembles.pdf` |
| TSMixer — Chen et al. (2023) | Temporal and feature mixing alternative | [paper](https://arxiv.org/abs/2303.06053v5) | `chen2023_tsmixer.pdf` |
| Flow Matching — Lipman et al. (2023) | Conditional generative comparator | [paper](https://arxiv.org/abs/2210.02747v2) | `lipman2023_flow_matching.pdf` |
| Rectified Flow — Liu et al. (2023) | Straight interpolation comparator | [paper](https://arxiv.org/abs/2209.03003v1) | `liu2023_rectified_flow.pdf` |
| DDPM — Ho et al. (2020) | Diffusion background | [paper](https://arxiv.org/abs/2006.11239v2) | `ho2020_ddpm.pdf` |
| RePaint — Lugmayr et al. (2022) and CoPaint — Zhang et al. (2023) | Inpainting background | [RePaint](https://arxiv.org/abs/2201.09865v4), [CoPaint](https://arxiv.org/abs/2304.03322v1) | `lugmayr2022_repaint.pdf`, `zhang2023_copaint.pdf` |

The checked FluSight README is `references/flusight-hub-readme-2026-09-05.md`. The local event-date inventory is `references/local-data-coverage-2026-09-05.json`; it summarizes the explorer and must not be mistaken for historical vintage eligibility.
