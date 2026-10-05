# Scenario fields

Generated from `tapestry.model.scenario.Scenario` (`CODES`, `MEANING`) by `tapestry.evaluation.plots.write_scenario_key`, rewritten by every report; do not edit by hand. A scenario string names only the fields that differ from these defaults, as `key=value` tokens joined by `,`; booleans are written `0`/`1`. "Design" = [the architecture](../architecture.md); execution is described in [Workflow](../workflow.md).

| Field | Type | Default | Allowed values | Meaning |
|---|---|---|---|---|
| `task` | str | `'forecast'` | `finalize`, `forecast`, `nowcast`, `pipeline` | forecast, nowcast, an independently fitted nowcast-to-forecast pipeline, or per-signal boundary-week finalization. |
| `reporting_augmentation` | str | `'none'` | `none`, `vintage`, `nowcast` | Stochastic joint training-season reporting-error windows, raw or after causal nowcasting; real-vintage scoring. Requires scheduled-final forecasting without a finality channel. |
| `reporting_missingness` | bool | `True` | `0`, `1` | Transport donor missingness with reporting augmentation; false retains native input availability while transporting numerical errors. |
| `replay_from` | str | `''` | any str (checked in `Scenario.__post_init__`) | Completed experiment supplying fixed CV checkpoints for inference-only input replay. |
| `replay_inputs` | str | `'none'` | `finalized`, `none`, `nowcast`, `vintage` | Fixed-checkpoint replay: finalized B2 inputs, full vintage history, or vintage plus eight-week seasonal target nowcasts. |
| `replay_flags` | str | `'native'` | `available`, `native`, `off` | Diagnostic finality encoding: native semantics, available to reproduce B2 training encoding, or off. Available does not claim that revisions are final. |
| `finalization_cv` | str | `'rolling'` | `rolling`, `season` | finalize only: three recent four-Wednesday rolling holdouts, or forward season holdouts. |
| `finalization_loss` | str | `'mae'` | `mae` | finalize only: triangle calibration minimizes normalized native-unit MAE, equally weighted by location. |
| `finalization_model` | str | `'triangle'` | `adaptive`, `adaptive_chain`, `seasonal`, `triangle` | Reported-value curve: robust triangle, seasonal direct, adaptive direct, or adaptive weekly chain. |
| `finalization_gap` | str | `'ridge'` | `proxy`, `ridge`, `seasonal`, `trend` | Missing targets: ridge, historical seasonal growth, damped trend, or observable national proxy growth. |
| `finalization_scope` | str | `'all'` | `all`, `targets` | Fit all supported signals or only the six NHSN/NSSP targets. |
| `finalization_halflife` | float | `8.0` | any float (checked in `Scenario.__post_init__`) | Adaptive curve recent-pair half-life in weeks. |
| `finalization_pool` | float | `4.0` | any float (checked in `Scenario.__post_init__`) | Adaptive/seasonal local state pooling strength in effective weeks. |
| `finalization_statistic` | str | `'mean'` | `mean`, `median` | Mean ratio of weighted totals or exposure-weighted median development ratio. |
| `finalization_quantize` | bool | `0` | `0`, `1` | Round NSSP point predictions and point comparators to the archived 0.0001 proportion grid. |
| `finalization_maturity` | int | `4` | any int (checked in `Scenario.__post_init__`) | finalize only: minimum reference age and training-label gap, in weeks; not a guarantee of finality. |
| `finalization_weeks` | int | `1` | any int (checked in `Scenario.__post_init__`) | finalize only: reconstruct this many completed weeks ending at each signal's T-X boundary. |
| `nowcast_weeks` | int | `2` | any int (checked in `Scenario.__post_init__`) | Completed weeks reconstructed, ending at the context Saturday. |
| `nowcast.<field>` | stage override | inherit | Model/training fields | Pipeline stage overrides written as nowcast.<field>=value; unprefixed fields supply defaults. |
| `forecast.<field>` | stage override | inherit | Model/training fields | Pipeline stage overrides written as forecast.<field>=value; unprefixed fields supply defaults. |
| `nowcast_members` | int | `16` | any int (checked in `Scenario.__post_init__`) | Cached out-of-fold history draws per training episode. |
| `lookback` | int | `12` | any int (checked in `Scenario.__post_init__`) | Context weeks per episode (the history the network sees). |
| `count_transform` | str | `'fourth_root'` | `fourth_root`, `log1p`, `rate`, `raw`, `sqrt` | Admissions in model space: raw counts, rate per 100,000, or its sqrt / fourth root / log1p (`model/network.py` `transform_counts`). |
| `ed_transform` | str | `'linear'` | `fourth_root`, `linear`, `logit` | ED proportions in model space (`model/network.py`); scores stay in native units. |
| `geography` | bool | `1` | `0`, `1` | Adds log population and a native-US flag per location as features. |
| `coordinates` | bool | `0` | `0`, `1` | Census state internal-point latitude/longitude and non-US indicator. |
| `dynamics` | bool | `1` | `0`, `1` | Recent-dynamics feature block (30 slope/acceleration/age/validity features); see architecture.md. |
| `loss_weights` | str | `'objective'` | `balanced_admissions`, `flu_only`, `influenza_first`, `objective` | Training-loss weight per channel (`model/objective.py` `LOSS_WEIGHTS`); `objective` = the score's target weights. |
| `encoder` | str | `'mlp'` | `conv`, `mlp`, `multiscale_conv` | Temporal context encoder; see architecture.md. |
| `spatial` | str | `'none'` | `attention`, `distance`, `gated_pool`, `gravity`, `joint_location_target`, `national_broadcast`, `neighbors`, `none`, `pathogen_spatial`, `pooled`, `target_spatial` | Cross-location information exchange (none, shared attention, pathogen/target/joint scopes); see architecture.md. |
| `heads` | str | `'shared'` | `shared`, `state_us` | State and US output heads shared or separate (`state_us`). |
| `decoder` | str | `'legacy'` | `legacy`, `residual2` | Horizon decoder: existing modulated residual (`legacy`) or `residual2`; see architecture.md. |
| `noise` | str | `'global'` | `global`, `local` | Global latent noise, or global plus a per-location latent (`local`). |
| `us_error` | str | `'none'` | `none`, `shared_factor` | Extra common noise factor (`shared_factor`); see architecture.md. |
| `latent` | int | `16` | any int (checked in `Scenario.__post_init__`) | Global latent (noise) dimension. |
| `width` | int | `64` | any int (checked in `Scenario.__post_init__`) | Hidden width of the network. |
| `epochs` | int | `50` | any int (checked in `Scenario.__post_init__`) | Epoch cap (the fixed number of epochs when patience = 0). |
| `patience` | int | `0` | any int (checked in `Scenario.__post_init__`) | Early-stopping patience in epochs; 0 = fixed epochs, no validation weeks; > 0 selects the epoch on the validation weeks, then refits on the full training seasons (design §4). |
| `batch_size` | int | `8` | any int (checked in `Scenario.__post_init__`) | Episodes per optimizer step. |
| `members` | int | `128` | any int (checked in `Scenario.__post_init__`) | Sampled members per training episode (fair CRPS loss). |
| `lr` | float | `0.001` | any float (checked in `Scenario.__post_init__`) | Adam learning rate. |
| `head_sharing` | str | `'shared'` | `pathogen`, `shared`, `target` | Output sharing: shared, three pathogen heads, or six target heads; see architecture.md. |
| `annual_calendar` | bool | `1` | `0`, `1` | Annual sine/cosine and Christmas-timing features. |
| `location_embedding` | int | `0` | any int (checked in `Scenario.__post_init__`) | Dimension of a learned location-ID embedding (0 = none). |
| `fit_partition` | str | `'all'` | `all`, `pathogen`, `target` | One model for all six targets, or separately fitted models per pathogen / target group (each sees all six inputs); see architecture.md. |
| `validation_members` | int | `256` | any int (checked in `Scenario.__post_init__`) | Members drawn for the early-stopping validation loss. |
| `weight_decay` | float | `0.0` | any float (checked in `Scenario.__post_init__`) | Adam weight decay. |
| `supplied_final` | bool | `0` | `0`, `1` | The network receives a known-final flag channel per cell; see architecture.md. |
| `mask_rate` | float | `0.0` | any float (checked in `Scenario.__post_init__`) | Probability an episode receives an artificial missingness pattern in training (not a fraction of cells); see architecture.md. |
| `mask_recent` | float | `0.5` | any float (checked in `Scenario.__post_init__`) | Share of masked episodes whose pattern hides recent reports (with mask_gap, mask_outage sums to 1). |
| `mask_gap` | float | `0.3` | any float (checked in `Scenario.__post_init__`) | Share of masked episodes whose pattern is a local gap in one location history. |
| `mask_outage` | float | `0.2` | any float (checked in `Scenario.__post_init__`) | Share of masked episodes whose pattern is a whole-channel outage. |
| `covariate_encoder` | str | `'raw'` | `raw`, `shared`, `smooth`, `summary` | Raw standardized history, signed-log trailing-three-week smoothing, six summaries, or a shared 4-dimensional encoder plus coverage/age. |
| `signal_features` | str | `'none'` | `multiscale`, `none`, `smooth_multiscale` | Optional causal 3/6/12-week level, slope and curvature features for targets and covariates, with optional three-week smoothing. |
| `covariate_set` | str | `''` | `+`-joined subset of `inpatient`, `outpatient`, `ww_wval_like`, `ww_pct_rank`, `kinsa`, `ilinet`, `clinical_lab`, `flusurv` | `+`-joined covariate source groups fed to the context encoder; '' = none; see workflow.md and experiments/b-2-t0/index.md#protocol. |
| `input_mode` | str | `'finalized'` | `finalized`, `finalized_available`, `scheduled_final`, `vintaged` | scheduled_final supplies T-0 final targets and source-specific T-0/T-1 covariates; finalized truth, finalized_available (final truth masked by Wednesday reporting availability), or Wednesday-vintage context (design §3). |
| `training_inputs` | str | `'same'` | `finalized`, `same` | same as forecasting, or complete finalized target and covariate histories during fitting only. |
| `input_normalization` | str | `'none'` | `b0`, `none` | none, or B0 per-location transformed target scales fitted on training contexts only. |
| `validation_calendar` | str | `'season'` | `b0`, `season` | season-relative blocks, or B0 blocks counted from the first stored week of each season. |
| `evaluation_seasons` | str | `'all'` | `all`, `recent_two` | all three held-out seasons, or recent_two (2025-26 and 2024-25). |
| `asof_weeks` | int | `2` | any int (checked in `Scenario.__post_init__`) | Standalone forecast only (nowcast/pipeline use all-as-of history): most recent context weeks whose targets are as visible at the issuance; older weeks take final truth (design §3). |
| `validation_weeks` | int | `3` | any int (checked in `Scenario.__post_init__`) | patience > 0 only: consecutive early-stopping weeks hidden per block (design §4). |
| `validation_spacing` | int | `16` | any int (checked in `Scenario.__post_init__`) | patience > 0 only: one validation block every this many weeks of a training season. |
| `validation_offset` | int | `4` | any int (checked in `Scenario.__post_init__`) | patience > 0 only: week of each training season where the first block starts (0-based, counted from the season's first epiweek, CDC week 31). |

### Historical reporting-error training

`reporting_augmentation=nowcast,reporting_missingness=0` perturbs finalized
training inputs with historical nowcast errors while retaining native input
availability and finalized future labels. `reporting_method` selects
`local_log` (default), `calendar_log`, or `local_additive`.
`reporting_strength` in (0, 1] scales the signed error before transport.
Local matching uses the same location's seasonal timing, recent levels and
changes; calendar matching uses timing alone. Each location draws its own
joint age/signal window. [Exact assumptions and two-season C1 protocol](../experiments/c1-local-errors-20261002/index.md).

### Context nowcaster extensions

`finalization_model=conditional_chain` (or `conditional`) conditions the seasonal
fit on growth/holiday context; `finalization_growth` controls its bandwidth.
`finalization_model=context_residual` trains a causal trajectory residual with
`finalization_penalty` ridge shrinkage and requires at least four output weeks.
Replay supports `replay_nowcaster`, `replay_growth`, `replay_penalty`, and
`replay_schedule=1` for the documented availability hypothesis with explicit
frozen-final proxies. [Protocol and assumptions](../experiments/context-nowcast-20261001/index.md).

`finalization_features=age` gives growth, holiday and local effects separate
smooth level/slope age profiles; `momentum` adds observable vintage-to-vintage
changes, and `age_momentum` combines them. `finalization_gate=admissions` retains
the seasonal ED estimates; `causal` selects residual strength using previously
matured trajectory errors. Matching `replay_features` and `replay_gate` options
replay these exact nowcasts. `replay_uncertainty` adds causal joint-history
bootstrap uncertainty (0 = deterministic; experimental strengths 0.5 and 1).
Residual calibration currently requires `finalization_scope=targets`; its count
and ED-fraction conventions have not been extended to ancillary source units.
`finalization_strength` (and `replay_strength`) scales the additional residual
between zero and one before output rounding. The seasonal base remains intact.
