# Scenario fields

Generated from `chromantis.model.scenario.Scenario` (`CODES`, `MEANING`) by `chromantis.evaluation.report.write_scenario_key`, rewritten by every report; do not edit by hand. A scenario string names only the fields that differ from these defaults, as `key=value` tokens joined by `,`; booleans are written `0`/`1`. See [the architecture](../architecture.md) and [Workflow](../workflow.md).

| Field | Type | Default | Allowed values | Meaning |
|---|---|---|---|---|
| `history_source` | str | `'finalized'` | `artificial`, `finalized`, `reported` | Choice A1, where the network's training histories come from: final reference histories (finalized), actual archived Wednesday reports (reported), or final histories made preliminary with artificial reporting errors (artificial; choices B and C). Was part of pilot_method. |
| `history_correction` | bool | `0` | `0`, `1` | Choice A2: correct the artificial training histories with the cross-fitted correction model (choice D) before the network learns from them (was pilot_method=two_stage/corrected). |
| `reconstruction_labels` | bool | `0` | `0`, `1` | Choice A3: also learn the final values of the last reconstruction_weeks context weeks (was pilot_method=joint). Labels, not inputs. |
| `error_scope` | str | `'all'` | `all`, `early_actual` | Apply artificial reporting errors in all training seasons, or early two with actual/proxy recent-season reports. |
| `error_signals` | str | `'admissions'` | `admissions`, `all`, `targets` | Choice C: admissions perturbs counts only; targets also perturbs ED but keeps covariates unchanged; all perturbs both targets and covariates. |
| `corrector_examples` | str | `'synthetic'` | `real`, `synthetic`, `synthetic_then_real` | Choice D, what the correction trees (the nowcaster) learn from: artificial errors paired with final values (synthetic), real archived reports paired with their mature values (real), or synthetic pretraining then real (synthetic_then_real, MLP only). Was pilot_nowcaster. |
| `corrector_model` | str | `'tree'` | `mlp`, `tree` | Correction model: gradient-boosted trees (tree) or a residual MLP (mlp). Was pilot_nowcaster. |
| `forecast_targets` | str | `'all'` | `all`, `flu` | All six targets or flu admissions + ED only; exported forecasts and scoring follow this scope. |
| `pathogen_inputs` | str | `'all'` | `all`, `flu`, `flu_covid`, `flu_ed`, `flu_hosp`, `flu_rsv` | Allowed admissions/ED input pathogens in both forecaster and nowcaster; excluded values and masks are removed. |
| `ili_units` | str | `'own'` | `flu_scaled`, `own` | Own-source ILI proportions or auxiliary flu-head pseudo-tasks rescaled by fitting-only modern Q95 / historical ILI Q95. |
| `ili_steps` | int | `200` | any int (checked in `Scenario.__post_init__`) | Historical forecasting pretraining updates; each uses 64 auxiliary examples (series models) or batch_size historical panel episodes (flu MLP). |
| `correction_weeks` | int | `8` | any int (checked in `Scenario.__post_init__`) | Recent admissions ages corrected by the nowcaster, clipped to lookback. |
| `correction_ed` | bool | `0` | `0`, `1` | Also fit/apply report-to-mature ED correction on the recent correction_weeks, restricted to allowed input channels; future labels unchanged. |
| `ili_training` | str | `'none'` | `joint`, `none`, `pretrain` | No historical transfer, forecasting pretraining, or joint forecasting on the pinned pre-2022 ILI archive. |
| `ili_path` | str | `'data/processed/historical_ili.npz'` | any str (checked in `Scenario.__post_init__`) | Historical ILI auxiliary dataset; its hash is pinned by the experiment manager. |
| `ili_weight` | float | `0.25` | any float (checked in `Scenario.__post_init__`) | Historical ILI loss weight for joint training after within-history normalization. |
| `growth_anchor` | bool | `0` | `0`, `1` | Add damped observed two-week transformed growth to the level anchor. |
| `reporting_probability` | float | `1.0` | any float (checked in `Scenario.__post_init__`) | Probability a training episode receives artificial reporting errors. |
| `reporting_random_strength` | bool | `0` | `0`, `1` | Scale each drawn error window by a uniform(0, 1) multiplier. |
| `reporting_recent` | int | `0` | any int (checked in `Scenario.__post_init__`) | Perturb only this many most recent context weeks; 0 = all. |
| `joint_weight` | float | `1.0` | any float (checked in `Scenario.__post_init__`) | joint only: weight of the reconstruction loss relative to the forecast loss. |
| `correction_penalty` | float | `10.0` | any float (checked in `Scenario.__post_init__`) | Ridge penalty of the correction trees' linear leaves. |
| `correction_strength` | float | `1.0` | any float (checked in `Scenario.__post_init__`) | Multiplier of the fitted correction, 1 = full. |
| `reporting_missingness` | bool | `1` | `0`, `1` | Transfer donor reporting masks with the errors; false transports numerical errors only and retains native input availability. |
| `reporting_method` | str | `'local_log'` | `calendar_log`, `local_additive`, `local_log`, `phase_log`, `synchronous_log`, `synchronous_phase_log` | Donor matching of artificial reporting errors; see dataset/reporting_error.py. |
| `reporting_strength` | float | `1.0` | any float (checked in `Scenario.__post_init__`) | Multiplier of drawn log reporting errors. |
| `reconstruction_weeks` | int | `2` | any int (checked in `Scenario.__post_init__`) | joint only: recent context weeks whose final values are reconstruction labels. |
| `lookback` | int | `12` | any int (checked in `Scenario.__post_init__`) | Context weeks per episode (the history the network sees). |
| `count_transform` | str | `'fourth_root'` | `fourth_root`, `log1p`, `rate`, `raw`, `sqrt` | Admissions in model space: raw counts, rate per 100,000, or its sqrt / fourth root / log1p (`model/network.py` `transform_counts`). |
| `ed_transform` | str | `'linear'` | `fourth_root`, `linear`, `logit` | ED proportions in model space (`model/network.py`); scores stay in native units. |
| `geography` | bool | `1` | `0`, `1` | Adds log population and a native-US flag per location as features. |
| `coordinates` | bool | `0` | `0`, `1` | Census state internal-point latitude/longitude and non-US indicator. |
| `dynamics` | bool | `1` | `0`, `1` | Recent-dynamics feature block (30 slope/acceleration/age/validity features); see architecture.md. |
| `loss_weights` | str | `'objective'` | `balanced_admissions`, `flu_ed`, `flu_hosp_ed`, `flu_only`, `influenza_first`, `objective` | Training-loss weight per channel (`model/objective.py` `LOSS_WEIGHTS`); `objective` = the score's target weights. |
| `encoder` | str | `'mlp'` | `conv`, `mlp`, `multiscale_conv`, `series_mixer`, `series_mlp` | Temporal context encoder; see architecture.md. |
| `spatial` | str | `'none'` | `attention`, `distance`, `gated_pool`, `gravity`, `joint_location_target`, `national_broadcast`, `neighbors`, `none`, `pathogen_spatial`, `pooled`, `target_spatial` | Cross-location information exchange (none, shared attention, pathogen/target/joint scopes); see architecture.md. |
| `heads` | str | `'shared'` | `shared`, `state_us` | State and US output heads shared or separate (`state_us`). |
| `decoder` | str | `'legacy'` | `legacy`, `quantile`, `quantile_small`, `residual2` | Horizon decoder: existing modulated residual (`legacy`), `residual2`, or direct ordered quantiles; see architecture.md. |
| `decoder_blocks` | int | `0` | any int (checked in `Scenario.__post_init__`) | Residual decoder depth (0 = original head). |
| `sum_wis_weight` | float | `0.0` | any float (checked in `Scenario.__post_init__`) | Auxiliary four-week flu admission sum WIS weight; sampled heads only; normalized by four times training-only admission Q95. |
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
| `covariate_encoder` | str | `'raw'` | `growth`, `raw`, `shared`, `smooth`, `summary` | Raw standardized history, signed-log trailing-three-week smoothing, six summaries, a shared 4-dimensional encoder, or recent growth (level, 1/2-week log growth, acceleration) plus coverage/age. |
| `signal_features` | str | `'none'` | `multiscale`, `none`, `smooth_multiscale` | Optional causal 3/6/12-week level, slope and curvature features for targets and covariates, with optional three-week smoothing. |
| `covariate_set` | str | `''` | `+`-joined subset of `inpatient`, `outpatient`, `ww_wval_like`, `ww_pct_rank`, `kinsa`, `ilinet`, `clinical_lab`, `flusurv` | `+`-joined covariate source groups fed to the context encoder; '' = none; see workflow.md and experiments/b-2-t0/index.md#protocol. |
| `input_normalization` | str | `'none'` | `b0`, `none` | none, or B0 per-location transformed target scales fitted on training contexts only. |
| `validation_calendar` | str | `'season'` | `b0`, `season` | season-relative blocks, or B0 blocks counted from the first stored week of each season. |
| `evaluation_seasons` | str | `'all'` | `all`, `production`, `recent_two` | all three held-out seasons, recent_two (2025-26 and 2024-25), or production (fit on all four seasons 2022-23 to 2025-26; forecast 2026-27 inputs, unscored). |
| `validation_weeks` | int | `3` | any int (checked in `Scenario.__post_init__`) | patience > 0 only: consecutive early-stopping weeks hidden per block (design §4). |
| `validation_spacing` | int | `16` | any int (checked in `Scenario.__post_init__`) | patience > 0 only: one validation block every this many weeks of a training season. |
| `validation_offset` | int | `4` | any int (checked in `Scenario.__post_init__`) | patience > 0 only: week of each training season where the first block starts (0-based, counted from the season's first epiweek, CDC week 31). |
| `error_seasons` | str | `'latest'` | `all`, `latest` | Choice B: archived seasons supplying artificial reporting errors and real correction pairs: latest permitted training season, or every archived training season with recency weights 1, 1/2, 1/4. |
| `actual_share` | float | `0.0` | any float (checked in `Scenario.__post_init__`) | Probability a training episode uses the actual archived report (finalized fill where none) instead of an artificial-error draw. |
| `correction_realizations` | int | `1` | any int (checked in `Scenario.__post_init__`) | Independent artificial-report-then-correction realizations per training episode; one is drawn per minibatch. |
| `uncorrected_share` | float | `0.0` | any float (checked in `Scenario.__post_init__`) | Probability a corrected-history training episode is shown uncorrected instead (robustness to correction failure). |
| `correction_noise` | float | `0.0` | any float (checked in `Scenario.__post_init__`) | Scale of sampled out-of-fold correction residuals added to corrected admissions; 0 = point correction. Evaluation uses 16 sampled histories per issuance. |
| `correction_noise_train` | bool | `0` | `0`, `1` | Also add sampled correction residuals to corrected training histories each minibatch. |
| `log_loss_weight` | float | `0.0` | any float (checked in `Scenario.__post_init__`) | Weight of an extra weekly flu-admission loss on the log(1 + count) scale, added to the native-scale loss. |
| `covariate_dropout` | float | `0.0` | any float (checked in `Scenario.__post_init__`) | Probability a training episode has all covariates (e.g. Kinsa) hidden. |
| `training_window` | str | `'all'` | `all`, `last2`, `recent2` | all = every non-held-out season; recent2 = two preceding seasons; last2 = latest two permitted non-held-out seasons (nested removal of oldest training season). |
| `stress_views` | bool | `0` | `0`, `1` | Also evaluate delayed-admission and covariate-missing input views of the same fitted model. |
| `error_reference` | str | `''` | any str (checked in `Scenario.__post_init__`) | Choice B: prescribed reporting-error season for training errors and prescribed evaluation inputs, independent of the held-out epidemic season; empty = the fold's own seasons. |
| `evaluation_inputs` | str | `'reported'` | `prescribed`, `reported` | Choice E: evaluate on real archived Wednesday reports (reported) or on final histories made preliminary by the prescribed error process of error_reference (prescribed). Was evaluation_vintaging. |
| `evaluation_draws` | int | `1` | any int (checked in `Scenario.__post_init__`) | Independent artificial evaluation histories (1, 3 or 5) when evaluation_inputs=prescribed. |
| `forecast_view` | str | `'corrected'` | `corrected`, `half`, `raw` | Choice F, how the fitted pair is used at forecast time, part of the recipe: inputs as given (raw), newest weeks corrected by the fold's correction model (corrected), or the 50/50 mixture of those two forecast distributions (half). Ranking and production use this view; the others are diagnostics. |
