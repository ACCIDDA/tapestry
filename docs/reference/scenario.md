# Scenario fields

Generated from `tapestry.model.scenario.Scenario` (`CODES`, `MEANING`) by `tapestry.evaluation.plots.write_scenario_key`, rewritten by every report; do not edit by hand. A scenario string names only the fields that differ from these defaults, as `key=value` tokens joined by `,`; booleans are written `0`/`1`. "Design" = [the unified design](../design/restructure-2026-unified.md); other documents are under `docs/`.

| Field | Type | Default | Allowed values | Meaning |
|---|---|---|---|---|
| `lookback` | int | `12` | any int (checked in `Scenario.__post_init__`) | Context weeks per episode (the history the network sees). |
| `count_transform` | str | `'fourth_root'` | `fourth_root`, `log1p`, `rate`, `raw`, `sqrt` | Admissions in model space: raw counts, rate per 100,000, or its sqrt / fourth root / log1p (`model/network.py` `transform_counts`). |
| `ed_transform` | str | `'linear'` | `fourth_root`, `linear`, `logit` | ED proportions in model space (`model/network.py`); scores stay in native units. |
| `geography` | bool | `1` | `0`, `1` | Adds log population and a native-US flag per location as features. |
| `dynamics` | bool | `1` | `0`, `1` | Recent-dynamics feature block (30 slope/acceleration/age/validity features); see design/b0.1.md. |
| `loss_weights` | str | `'objective'` | `balanced_admissions`, `flu_only`, `influenza_first`, `objective` | Training-loss weight per channel (`model/objective.py` `LOSS_WEIGHTS`); `objective` = the score's target weights. |
| `encoder` | str | `'mlp'` | `conv`, `mlp`, `multiscale_conv` | Temporal context encoder; see design/b0.1.md. |
| `spatial` | str | `'none'` | `attention`, `joint_location_target`, `none`, `pathogen_spatial`, `target_spatial` | Cross-location information exchange (none, shared attention, pathogen/target/joint scopes); see design/b0.1.md. |
| `heads` | str | `'shared'` | `shared`, `state_us` | State and US output heads shared or separate (`state_us`). |
| `decoder` | str | `'legacy'` | `legacy`, `residual2` | Horizon decoder: existing modulated residual (`legacy`) or `residual2`; see design/b0.1.md. |
| `noise` | str | `'global'` | `global`, `local` | Global latent noise, or global plus a per-location latent (`local`). |
| `us_error` | str | `'none'` | `none`, `shared_factor` | Extra common noise factor (`shared_factor`); see design/b0.1.md and design/b1.md. |
| `latent` | int | `16` | any int (checked in `Scenario.__post_init__`) | Global latent (noise) dimension. |
| `width` | int | `64` | any int (checked in `Scenario.__post_init__`) | Hidden width of the network. |
| `epochs` | int | `50` | any int (checked in `Scenario.__post_init__`) | Epoch cap (the fixed number of epochs when patience = 0). |
| `patience` | int | `0` | any int (checked in `Scenario.__post_init__`) | Early-stopping patience in epochs; 0 = fixed epochs, no validation weeks; > 0 selects the epoch on the validation weeks, then refits on the full training seasons (design §4). |
| `batch_size` | int | `8` | any int (checked in `Scenario.__post_init__`) | Episodes per optimizer step. |
| `members` | int | `128` | any int (checked in `Scenario.__post_init__`) | Sampled members per training episode (fair CRPS loss). |
| `lr` | float | `0.001` | any float (checked in `Scenario.__post_init__`) | Adam learning rate. |
| `head_sharing` | str | `'shared'` | `pathogen`, `shared`, `target` | Output sharing: shared, three pathogen heads, or six target heads; see design/b0.1.md. |
| `annual_calendar` | bool | `1` | `0`, `1` | Annual sine/cosine and Christmas-timing features. |
| `location_embedding` | int | `0` | any int (checked in `Scenario.__post_init__`) | Dimension of a learned location-ID embedding (0 = none). |
| `fit_partition` | str | `'all'` | `all`, `pathogen`, `target` | One model for all six targets, or separately fitted models per pathogen / target group (each sees all six inputs); see design/b0.1.md. |
| `validation_members` | int | `256` | any int (checked in `Scenario.__post_init__`) | Members drawn for the early-stopping validation loss. |
| `weight_decay` | float | `0.0` | any float (checked in `Scenario.__post_init__`) | Adam weight decay. |
| `supplied_final` | bool | `0` | `0`, `1` | The network receives a known-final flag channel per cell; see design/b1.md. |
| `mask_rate` | float | `0.0` | any float (checked in `Scenario.__post_init__`) | Probability an episode receives an artificial missingness pattern in training (not a fraction of cells); see design/b1.md. |
| `mask_recent` | float | `0.5` | any float (checked in `Scenario.__post_init__`) | Share of masked episodes whose pattern hides recent reports (with mask_gap, mask_outage sums to 1). |
| `mask_gap` | float | `0.3` | any float (checked in `Scenario.__post_init__`) | Share of masked episodes whose pattern is a local gap in one location history. |
| `mask_outage` | float | `0.2` | any float (checked in `Scenario.__post_init__`) | Share of masked episodes whose pattern is a whole-channel outage. |
| `covariate_set` | str | `''` | `+`-joined subset of `inpatient`, `outpatient`, `ww_wval_like`, `ww_pct_rank`, `kinsa` | `+`-joined covariate source groups fed to the context encoder; '' = none; see workflows/training.md and design/b2.md. |
| `input_mode` | str | `'finalized'` | `finalized`, `vintaged` | How episodes are cut from the panel: finalized truth, or Wednesday-vintage as-of context (design §3). |
| `asof_weeks` | int | `2` | any int (checked in `Scenario.__post_init__`) | Vintaged only: most recent context weeks whose targets are as visible at the issuance; older weeks take final truth (design §3). |
| `validation_weeks` | int | `3` | any int (checked in `Scenario.__post_init__`) | patience > 0 only: consecutive early-stopping weeks hidden per block (design §4). |
| `validation_spacing` | int | `16` | any int (checked in `Scenario.__post_init__`) | patience > 0 only: one validation block every this many weeks of a training season. |
| `validation_offset` | int | `4` | any int (checked in `Scenario.__post_init__`) | patience > 0 only: week of each training season where the first block starts (0-based, counted from the season's first epiweek, CDC week 31). |
