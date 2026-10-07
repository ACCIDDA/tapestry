"""One scenario dataclass for the unified model, with a self-describing string codec.

Replaces the old `TrainingScenario` -> `B1Scenario` -> `B2Scenario` subclass chain
and their hand-written positional string codecs (`b0:h12:tr_4rt:...`). A scenario
string here is `key=value` tokens joined by `,`, one token per field that differs
from `Scenario()`'s default, in any order:

    >>> Scenario(width=32).scenario_string
    'width=32'
    >>> Scenario.from_string('width=32') == Scenario(width=32)
    True

Consequences, all intentional (see docs/architecture.md):
- A string only ever needs to name what differs from the default. Two strings
  naming the same non-default fields are equal regardless of token order.
- Adding a new dataclass field never breaks an existing saved string; the
  string simply doesn't mention it, so it takes its default.
- A typo in an enum-like field's value still raises (`CODES`), rather than
  silently becoming a new dataclass field, because unknown keys are rejected
  and known enum keys are checked against a fixed value set.

`covariate_set=''` + `input_mode='finalized'` reproduces plain B0. `covariate_set=''`
+ `input_mode='vintaged'` reproduces B1-direct. A nonempty `covariate_set` with
`input_mode='vintaged'` reproduces B2. One scenario space, no subclassing.
"""
from dataclasses import dataclass, fields, replace
import hashlib

# Enum-like string fields: a fixed set of legal values, so a typo raises instead
# of silently becoming a new field the way an unconstrained string would.
CODES = {
    'reporting_method': {'local_log', 'calendar_log', 'local_additive', 'synchronous_log', 'phase_log'},
    'weekend_family': {'none', 'direct', 'two_stage', 'joint'},
    'reporting_augmentation': {'none', 'vintage', 'nowcast'},
    'replay_inputs': {'none', 'finalized', 'vintage', 'nowcast'},
    'replay_flags': {'native', 'available', 'off'},
    'replay_nowcaster': {'selected', 'conditional_chain', 'conditional', 'context_residual'},
    'training_inputs': {'same', 'finalized'},
    'input_normalization': {'none', 'b0'},
    'validation_calendar': {'season', 'b0'},
    'evaluation_seasons': {'all', 'recent_two', 'production'},
    'count_transform': {'raw', 'rate', 'sqrt', 'fourth_root', 'log1p'},
    'ed_transform': {'linear', 'logit', 'fourth_root'},
    'loss_weights': {'influenza_first', 'balanced_admissions', 'flu_only', 'flu_hosp_ed', 'objective'},
    'signal_features': {'none', 'multiscale', 'smooth_multiscale'},
    'covariate_encoder': {'raw', 'smooth', 'summary', 'shared', 'growth'},
    'encoder': {'mlp', 'conv', 'multiscale_conv', 'series_mlp', 'series_mixer'},
    'pilot_method': {'none', 'finalized', 'reported', 'errors', 'corrected', 'two_stage', 'joint'},
    'revision_scope': {'all', 'early_actual'},
    'revision_signals': {'all', 'admissions'},
    'pilot_nowcaster': {'synthetic_tree', 'real_tree', 'real_mlp', 'pretrained_mlp'},
    'ili_training': {'none', 'pretrain', 'joint'},
    'vintage_seasons': {'latest', 'all'},
    'training_window': {'all', 'recent2'},
    'ili_units': {'own', 'flu_scaled'},
    'forecast_targets': {'all', 'flu'},
    'pathogen_inputs': {'all', 'flu', 'flu_covid', 'flu_rsv'},
    'spatial': {'neighbors', 'distance', 'gravity', 'none', 'pooled', 'attention', 'pathogen_spatial', 'target_spatial', 'joint_location_target', 'national_broadcast', 'gated_pool'},
    'heads': {'shared', 'state_us'},
    'decoder': {'legacy', 'residual2', 'quantile', 'quantile_small'},
    'noise': {'global', 'local'},
    'us_error': {'none', 'shared_factor'},
    'head_sharing': {'shared', 'pathogen', 'target'},
    'fit_partition': {'all', 'pathogen', 'target'},
    'input_mode': {'finalized', 'finalized_available', 'scheduled_final', 'vintaged', 'reported'},
    'task': {'forecast', 'nowcast', 'pipeline', 'finalize'},
    'finalization_cv': {'rolling', 'season'},
    'finalization_loss': {'mae'},
    'finalization_model': {'triangle', 'seasonal', 'adaptive', 'adaptive_chain', 'conditional', 'conditional_chain', 'context_residual'},
    'finalization_gap': {'ridge', 'seasonal', 'trend', 'proxy'},
    'finalization_scope': {'all', 'targets'},
    'finalization_statistic': {'mean', 'median'},
    'finalization_features': {'basic', 'momentum', 'age', 'age_momentum'},
    'finalization_gate': {'none', 'causal', 'admissions'},
    'replay_features': {'basic', 'momentum', 'age', 'age_momentum'},
    'replay_gate': {'none', 'causal', 'admissions'},
}


# One line per field for the generated key `docs/reference/scenario.md` (2026-09-22),
# taken from the existing design docs and code; where the meaning is longer than a
# line the entry points to the document that defines it.
MEANING = {
    'forecast_targets': 'All six targets or flu admissions + ED only; exported forecasts and scoring follow this scope.',
    'pathogen_inputs': 'Allowed admissions/ED input pathogens in both forecaster and nowcaster; excluded values and masks are removed.',
    'vintage_seasons': 'Archived seasons supplying artificial reporting errors and real correction pairs: latest permitted training season, or every archived training season with recency weights 1, 1/2, 1/4.',
    'actual_share': 'Probability a training episode uses the actual archived report (finalized fill where none) instead of an artificial-error draw.',
    'correction_realizations': 'Independent artificial-report-then-correction realizations per training episode; one is drawn per minibatch.',
    'uncorrected_share': 'Probability a corrected-history training episode is shown uncorrected instead (robustness to correction failure).',
    'nowcast_noise': 'Scale of sampled out-of-fold correction residuals added to corrected admissions; 0 = point correction. Evaluation uses 16 sampled histories per issuance.',
    'nowcast_noise_train': 'Also add sampled correction residuals to corrected training histories each minibatch.',
    'log_loss_weight': 'Weight of an extra weekly flu-admission loss on the log(1 + count) scale, added to the native-scale loss.',
    'covariate_dropout': 'Probability a training episode has all covariates (e.g. Kinsa) hidden.',
    'training_window': 'Training seasons per fold: all four seasons except the held-out one, or only the two seasons immediately before it.',
    'stress_views': 'Also evaluate delayed-admission and covariate-missing input views of the same fitted model.',
    'ili_units': 'Own-source ILI proportions or auxiliary flu-head pseudo-tasks rescaled by fitting-only modern Q95 / historical ILI Q95.',
    'ili_steps': 'Historical forecasting pretraining updates; each uses 64 auxiliary examples (series models) or batch_size historical panel episodes (flu MLP).',
    'correction_weeks': 'Recent admissions ages corrected by the pilot nowcaster, clipped to lookback.',
    'pilot_method': 'Exploratory forecaster treatment; all variants use the standard per-Hub scorer and raw/half/full admission-correction views.',
    'revision_scope': 'Apply artificial reporting errors in all training seasons, or early two with actual/proxy recent-season reports.',
    'revision_signals': 'Signals perturbed by empirical errors; admissions leaves ED and auxiliary covariates unchanged.',
    'pilot_nowcaster': 'Admissions-only synthetic/real-pair tree or matched real-pair neural corrector, optionally synthetic-pretrained.',
    'ili_training': 'No historical transfer, forecasting pretraining, or joint forecasting on the pinned pre-2022 ILI archive.',
    'ili_path': 'Historical ILI auxiliary dataset; its hash is pinned by the experiment manager.',
    'ili_weight': 'Historical ILI loss weight for joint training after within-history normalization.',
    'growth_anchor': 'Shared-series models: add damped observed two-week transformed growth to the level anchor.',
    'reporting_missingness': 'Transfer donor reporting masks with reporting augmentation; false transports numerical errors only and retains native input availability.',
    'reporting_augmentation': 'Stochastic training-season joint revision windows; vintage errors or causal nowcast residuals; real-vintage scoring.',
    'replay_from': 'Completed experiment supplying fixed CV checkpoints for inference-only input replay.',
    'replay_inputs': 'Fixed-checkpoint replay: finalized B2 inputs, full vintage history, or vintage plus eight-week seasonal target nowcasts.',
    'replay_flags': 'Diagnostic finality encoding: native semantics, available to reproduce B2 training encoding, or off. Available does not claim that revisions are final.',
    'finalization_model': 'Reported-value curve: triangle, seasonal/adaptive, growth-conditioned, or trajectory-trained residual calibration.',
    'finalization_growth': 'Conditional curve log-growth kernel bandwidth per week.',
    'finalization_penalty': 'Ridge penalty for online trajectory residual calibration.',
    'finalization_growth_weight': 'Relative weight of growth in residual training; evaluation is unchanged.',
    'finalization_residual_halflife': 'Residual-training example recency half-life in weeks.',
    'replay_growth_weight': 'Residual nowcaster training growth weight for replay.',
    'replay_residual_halflife': 'Residual nowcaster training recency for replay.',
    'replay_nowcaster': 'Fixed-checkpoint replay target nowcaster.',
    'replay_growth': 'Conditional replay growth bandwidth.',
    'replay_penalty': 'Residual replay ridge penalty.',
    'replay_schedule': 'Apply documented source schedule; absent target reports use explicit frozen-final proxies.',
    'replay_uncertainty': 'Scale of causal joint revision-error bootstrap in fixed-checkpoint nowcast replay; zero is deterministic.',
    'finalization_strength': 'Residual correction multiplier before native-unit rounding; 1 is full strength.',
    'replay_strength': 'Residual correction multiplier for matched forecast replay.',
    'finalization_features': 'Basic context/local effects, revision momentum, smooth age-dependent effects, or their combination.',
    'finalization_gate': 'Apply residual to every target, admissions only, or causally selected strength from mature past trajectory errors.',
    'replay_features': 'Residual feature family for exact nowcaster replay.',
    'replay_gate': 'Residual application rule for exact nowcaster replay.',
    'finalization_gap': 'Missing targets: ridge, historical seasonal growth, damped trend, or observable national proxy growth.',
    'finalization_scope': 'Fit all supported signals or only the six NHSN/NSSP targets.',
    'finalization_halflife': 'Adaptive curve recent-pair half-life in weeks.',
    'finalization_pool': 'Adaptive/seasonal local state pooling strength in effective weeks.',
    'finalization_statistic': 'Mean ratio of weighted totals or exposure-weighted median development ratio.',
    'finalization_quantize': 'Round NSSP point predictions and point comparators to the archived 0.0001 proportion grid.',
    'finalization_cv': 'finalize only: three recent four-Wednesday rolling holdouts, or forward season holdouts.',
    'finalization_loss': 'finalize only: triangle calibration minimizes normalized native-unit MAE, equally weighted by location.',
    'finalization_maturity': 'finalize only: minimum reference age and training-label gap, in weeks; not a guarantee of finality.',
    'finalization_weeks': 'finalize only: reconstruct this many completed weeks ending at each signal\'s T-X boundary.',
    'signal_features': 'Optional causal 3/6/12-week level, slope and curvature features for targets and covariates, with optional three-week smoothing.',
    'coordinates': 'Census state internal-point latitude/longitude and non-US indicator.',
    'evaluation_seasons': 'all three held-out seasons, recent_two (2025-26 and 2024-25), or production (fit on all four seasons 2022-23 to 2025-26; forecast 2026-27 inputs, unscored).',
    'training_inputs': 'same as forecasting, or complete finalized target and covariate histories during fitting only.',
    'input_normalization': 'none, or B0 per-location transformed target scales fitted on training contexts only.',
    'validation_calendar': 'season-relative blocks, or B0 blocks counted from the first stored week of each season.',
    'task': 'forecast, nowcast, an independently fitted nowcast-to-forecast pipeline, or per-signal boundary-week finalization.',
    'nowcast_weeks': 'Completed weeks reconstructed, ending at the context Saturday.',
    'nowcast': 'Pipeline stage overrides written as nowcast.<field>=value; unprefixed fields supply defaults.',
    'forecast': 'Pipeline stage overrides written as forecast.<field>=value; unprefixed fields supply defaults.',
    'nowcast_members': 'Cached out-of-fold history draws per training episode.',
    'lookback': 'Context weeks per episode (the history the network sees).',
    'count_transform': 'Admissions in model space: raw counts, rate per 100,000, or its sqrt / fourth root / log1p '
                       '(`model/network.py` `transform_counts`).',
    'ed_transform': 'ED proportions in model space (`model/network.py`); scores stay in native units.',
    'geography': 'Adds log population and a native-US flag per location as features.',
    'dynamics': 'Recent-dynamics feature block (30 slope/acceleration/age/validity features); see architecture.md.',
    'loss_weights': 'Training-loss weight per channel (`model/objective.py` `LOSS_WEIGHTS`); `objective` = the '
                    "score's target weights.",
    'encoder': 'Temporal context encoder; see architecture.md.',
    'spatial': 'Cross-location information exchange (none, shared attention, pathogen/target/joint scopes); '
               'see architecture.md.',
    'heads': 'State and US output heads shared or separate (`state_us`).',
    'sum_wis_weight': 'Auxiliary four-week flu admission sum WIS weight; sampled heads only; normalized by four times training-only admission Q95.',
    'decoder': 'Horizon decoder: existing modulated residual (`legacy`) or `residual2`; see architecture.md.',
    'noise': 'Global latent noise, or global plus a per-location latent (`local`).',
    'us_error': 'Extra common noise factor (`shared_factor`); see architecture.md.',
    'latent': 'Global latent (noise) dimension.',
    'width': 'Hidden width of the network.',
    'epochs': 'Epoch cap (the fixed number of epochs when patience = 0).',
    'patience': 'Early-stopping patience in epochs; 0 = fixed epochs, no validation weeks; > 0 selects the epoch '
                'on the validation weeks, then refits on the full training seasons (design §4).',
    'batch_size': 'Episodes per optimizer step.',
    'members': 'Sampled members per training episode (fair CRPS loss).',
    'lr': 'Adam learning rate.',
    'head_sharing': 'Output sharing: shared, three pathogen heads, or six target heads; see architecture.md.',
    'annual_calendar': 'Annual sine/cosine and Christmas-timing features.',
    'location_embedding': 'Dimension of a learned location-ID embedding (0 = none).',
    'fit_partition': 'One model for all six targets, or separately fitted models per pathogen / target group '
                     '(each sees all six inputs); see architecture.md.',
    'validation_members': 'Members drawn for the early-stopping validation loss.',
    'weight_decay': 'Adam weight decay.',
    'supplied_final': 'The network receives a known-final flag channel per cell; see architecture.md.',
    'mask_rate': 'Probability an episode receives an artificial missingness pattern in training (not a fraction '
                 'of cells); see architecture.md.',
    'mask_recent': 'Share of masked episodes whose pattern hides recent reports (with mask_gap, mask_outage sums to 1).',
    'mask_gap': 'Share of masked episodes whose pattern is a local gap in one location history.',
    'mask_outage': 'Share of masked episodes whose pattern is a whole-channel outage.',
    'covariate_encoder': 'Raw standardized history, signed-log trailing-three-week smoothing, six summaries, a shared 4-dimensional encoder, or recent growth (level, 1/2-week log growth, acceleration) plus coverage/age.',
    'covariate_set': "`+`-joined covariate source groups fed to the context encoder; '' = none; see "
                     'workflow.md and experiments/b-2-t0/index.md#protocol.',
    'input_mode': 'scheduled_final supplies T-0 final targets and source-specific T-0/T-1 covariates; finalized truth, finalized_available (final truth masked by Wednesday reporting availability), or Wednesday-vintage context '
                  '(design §3).',
    'asof_weeks': 'Standalone forecast only (nowcast/pipeline use all-as-of history): most recent context weeks whose targets are as visible at the issuance; '
                  'older weeks take final truth (design §3).',
    'validation_weeks': 'patience > 0 only: consecutive early-stopping weeks hidden per block (design §4).',
    'validation_spacing': 'patience > 0 only: one validation block every this many weeks of a training season.',
    'validation_offset': 'patience > 0 only: week of each training season where the first block starts '
                         "(0-based, counted from the season's first epiweek, CDC week 31).",
}


def _encode(name, value):
    if name in CODES:
        if value not in CODES[name]:
            raise ValueError(f'Invalid {name}: {value!r}')
        return str(value)
    if isinstance(value, bool):
        return '1' if value else '0'
    if isinstance(value, float):
        return repr(value)  # round-trips exactly, unlike `:g` (which loses precision past 6 sig figs)
    return str(value)


def _decode(name, raw, field_type):
    if name in CODES:
        if raw not in CODES[name]:
            raise ValueError(f'Invalid {name}: {raw!r}')
        return raw
    if field_type is bool:
        if raw not in ('0', '1'):
            raise ValueError(f'Invalid {name}: {raw!r}')
        return raw == '1'
    try:
        return field_type(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f'Invalid {name}: {raw!r}') from error


@dataclass(frozen=True)
class Scenario:
    # Network shape and training budget (formerly TrainingScenario).
    pilot_method: str = 'none'
    revision_scope: str = 'all'
    revision_signals: str = 'admissions'
    pilot_nowcaster: str = 'synthetic_tree'
    forecast_targets: str = 'all'
    pathogen_inputs: str = 'all'
    ili_units: str = 'own'
    ili_steps: int = 200
    correction_weeks: int = 8
    ili_training: str = 'none'
    ili_path: str = 'data/processed/historical_ili.npz'
    ili_weight: float = .25
    growth_anchor: bool = False
    task: str = 'forecast'
    weekend_family: str = 'none'
    reporting_probability: float = 1.
    reporting_random_strength: bool = False
    reporting_recent: int = 0
    joint_weight: float = 1.
    correction_penalty: float = 10.
    correction_strength: float = 1.
    correction_features: str = 'phase'
    reporting_augmentation: str = 'none'
    reporting_missingness: bool = True
    reporting_method: str = 'local_log'
    reporting_strength: float = 1.
    replay_from: str = ''
    replay_inputs: str = 'none'
    replay_flags: str = 'native'
    replay_nowcaster: str = 'selected'
    replay_features: str = 'basic'
    replay_gate: str = 'none'
    replay_growth_weight: float = 1.
    replay_residual_halflife: float = 26.
    replay_penalty: float = 10.
    replay_growth: float = .2
    replay_schedule: bool = False
    replay_uncertainty: float = 0.
    replay_strength: float = 1.
    finalization_cv: str = 'rolling'
    finalization_loss: str = 'mae'
    finalization_model: str = 'triangle'
    finalization_gap: str = 'ridge'
    finalization_scope: str = 'all'
    finalization_features: str = 'basic'
    finalization_gate: str = 'none'
    finalization_strength: float = 1.
    finalization_growth_weight: float = 1.
    finalization_residual_halflife: float = 26.
    finalization_penalty: float = 10.
    finalization_growth: float = .2
    finalization_halflife: float = 8.
    finalization_pool: float = 4.
    finalization_statistic: str = 'mean'
    finalization_quantize: bool = False
    finalization_maturity: int = 4
    finalization_weeks: int = 1
    nowcast_weeks: int = 2
    nowcast: tuple = ()
    forecast: tuple = ()
    nowcast_members: int = 16
    lookback: int = 12
    count_transform: str = 'fourth_root'
    ed_transform: str = 'linear'
    geography: bool = True
    coordinates: bool = False
    dynamics: bool = True
    loss_weights: str = 'objective'
    encoder: str = 'mlp'
    spatial: str = 'none'
    heads: str = 'shared'
    decoder: str = 'legacy'
    sum_wis_weight: float = 0.
    noise: str = 'global'
    us_error: str = 'none'
    latent: int = 16
    width: int = 64
    epochs: int = 50
    patience: int = 0
    batch_size: int = 8
    members: int = 128
    lr: float = .001
    head_sharing: str = 'shared'
    annual_calendar: bool = True
    location_embedding: int = 0
    fit_partition: str = 'all'
    validation_members: int = 256
    weight_decay: float = 0.
    # Vintage-aware direct pipeline (formerly B1Scenario). `supplied_final` gates
    # whether the network receives a known-final flag channel at all; it stays
    # meaningful independent of `input_mode` (an ablation can still turn it off
    # even when training in `input_mode=vintaged`).
    supplied_final: bool = False
    mask_rate: float = 0.
    mask_recent: float = .5
    mask_gap: float = .3
    mask_outage: float = .2
    # Covariates and dataset selection (formerly B2Scenario).
    covariate_encoder: str = 'raw'
    signal_features: str = 'none'
    covariate_set: str = ''
    input_mode: str = 'finalized'
    training_inputs: str = 'same'
    input_normalization: str = 'none'
    validation_calendar: str = 'season'
    evaluation_seasons: str = 'all'
    # Vintaged episodes only (2026-09-22): the number of most recent context weeks
    # whose targets are taken as visible at the issuance cutoff; older context weeks
    # take final truth. 2 reproduces the previous vintaged builder (B1); a value >=
    # lookback makes every context week as-of. Covariates are always as-of for
    # every context week of a vintaged episode (the previous builder's behaviour),
    # so one field serves both. See dataset/episodes.py.
    asof_weeks: int = 2
    # Cross-validation early stopping (user decision 2026-09-22: CV settings belong to
    # the scenario). Only used when patience > 0: hide `validation_weeks` consecutive
    # weeks of every `validation_spacing`, from week `validation_offset` of each
    # training season (defaults: 0-based weeks 4-6, 20-22, 36-38 counted from the
    # season's first epiweek, CDC week 31). See dataset/cv.py.
    validation_weeks: int = 3
    validation_spacing: int = 16
    validation_offset: int = 4
    # B5 / B4.polish options (2026-10-06); defaults reproduce B4 exactly.
    vintage_seasons: str = 'latest'
    actual_share: float = 0.
    correction_realizations: int = 1
    uncorrected_share: float = 0.
    nowcast_noise: float = 0.
    nowcast_noise_train: bool = False
    log_loss_weight: float = 0.
    covariate_dropout: float = 0.
    training_window: str = 'all'
    stress_views: bool = False

    def __post_init__(self):
        if not 0 <= self.sum_wis_weight <= 1:
            raise ValueError('Sum WIS weight must be between zero and one')
        if self.sum_wis_weight and (self.decoder not in ('legacy', 'residual2') or self.encoder in ('series_mlp', 'series_mixer') or self.forecast_targets != 'flu' or self.task != 'forecast'):
            raise ValueError('Sum WIS requires sampled flu forecasting')
        if self.forecast_targets == 'flu' and self.loss_weights != 'flu_hosp_ed':
            raise ValueError('Flu-only forecasts require flu admission + ED loss weights')
        if self.ili_units != 'own' and self.ili_training == 'none':
            raise ValueError('Scaled ILI requires ILI training')
        if self.ili_steps < 1 or self.correction_weeks < 1:
            raise ValueError('Invalid ILI or correction budget')
        if self.growth_anchor and self.encoder not in ('series_mlp', 'series_mixer', 'mlp'):
            raise ValueError('Growth anchor needs an MLP or shared series encoder')
        if not (0 <= self.actual_share <= 1 and 0 <= self.uncorrected_share <= 1 and 0 <= self.covariate_dropout <= 1
                and self.correction_realizations >= 1 and self.nowcast_noise >= 0 and self.log_loss_weight >= 0):
            raise ValueError('Invalid B5 vintage/nowcast/loss option')
        if (self.correction_realizations > 1 or self.uncorrected_share or self.nowcast_noise_train) and self.pilot_method not in ('corrected', 'two_stage'):
            raise ValueError('Correction realizations, uncorrected share and training noise need corrected training histories')
        if self.evaluation_seasons == 'production' and (self.training_window != 'all' or self.pilot_method == 'none'):
            raise ValueError('Production fits use every season and the pilot fitting path')
        if self.nowcast_noise_train and not self.nowcast_noise:
            raise ValueError('Training correction noise needs nowcast_noise > 0')
        if (self.actual_share or self.vintage_seasons != 'latest') and self.pilot_method in ('none', 'finalized', 'reported'):
            raise ValueError('Actual-report mixing and multi-season vintages need artificial training reports')
        if self.ili_weight < 0:
            raise ValueError('Historical source loss weight must be nonnegative')
        if self.decoder in ('quantile', 'quantile_small') and (self.noise != 'global' or self.us_error != 'none'):
            raise ValueError('Direct quantiles cannot add sampled output noise')
        if self.ili_training != 'none' and self.encoder not in ('series_mlp', 'series_mixer') and \
                (self.encoder != 'mlp' or self.ili_training != 'pretrain' or self.ili_units != 'flu_scaled' or self.forecast_targets != 'flu'):
            raise ValueError('Historical ILI needs a shared series encoder, or flu-scaled panel pretraining of a flu MLP')
        if self.encoder in ('series_mlp', 'series_mixer') and (self.fit_partition != 'all' or self.spatial != 'none' or self.covariate_set):
            raise ValueError('Pilot series models use one shared fit, no spatial messages or covariates')
        if self.pilot_method != 'none' and (self.task != 'forecast' or self.supplied_final or self.mask_rate):
            raise ValueError('Pilot requires forecast with no finality flag or artificial masking')
        if self.weekend_family != 'none' and (self.task != 'forecast' or self.input_mode != 'scheduled_final' or self.supplied_final or self.reporting_missingness or self.mask_rate):
            raise ValueError('Weekend experiments require scheduled-final forecast, no final flag, no added missingness or masking')
        if not 0 <= self.reporting_probability <= 1 or self.reporting_recent < 0 or self.joint_weight <= 0:
            raise ValueError('Invalid revision probability, recent window, or joint loss weight')
        if self.correction_penalty <= 0 or not 0 <= self.correction_strength <= 1 or self.correction_features not in ('age', 'phase', 'phase_local'):
            raise ValueError('Invalid nowcaster settings')
        if not 0 < self.reporting_strength <= 2:
            raise ValueError('reporting_strength must be in (0, 2]')
        for key in CODES:
            if getattr(self, key) not in CODES[key]:
                raise ValueError(f'Invalid {key}: {getattr(self, key)!r}')
        if self.reporting_augmentation != 'none' and (self.task != 'forecast' or self.input_mode != 'scheduled_final' or self.supplied_final or self.replay_from):
            raise ValueError('Reporting augmentation requires scheduled-final forecast training, no finality channel and no replay')
        if bool(self.replay_from) != (self.replay_inputs != 'none'):
            raise ValueError('Replay requires both replay_from and replay_inputs')
        if self.replay_uncertainty < 0 or (self.replay_uncertainty and self.replay_inputs != 'nowcast'):
            raise ValueError('Nonnegative uncertainty requires nowcast replay')
        if not 0 <= self.finalization_strength <= 1 or not 0 <= self.replay_strength <= 1:
            raise ValueError('Residual strength must be between zero and one')
        if min(self.replay_growth_weight, self.replay_residual_halflife, self.replay_growth, self.replay_penalty) <= 0:
            raise ValueError('replay_growth must be positive')
        if (self.replay_nowcaster != 'selected' or self.replay_schedule) and not self.replay_from:
            raise ValueError('Replay context options require replay_from')
        if self.replay_flags != 'native' and not self.replay_from:
            raise ValueError('Finality encoding controls require checkpoint replay')
        if self.replay_from and (self.task != 'forecast' or self.input_mode != 'scheduled_final'):
            raise ValueError('Input replay currently uses scheduled-final standalone forecast checkpoints')
        if self.training_inputs != 'same' and (self.task != 'forecast' or self.input_mode != 'vintaged'):
            raise ValueError('Complete target training requires standalone forecast with dated vintage evaluation')
        if self.validation_calendar == 'b0' and not self.patience:
            raise ValueError('B0 validation calendar requires early stopping')
        if min(self.lookback, self.latent, self.width, self.epochs, self.batch_size) < 1 or self.members < 2:
            raise ValueError('Positive dimensions/epochs required; training members >= 2')
        if self.lookback < 2 and self.task != 'finalize':
            raise ValueError('lookback must be at least two weeks')
        if self.finalization_maturity < 0:
            raise ValueError('finalization_maturity must be nonnegative')
        if self.finalization_model == 'context_residual' and self.finalization_weeks < 4:
            raise ValueError('Trajectory residual calibration needs at least four reconstructed weeks')
        if self.task=='finalize' and self.finalization_model=='context_residual' and self.finalization_scope!='targets':
            raise ValueError('Context residual calibration currently supports the six targets; use finalization_scope=targets')
        if self.finalization_weeks < 1:
            raise ValueError('finalization_weeks must be positive')
        if min(self.finalization_growth_weight, self.finalization_residual_halflife) <= 0 or self.finalization_penalty <= 0 or self.finalization_growth <= 0 or self.finalization_halflife <= 0 or self.finalization_pool < 0:
            raise ValueError('Finalization half-life must be positive and pooling nonnegative')
        if self.task == 'finalize' and self.patience:
            raise ValueError('Finalization uses fixed epochs; compare epoch budgets in separate scenarios')
        if self.validation_members < 2 or self.location_embedding < 0 or self.weight_decay < 0:
            raise ValueError('Validation members >=2, nonnegative embedding and weight decay required')
        if self.patience < 0 or (self.patience and self.patience >= self.epochs):
            raise ValueError('Patience must be 0 (fixed epochs) or below the epoch cap')
        if not (0 < self.lr < float('inf')):
            raise ValueError(f'lr must be finite and positive: {self.lr}')
        if self.spatial not in ('none', 'pooled', 'national_broadcast', 'gated_pool') and self.width % 4:
            raise ValueError('Spatial attention requires width divisible by four')
        if not 0 <= self.mask_rate <= 1:
            raise ValueError('mask_rate must be between zero and one')
        mix = (self.mask_recent, self.mask_gap, self.mask_outage)
        if any(p < 0 for p in mix) or abs(sum(mix) - 1) > 1e-9:
            raise ValueError('mask_recent/mask_gap/mask_outage must be nonnegative and sum to one')
        if self.nowcast_weeks < 1 or (self.task == 'nowcast' and self.nowcast_weeks > self.lookback):
            raise ValueError('nowcast_weeks must be between 1 and lookback')
        if self.nowcast_members < 2:
            raise ValueError('nowcast_members >= 2 required')
        if self.task != 'forecast' and self.input_mode != 'vintaged':
            raise ValueError('nowcast and pipeline require input_mode=vintaged')
        if self.task != 'pipeline' and (self.nowcast or self.forecast):
            raise ValueError('Stage overrides only apply to task=pipeline')
        if self.asof_weeks < 0:
            raise ValueError('asof_weeks must be nonnegative')
        if self.input_mode != 'vintaged' and self.asof_weeks != 2:
            raise ValueError('asof_weeks only applies to input_mode=vintaged')
        if not (1 <= self.validation_weeks and 0 <= self.validation_offset
                and self.validation_offset + self.validation_weeks <= self.validation_spacing):
            raise ValueError('Need 1 <= validation_weeks and 0 <= validation_offset, '
                             'validation_offset + validation_weeks <= validation_spacing')
        if not self.patience and (self.validation_weeks, self.validation_spacing, self.validation_offset) != (3, 16, 4):
            raise ValueError('validation_* settings only apply with patience > 0')
        from tapestry.dataset.build import SOURCE_GROUPS
        # Canonicalize so equivalent spellings ('a+b' vs 'b+a' vs 'a+a+b') collapse
        # to one string and one `run_id`, instead of silently training the same
        # configuration twice under different scenario strings.
        for field in ('covariate_set',):
            value = getattr(self, field)
            groups = value.split('+') if value else []
            unknown = set(groups) - set(SOURCE_GROUPS)
            if unknown:
                raise ValueError(f'Unknown covariate group(s): {sorted(unknown)}')
            canonical = '+'.join(group for group in SOURCE_GROUPS if group in groups)
            object.__setattr__(self, field, canonical)

        field_types = {f.name: f.type for f in fields(self)}
        for stage in ('nowcast', 'forecast'):
            overrides = dict(getattr(self, stage))
            unknown = set(overrides) - self.stage_fields()
            if unknown:
                raise ValueError(f'Invalid {stage} override fields: {sorted(unknown)}')
            # Copy to immutable, canonical key/value pairs, preserving frozen scenarios.
            overrides = {key: _decode(key, _encode(key, value), field_types[key])
                         for key, value in overrides.items()}
            object.__setattr__(self, stage, tuple(sorted(overrides.items())))
        if self.task == 'pipeline':
            for name in ('nowcast', 'forecast'):
                resolved = self.stage(name)
                object.__setattr__(self, name, tuple((key, getattr(resolved, key))
                                                    for key, _ in getattr(self, name)))

    @classmethod
    def stage_fields(cls):
        return {f.name for f in fields(cls)} - {
            'task', 'nowcast', 'forecast', 'nowcast_weeks', 'nowcast_members', 'input_mode', 'asof_weeks'}

    def stage(self, name):
        """Resolve independent stage settings; epochs/patience have identical meanings in both."""
        if self.task != 'pipeline' or name not in ('nowcast', 'forecast'):
            raise ValueError('Resolve nowcast or forecast only on a pipeline scenario')
        overrides = dict(getattr(self, name))
        return replace(self, task=name, nowcast=(), forecast=(),
                       asof_weeks=overrides.get('lookback', self.lookback), **overrides)

    def episode_scenario(self):
        """Operational input window and forecast labels for the coupled pipeline."""
        nowcast, forecast = self.stage('nowcast'), self.stage('forecast')
        lookback = max(nowcast.lookback, forecast.lookback)
        groups = '+'.join(filter(None, (nowcast.covariate_set, forecast.covariate_set)))
        return replace(forecast, lookback=lookback, asof_weeks=lookback, covariate_set=groups)

    @property
    def scored_seasons(self):
        if self.task == 'finalize' and self.finalization_cv == 'rolling':
            return ('rolling_1', 'rolling_2', 'rolling_3')
        if self.evaluation_seasons == 'production':
            return ('2026-2027',)
        return ('2025-2026', '2024-2025') if self.evaluation_seasons == 'recent_two' else ('2023-2024', '2024-2025', '2025-2026')

    @property
    def horizons(self):
        if self.weekend_family == 'joint' or self.pilot_method == 'joint':
            return tuple(range(1 - self.nowcast_weeks, 5))
        return tuple(range(1 - self.nowcast_weeks, 1)) if self.task == 'nowcast' else (1, 2, 3, 4)

    @property
    def scenario_string(self):
        tokens = [f'{f.name}={_encode(f.name, v)}' for f in fields(self)
                  if f.name not in ('nowcast', 'forecast') and (v := getattr(self, f.name)) != f.default]
        tokens.extend(f'{stage}.{key}={_encode(key, value)}'
                      for stage in ('nowcast', 'forecast') for key, value in getattr(self, stage))
        return ','.join(tokens)

    @classmethod
    def from_string(cls, value):
        """Any subset of `key=value` tokens, in any order; missing fields default."""
        if not value:
            return cls()
        field_types = {f.name: f.type for f in fields(cls)}
        options = {}
        for token in value.split(','):
            if '=' not in token:
                raise ValueError(f'Invalid scenario token: {token!r}')
            key, _, raw = token.partition('=')
            if '.' in key:
                stage, key = key.split('.', 1)
                if stage not in ('nowcast', 'forecast') or key not in cls.stage_fields():
                    raise ValueError(f'Unknown stage setting: {stage}.{key}')
                options.setdefault(stage, {})[key] = _decode(key, raw, field_types[key])
            else:
                if key not in field_types or key in ('nowcast', 'forecast'):
                    raise ValueError(f'Unknown scenario field: {key!r}')
                options[key] = _decode(key, raw, field_types[key])
        try:
            return cls(**options)
        except TypeError as error:
            raise ValueError(f'Invalid scenario string {value!r}: {error}') from error

    @property
    def run_id(self):
        """Filesystem-safe short name; the full string stays in manifests."""
        digest = hashlib.sha256(self.scenario_string.encode()).hexdigest()[:12]
        return f'{self.encoder}-{self.fit_partition}-{self.input_mode}-{digest}'

    @property
    def mask_probabilities(self):
        return (1 - self.mask_rate, self.mask_rate * self.mask_recent,
                self.mask_rate * self.mask_gap, self.mask_rate * self.mask_outage)

    def model_options(self):
        return dict(pathogen_inputs=self.pathogen_inputs, signal_features=self.signal_features, covariate_encoder=self.covariate_encoder, encoder=self.encoder, spatial=self.spatial, decoder=self.decoder, heads=self.heads,
                    noise=self.noise, head_sharing=self.head_sharing, count_transform=self.count_transform,
                    ed_transform=self.ed_transform, geography=self.geography, coordinates=self.coordinates, dynamics=self.dynamics,
                    annual_calendar=self.annual_calendar, location_embedding=self.location_embedding,
                    us_error=self.us_error, supplied_final=self.supplied_final, growth_anchor=self.growth_anchor)
