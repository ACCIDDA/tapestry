"""One scenario dataclass for the forecasting model, with a self-describing string codec.

A scenario string is `key=value` tokens joined by `,`, one token per field that differs
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

History. 2026-10-08: every submitted model (System2, B7) and every experiment since B3
used one training route, the former "pilot" route (`experiment/fit.py`); the other routes and
their fields were deleted. The same day and on 2026-10-09 (review) the confusing names were
replaced: `pilot_method` (one field bundling three choices) became `history_source`,
`history_correction` and `reconstruction_labels`; `pilot_nowcaster` became
`corrector_examples` + `corrector_model`; `evaluation_vintaging` -> `evaluation_inputs`;
`revision_*`/`vintage_seasons` -> `error_*`; `evaluation_replicates` -> `evaluation_draws`;
`nowcast_weeks` -> `reconstruction_weeks`; `nowcast_noise*` -> `correction_noise*`; and the
forecast-time view became a recipe field, `forecast_view`. On 2026-10-09 the stored strings of
the B7 experiments (folds, production fits, operational forecasts) and of the System2
production fits were rewritten to these names and the translation code was removed (user
decision): strings with old names now raise "Unknown scenario field". Older experiments
are history, kept as reports. The model choices A-F are explained in
docs/reference/model-choices.md.
"""
from dataclasses import dataclass, fields
import hashlib

# Enum-like string fields: a fixed set of legal values, so a typo raises instead
# of silently becoming a new field the way an unconstrained string would.
CODES = {
    'reporting_method': {'local_log', 'calendar_log', 'local_additive', 'synchronous_log', 'phase_log', 'synchronous_phase_log'},
    'input_normalization': {'none', 'b0'},
    'validation_calendar': {'season', 'b0'},
    'evaluation_seasons': {'all', 'recent_two', 'production'},
    'count_transform': {'raw', 'rate', 'sqrt', 'fourth_root', 'log1p'},
    'ed_transform': {'linear', 'logit', 'fourth_root'},
    'loss_weights': {'influenza_first', 'balanced_admissions', 'flu_only', 'flu_ed', 'flu_hosp_ed', 'objective'},
    'signal_features': {'none', 'multiscale', 'smooth_multiscale'},
    'covariate_encoder': {'raw', 'smooth', 'summary', 'shared', 'growth'},
    'encoder': {'mlp', 'conv', 'multiscale_conv', 'series_mlp', 'series_mixer'},
    'history_source': {'finalized', 'reported', 'artificial'},
    'forecast_view': {'raw', 'corrected', 'half'},
    'error_scope': {'all', 'early_actual'},
    'error_signals': {'all', 'admissions', 'targets'},
    'corrector_examples': {'synthetic', 'real', 'synthetic_then_real'},
    'corrector_model': {'tree', 'mlp'},
    'evaluation_inputs': {'reported', 'prescribed'},
    'ili_training': {'none', 'pretrain', 'joint'},
    'error_seasons': {'latest', 'all'},
    'training_window': {'all', 'recent2', 'last2'},
    'ili_units': {'own', 'flu_scaled'},
    'forecast_targets': {'all', 'flu'},
    'pathogen_inputs': {'all', 'flu', 'flu_hosp', 'flu_ed', 'flu_covid', 'flu_rsv'},
    'spatial': {'neighbors', 'distance', 'gravity', 'none', 'pooled', 'attention', 'pathogen_spatial', 'target_spatial', 'joint_location_target', 'national_broadcast', 'gated_pool'},
    'heads': {'shared', 'state_us'},
    'decoder': {'legacy', 'residual2', 'quantile', 'quantile_small'},
    'noise': {'global', 'local'},
    'us_error': {'none', 'shared_factor'},
    'head_sharing': {'shared', 'pathogen', 'target'},
    'fit_partition': {'all', 'pathogen', 'target'},
}

# One line per field for the generated key `docs/reference/scenario.md`,
# taken from the design docs and code; where the meaning is longer than a
# line the entry points to the document that defines it.
MEANING = {
    'history_source': 'Choice A1, where the network\'s training histories come from: final reference histories '
                      '(finalized), actual archived Wednesday reports (reported), or final histories made preliminary '
                      'with artificial reporting errors (artificial; choices B and C). Was part of pilot_method.',
    'history_correction': 'Choice A2: correct the artificial training histories with the cross-fitted correction model '
                          '(choice D) before the network learns from them (was pilot_method=two_stage/corrected).',
    'reconstruction_labels': 'Choice A3: also learn the final values of the last reconstruction_weeks context weeks '
                             '(was pilot_method=joint). Labels, not inputs.',
    'forecast_view': 'Choice F, how the fitted pair is used at forecast time, part of the recipe: inputs as given (raw), '
                     'newest weeks corrected by the fold\'s correction model (corrected), or the 50/50 mixture of those two '
                     'forecast distributions (half). Ranking and production use this view; the others are diagnostics.',
    'corrector_examples': 'Choice D, what the correction trees (the nowcaster) learn from: artificial errors paired with '
                          'final values (synthetic), real archived reports paired with their mature values (real), or '
                          'synthetic pretraining then real (synthetic_then_real, MLP only). Was pilot_nowcaster.',
    'corrector_model': 'Correction model: gradient-boosted trees (tree) or a residual MLP (mlp). Was pilot_nowcaster.',
    'evaluation_inputs': 'Choice E: evaluate on real archived Wednesday reports (reported) or on final histories made '
                         'preliminary by the prescribed error process of error_reference (prescribed). Was evaluation_vintaging.',
    'forecast_targets': 'All six targets or flu admissions + ED only; exported forecasts and scoring follow this scope.',
    'pathogen_inputs': 'Allowed admissions/ED input pathogens in both forecaster and nowcaster; excluded values and masks are removed.',
    'error_seasons': 'Choice B: archived seasons supplying artificial reporting errors and real correction pairs: latest permitted training season, or every archived training season with recency weights 1, 1/2, 1/4.',
    'actual_share': 'Probability a training episode uses the actual archived report (finalized fill where none) instead of an artificial-error draw.',
    'correction_realizations': 'Independent artificial-report-then-correction realizations per training episode; one is drawn per minibatch.',
    'uncorrected_share': 'Probability a corrected-history training episode is shown uncorrected instead (robustness to correction failure).',
    'correction_noise': 'Scale of sampled out-of-fold correction residuals added to corrected admissions; 0 = point correction. Evaluation uses 16 sampled histories per issuance.',
    'correction_noise_train': 'Also add sampled correction residuals to corrected training histories each minibatch.',
    'reconstruction_weeks': 'joint only: recent context weeks whose final values are reconstruction labels.',
    'log_loss_weight': 'Weight of an extra weekly flu-admission loss on the log(1 + count) scale, added to the native-scale loss.',
    'covariate_dropout': 'Probability a training episode has all covariates (e.g. Kinsa) hidden.',
    'training_window': 'all = every non-held-out season; recent2 = two preceding seasons; last2 = latest two permitted non-held-out seasons (nested removal of oldest training season).',
    'stress_views': 'Also evaluate delayed-admission and covariate-missing input views of the same fitted model.',
    'ili_units': 'Own-source ILI proportions or auxiliary flu-head pseudo-tasks rescaled by fitting-only modern Q95 / historical ILI Q95.',
    'ili_steps': 'Historical forecasting pretraining updates; each uses 64 auxiliary examples (series models) or batch_size historical panel episodes (flu MLP).',
    'correction_weeks': 'Recent admissions ages corrected by the nowcaster, clipped to lookback.',
    'correction_ed': 'Also fit/apply report-to-mature ED correction on the recent correction_weeks, restricted to allowed input channels; future labels unchanged.',
    'correction_penalty': 'Ridge penalty of the correction trees\' linear leaves.',
    'correction_strength': 'Multiplier of the fitted correction, 1 = full.',
    'error_scope': 'Apply artificial reporting errors in all training seasons, or early two with actual/proxy recent-season reports.',
    'error_signals': 'Choice C: admissions perturbs counts only; targets also perturbs ED but keeps covariates unchanged; all perturbs both targets and covariates.',
    'error_reference': 'Choice B: prescribed reporting-error season for training errors and prescribed evaluation '
                       'inputs, independent of the held-out epidemic season; empty = the fold\'s own seasons.',
    'evaluation_draws': 'Independent artificial evaluation histories (1, 3 or 5) when evaluation_inputs=prescribed.',
    'ili_training': 'No historical transfer, forecasting pretraining, or joint forecasting on the pinned pre-2022 ILI archive.',
    'ili_path': 'Historical ILI auxiliary dataset; its hash is pinned by the experiment manager.',
    'ili_weight': 'Historical ILI loss weight for joint training after within-history normalization.',
    'growth_anchor': 'Add damped observed two-week transformed growth to the level anchor.',
    'reporting_method': 'Donor matching of artificial reporting errors; see dataset/reporting_error.py.',
    'reporting_probability': 'Probability a training episode receives artificial reporting errors.',
    'reporting_random_strength': 'Scale each drawn error window by a uniform(0, 1) multiplier.',
    'reporting_recent': 'Perturb only this many most recent context weeks; 0 = all.',
    'reporting_strength': 'Multiplier of drawn log reporting errors.',
    'reporting_missingness': 'Transfer donor reporting masks with the errors; false transports numerical errors only and retains native input availability.',
    'joint_weight': 'joint only: weight of the reconstruction loss relative to the forecast loss.',
    'signal_features': 'Optional causal 3/6/12-week level, slope and curvature features for targets and covariates, with optional three-week smoothing.',
    'coordinates': 'Census state internal-point latitude/longitude and non-US indicator.',
    'evaluation_seasons': 'all three held-out seasons, recent_two (2025-26 and 2024-25), or production (fit on all four seasons 2022-23 to 2025-26; forecast 2026-27 inputs, unscored).',
    'input_normalization': 'none, or B0 per-location transformed target scales fitted on training contexts only.',
    'validation_calendar': 'season-relative blocks, or B0 blocks counted from the first stored week of each season.',
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
    'decoder': 'Horizon decoder: existing modulated residual (`legacy`), `residual2`, or direct ordered quantiles; see architecture.md.',
    'decoder_blocks': 'Residual decoder depth (0 = original head).',
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
    'covariate_encoder': 'Raw standardized history, signed-log trailing-three-week smoothing, six summaries, a shared 4-dimensional encoder, or recent growth (level, 1/2-week log growth, acceleration) plus coverage/age.',
    'covariate_set': "`+`-joined covariate source groups fed to the context encoder; '' = none; see "
                     'workflow.md and experiments/b-2-t0/index.md#protocol.',
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
    # Training histories and their reporting errors / corrections.
    history_source: str = 'finalized'
    history_correction: bool = False
    reconstruction_labels: bool = False
    error_scope: str = 'all'
    error_signals: str = 'admissions'
    corrector_examples: str = 'synthetic'
    corrector_model: str = 'tree'
    forecast_targets: str = 'all'
    pathogen_inputs: str = 'all'
    ili_units: str = 'own'
    ili_steps: int = 200
    correction_weeks: int = 8
    correction_ed: bool = False
    ili_training: str = 'none'
    ili_path: str = 'data/processed/historical_ili.npz'
    ili_weight: float = .25
    growth_anchor: bool = False
    reporting_probability: float = 1.
    reporting_random_strength: bool = False
    reporting_recent: int = 0
    joint_weight: float = 1.
    correction_penalty: float = 10.
    correction_strength: float = 1.
    reporting_missingness: bool = True
    reporting_method: str = 'local_log'
    reporting_strength: float = 1.
    reconstruction_weeks: int = 2
    # Network shape and training budget.
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
    decoder_blocks: int = 0
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
    # Covariates and evaluation.
    covariate_encoder: str = 'raw'
    signal_features: str = 'none'
    covariate_set: str = ''
    input_normalization: str = 'none'
    validation_calendar: str = 'season'
    evaluation_seasons: str = 'all'
    # Cross-validation early stopping (user decision 2026-09-22: CV settings belong to
    # the scenario). Only used when patience > 0: hide `validation_weeks` consecutive
    # weeks of every `validation_spacing`, from week `validation_offset` of each
    # training season (defaults: 0-based weeks 4-6, 20-22, 36-38 counted from the
    # season's first epiweek, CDC week 31). See dataset/cv.py.
    validation_weeks: int = 3
    validation_spacing: int = 16
    validation_offset: int = 4
    # B5 / B4.polish options (2026-10-06); defaults reproduce B4 exactly.
    error_seasons: str = 'latest'
    actual_share: float = 0.
    correction_realizations: int = 1
    uncorrected_share: float = 0.
    correction_noise: float = 0.
    correction_noise_train: bool = False
    log_loss_weight: float = 0.
    covariate_dropout: float = 0.
    training_window: str = 'all'
    stress_views: bool = False
    # A prescribed empirical reporting process, shared by every held-out season.
    error_reference: str = ''
    evaluation_inputs: str = 'reported'
    evaluation_draws: int = 1
    forecast_view: str = 'corrected'

    def __post_init__(self):
        for key in CODES:
            if getattr(self, key) not in CODES[key]:
                raise ValueError(f'Invalid {key}: {getattr(self, key)!r}')
        if self.evaluation_draws not in (1, 3, 5):
            raise ValueError('evaluation_draws must be 1, 3 or 5')
        if self.error_reference and self.error_reference not in ('2023-2024','2024-2025','2025-2026'):
            raise ValueError('error_reference must name a completed reporting season')
        if self.corrector_examples == 'synthetic_then_real' and self.corrector_model != 'mlp':
            raise ValueError('Synthetic pretraining then real examples needs corrector_model=mlp')
        if self.evaluation_inputs == 'prescribed' and not self.error_reference:
            raise ValueError('Artificial evaluation histories require a revision reference')
        if self.decoder_blocks not in (0, 2, 3, 4):
            raise ValueError('decoder_blocks must be 0 (original head), 2, 3 or 4')
        if self.decoder_blocks and (self.encoder != 'mlp' or self.decoder not in ('legacy', 'residual2', 'quantile')):
            raise ValueError('Residual decoder depth requires an MLP sampled or ordered-quantile head')
        if not 0 <= self.sum_wis_weight <= 1:
            raise ValueError('Sum WIS weight must be between zero and one')
        if self.sum_wis_weight and (self.decoder not in ('legacy', 'residual2') or self.encoder in ('series_mlp', 'series_mixer') or self.forecast_targets != 'flu'):
            raise ValueError('Sum WIS requires sampled flu forecasting')
        if self.forecast_targets == 'flu' and self.loss_weights not in ('flu_hosp_ed', 'flu_only', 'flu_ed'):
            raise ValueError('Flu forecasts require flu admissions, ED, or both loss weights')
        if self.ili_units != 'own' and self.ili_training == 'none':
            raise ValueError('Scaled ILI requires ILI training')
        if self.ili_steps < 1 or self.correction_weeks < 1:
            raise ValueError('Invalid ILI or correction budget')
        if self.growth_anchor and self.encoder not in ('series_mlp', 'series_mixer', 'mlp'):
            raise ValueError('Growth anchor needs an MLP or shared series encoder')
        if not (0 <= self.actual_share <= 1 and 0 <= self.uncorrected_share <= 1 and 0 <= self.covariate_dropout <= 1
                and self.correction_realizations >= 1 and self.correction_noise >= 0 and self.log_loss_weight >= 0):
            raise ValueError('Invalid B5 vintage/nowcast/loss option')
        if self.history_correction and self.history_source != 'artificial':
            raise ValueError('history_correction corrects artificial training histories (history_source=artificial)')
        if self.reconstruction_labels and (self.history_source != 'artificial' or self.history_correction):
            raise ValueError('reconstruction_labels is implemented with uncorrected artificial histories only')
        if (self.correction_realizations > 1 or self.uncorrected_share or self.correction_noise_train) and not self.history_correction:
            raise ValueError('Correction realizations, uncorrected share and training noise need corrected training histories')
        if self.evaluation_seasons == 'production' and self.training_window != 'all':
            raise ValueError('Production fits use every season')
        if self.correction_noise_train and not self.correction_noise:
            raise ValueError('Training correction noise needs correction_noise > 0')
        if (self.actual_share or self.error_seasons != 'latest') and self.history_source != 'artificial':
            raise ValueError('Actual-report mixing and multi-season vintages need artificial training reports')
        if self.ili_weight < 0:
            raise ValueError('Historical source loss weight must be nonnegative')
        if self.decoder in ('quantile', 'quantile_small') and (self.noise != 'global' or self.us_error != 'none'):
            raise ValueError('Direct quantiles cannot add sampled output noise')
        if self.ili_training != 'none' and self.encoder not in ('series_mlp', 'series_mixer') and \
                (self.encoder != 'mlp' or self.ili_training != 'pretrain' or self.ili_units != 'flu_scaled' or self.forecast_targets != 'flu'):
            raise ValueError('Historical ILI needs a shared series encoder, or flu-scaled panel pretraining of a flu MLP')
        if self.encoder in ('series_mlp', 'series_mixer') and (self.fit_partition != 'all' or self.spatial != 'none' or self.covariate_set):
            raise ValueError('Series models use one shared fit, no spatial messages or covariates')
        if not 0 <= self.reporting_probability <= 1 or self.reporting_recent < 0 or self.joint_weight <= 0:
            raise ValueError('Invalid revision probability, recent window, or joint loss weight')
        if self.correction_penalty <= 0 or not 0 <= self.correction_strength <= 1:
            raise ValueError('Invalid nowcaster settings')
        if not 0 < self.reporting_strength <= 2:
            raise ValueError('reporting_strength must be in (0, 2]')
        if self.validation_calendar == 'b0' and not self.patience:
            raise ValueError('B0 validation calendar requires early stopping')
        if min(self.lookback, self.latent, self.width, self.epochs, self.batch_size) < 1 or self.members < 2:
            raise ValueError('Positive dimensions/epochs required; training members >= 2')
        if self.lookback < 2:
            raise ValueError('lookback must be at least two weeks')
        if self.validation_members < 2 or self.location_embedding < 0 or self.weight_decay < 0:
            raise ValueError('Validation members >=2, nonnegative embedding and weight decay required')
        if self.patience < 0 or (self.patience and self.patience >= self.epochs):
            raise ValueError('Patience must be 0 (fixed epochs) or below the epoch cap')
        if not (0 < self.lr < float('inf')):
            raise ValueError(f'lr must be finite and positive: {self.lr}')
        if self.spatial not in ('none', 'pooled', 'national_broadcast', 'gated_pool') and self.width % 4:
            raise ValueError('Spatial attention requires width divisible by four')
        if self.reconstruction_weeks < 1 or self.reconstruction_weeks > self.lookback:
            raise ValueError('reconstruction_weeks must be between 1 and lookback')
        if not (1 <= self.validation_weeks and 0 <= self.validation_offset
                and self.validation_offset + self.validation_weeks <= self.validation_spacing):
            raise ValueError('Need 1 <= validation_weeks and 0 <= validation_offset, '
                             'validation_offset + validation_weeks <= validation_spacing')
        if not self.patience and (self.validation_weeks, self.validation_spacing, self.validation_offset) != (3, 16, 4):
            raise ValueError('validation_* settings only apply with patience > 0')
        from chromantis.dataset.build import COVARIATE_SELECTORS
        # Canonicalize so equivalent spellings ('a+b' vs 'b+a' vs 'a+a+b') collapse
        # to one string and one `run_id`, instead of silently training the same
        # configuration twice under different scenario strings.
        groups = self.covariate_set.split('+') if self.covariate_set else []
        unknown = set(groups) - set(COVARIATE_SELECTORS)
        if unknown:
            raise ValueError(f'Unknown covariate group(s): {sorted(unknown)}')
        object.__setattr__(self, 'covariate_set', '+'.join(group for group in COVARIATE_SELECTORS if group in groups))

    @property
    def input_mode(self):
        """Episode inputs for fitting (`dataset.episodes`): actual Wednesday reports for
        `history_source=reported`, otherwise finalized targets on the documented T-0/T-1 schedule."""
        return 'reported' if self.history_source == 'reported' else 'scheduled_final'

    @property
    def scored_seasons(self):
        if self.evaluation_seasons == 'production':
            return ('2026-2027',)
        return ('2025-2026', '2024-2025') if self.evaluation_seasons == 'recent_two' else ('2023-2024', '2024-2025', '2025-2026')

    @property
    def horizons(self):
        """Model output weeks: 1-4, preceded by `reconstruction_weeks` weeks for `joint`."""
        if self.reconstruction_labels:
            return tuple(range(1 - self.reconstruction_weeks, 5))
        return (1, 2, 3, 4)

    @property
    def future(self):
        """Indices of the forecast weeks (horizon > 0) within `horizons`."""
        return [i for i, h in enumerate(self.horizons) if h > 0]

    @property
    def scenario_string(self):
        return ','.join(f'{f.name}={_encode(f.name, v)}' for f in fields(self)
                        if (v := getattr(self, f.name)) != f.default)

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
            if key not in field_types:
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

    def model_options(self):
        return dict(pathogen_inputs=self.pathogen_inputs, signal_features=self.signal_features, covariate_encoder=self.covariate_encoder, encoder=self.encoder, spatial=self.spatial, decoder=self.decoder, decoder_blocks=self.decoder_blocks, heads=self.heads,
                    noise=self.noise, head_sharing=self.head_sharing, count_transform=self.count_transform,
                    ed_transform=self.ed_transform, geography=self.geography, coordinates=self.coordinates, dynamics=self.dynamics,
                    annual_calendar=self.annual_calendar, location_embedding=self.location_embedding,
                    us_error=self.us_error, growth_anchor=self.growth_anchor)
