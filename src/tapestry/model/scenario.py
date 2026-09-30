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
    'training_inputs': {'same', 'finalized'},
    'input_normalization': {'none', 'b0'},
    'validation_calendar': {'season', 'b0'},
    'evaluation_seasons': {'all', 'recent_two'},
    'count_transform': {'raw', 'rate', 'sqrt', 'fourth_root', 'log1p'},
    'ed_transform': {'linear', 'logit', 'fourth_root'},
    'loss_weights': {'influenza_first', 'balanced_admissions', 'flu_only', 'objective'},
    'signal_features': {'none', 'multiscale', 'smooth_multiscale'},
    'covariate_encoder': {'raw', 'smooth', 'summary', 'shared'},
    'encoder': {'mlp', 'conv', 'multiscale_conv'},
    'spatial': {'neighbors', 'distance', 'gravity', 'none', 'pooled', 'attention', 'pathogen_spatial', 'target_spatial', 'joint_location_target', 'national_broadcast', 'gated_pool'},
    'heads': {'shared', 'state_us'},
    'decoder': {'legacy', 'residual2'},
    'noise': {'global', 'local'},
    'us_error': {'none', 'shared_factor'},
    'head_sharing': {'shared', 'pathogen', 'target'},
    'fit_partition': {'all', 'pathogen', 'target'},
    'input_mode': {'finalized', 'finalized_available', 'scheduled_final', 'vintaged'},
    'task': {'forecast', 'nowcast', 'pipeline'},
}


# One line per field for the generated key `docs/reference/scenario.md` (2026-09-22),
# taken from the existing design docs and code; where the meaning is longer than a
# line the entry points to the document that defines it.
MEANING = {
    'signal_features': 'Optional causal 3/6/12-week level, slope and curvature features for targets and covariates, with optional three-week smoothing.',
    'coordinates': 'Census state internal-point latitude/longitude and non-US indicator.',
    'evaluation_seasons': 'all three held-out seasons, or recent_two (2025-26 and 2024-25).',
    'training_inputs': 'same as forecasting, or complete finalized target and covariate histories during fitting only.',
    'input_normalization': 'none, or B0 per-location transformed target scales fitted on training contexts only.',
    'validation_calendar': 'season-relative blocks, or B0 blocks counted from the first stored week of each season.',
    'task': 'forecast, nowcast, or an independently fitted nowcast-to-forecast pipeline.',
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
    'covariate_encoder': 'Raw standardized history, signed-log trailing-three-week smoothing, six summaries, or a shared 4-dimensional encoder plus coverage/age.',
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
    task: str = 'forecast'
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

    def __post_init__(self):
        for key in CODES:
            if getattr(self, key) not in CODES[key]:
                raise ValueError(f'Invalid {key}: {getattr(self, key)!r}')
        if self.training_inputs != 'same' and (self.task != 'forecast' or self.input_mode != 'vintaged'):
            raise ValueError('Complete target training requires standalone forecast with dated vintage evaluation')
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
        return ('2025-2026', '2024-2025') if self.evaluation_seasons == 'recent_two' else ('2023-2024', '2024-2025', '2025-2026')

    @property
    def horizons(self):
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
        return dict(signal_features=self.signal_features, covariate_encoder=self.covariate_encoder, encoder=self.encoder, spatial=self.spatial, decoder=self.decoder, heads=self.heads,
                    noise=self.noise, head_sharing=self.head_sharing, count_transform=self.count_transform,
                    ed_transform=self.ed_transform, geography=self.geography, coordinates=self.coordinates, dynamics=self.dynamics,
                    annual_calendar=self.annual_calendar, location_embedding=self.location_embedding,
                    us_error=self.us_error, supplied_final=self.supplied_final)
