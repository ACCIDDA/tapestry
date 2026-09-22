"""One scenario dataclass for the unified model, with a self-describing string codec.

Replaces the old `TrainingScenario` -> `B1Scenario` -> `B2Scenario` subclass chain
and their hand-written positional string codecs (`b0:h12:tr_4rt:...`). A scenario
string here is `key=value` tokens joined by `,`, one token per field that differs
from `Scenario()`'s default, in any order:

    >>> Scenario(width=32).scenario_string
    'width=32'
    >>> Scenario.from_string('width=32') == Scenario(width=32)
    True

Consequences, all intentional (see docs/design/restructure-2026-unified.md §1):
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
from dataclasses import dataclass, fields
import hashlib

# Enum-like string fields: a fixed set of legal values, so a typo raises instead
# of silently becoming a new field the way an unconstrained string would.
CODES = {
    'count_transform': {'raw', 'rate', 'sqrt', 'fourth_root', 'log1p'},
    'ed_transform': {'linear', 'logit', 'fourth_root'},
    'loss_weights': {'influenza_first', 'balanced_admissions', 'flu_only', 'objective'},
    'encoder': {'mlp', 'conv', 'multiscale_conv'},
    'spatial': {'none', 'attention', 'pathogen_spatial', 'target_spatial', 'joint_location_target'},
    'heads': {'shared', 'state_us'},
    'decoder': {'legacy', 'residual2'},
    'noise': {'global', 'local'},
    'us_error': {'none', 'shared_factor'},
    'head_sharing': {'shared', 'pathogen', 'target'},
    'fit_partition': {'all', 'pathogen', 'target'},
    'input_mode': {'finalized', 'vintaged'},
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
    lookback: int = 12
    count_transform: str = 'fourth_root'
    ed_transform: str = 'linear'
    geography: bool = True
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
    covariate_set: str = ''
    input_mode: str = 'finalized'

    def __post_init__(self):
        for key in CODES:
            if getattr(self, key) not in CODES[key]:
                raise ValueError(f'Invalid {key}: {getattr(self, key)!r}')
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
        if self.spatial != 'none' and self.width % 4:
            raise ValueError('Spatial attention requires width divisible by four')
        if not 0 <= self.mask_rate <= 1:
            raise ValueError('mask_rate must be between zero and one')
        mix = (self.mask_recent, self.mask_gap, self.mask_outage)
        if any(p < 0 for p in mix) or abs(sum(mix) - 1) > 1e-9:
            raise ValueError('mask_recent/mask_gap/mask_outage must be nonnegative and sum to one')
        from tapestry.dataset.build import SOURCE_GROUPS
        # Canonicalize so equivalent spellings ('a+b' vs 'b+a' vs 'a+a+b') collapse
        # to one string and one `run_id`, instead of silently training the same
        # configuration twice under different scenario strings.
        groups = self.covariate_set.split('+') if self.covariate_set else []
        unknown = set(groups) - set(SOURCE_GROUPS)
        if unknown:
            raise ValueError(f'Unknown covariate group(s): {sorted(unknown)}')
        canonical = '+'.join(group for group in SOURCE_GROUPS if group in groups)
        if canonical != self.covariate_set:
            object.__setattr__(self, 'covariate_set', canonical)

    @property
    def scenario_string(self):
        return ','.join(f'{f.name}={_encode(f.name, v)}' for f in fields(self) if (v := getattr(self, f.name)) != f.default)

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

    @property
    def mask_probabilities(self):
        return (1 - self.mask_rate, self.mask_rate * self.mask_recent,
                self.mask_rate * self.mask_gap, self.mask_rate * self.mask_outage)

    def model_options(self):
        return dict(encoder=self.encoder, spatial=self.spatial, decoder=self.decoder, heads=self.heads,
                    noise=self.noise, head_sharing=self.head_sharing, count_transform=self.count_transform,
                    ed_transform=self.ed_transform, geography=self.geography, dynamics=self.dynamics,
                    annual_calendar=self.annual_calendar, location_embedding=self.location_embedding,
                    us_error=self.us_error, supplied_final=self.supplied_final)

    def flags(self):
        return ['--scenario', self.scenario_string]
