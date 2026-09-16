"""Influpaint-style immutable scenarios, short readable strings, and scenario suites.

Adapted from influpaint/influpaint/batch/scenarios.py (TrainingScenario,
scenario_string, essential set). Stable names replace position-based numeric IDs.
"""
from dataclasses import asdict, dataclass, fields, replace

from .experiments import COUNT_TRANSFORMS, ED_TRANSFORMS, LOSS_WEIGHTS

# Field order fixes the string order. Categorical values use short codes.
PREFIX = dict(lookback='h', count_transform='tr_', ed_transform='ed_', geography='geo', dynamics='dyn',
              loss_weights='lw_', encoder='enc_', spatial='sp_', heads='hd_', decoder='dec_', noise='nz_',
              us_error='us_', latent='z', width='w', epochs='ep', patience='pat', batch_size='bs', members='m',
              lr='lr', head_sharing='hs_', annual_calendar='cal', location_embedding='id',
              fit_partition='fit_', validation_members='vm', weight_decay='wd')
CODES = dict(count_transform={'raw': 'raw', 'rate': 'rate', 'sqrt': 'sqrt', 'fourth_root': '4rt', 'log1p': 'log1p'},
             ed_transform={'linear': 'lin', 'logit': 'logit', 'fourth_root': '4rt'},
             loss_weights={'influenza_first': 'first', 'balanced_admissions': 'bal', 'flu_only': 'fluonly',
                           'objective': 'obj'},
             encoder={'mlp': 'mlp', 'conv': 'conv', 'multiscale_conv': 'msc'},
             spatial={'none': 'none', 'attention': 'attn', 'pathogen_spatial': 'path',
                      'target_spatial': 'targ', 'joint_location_target': 'joint'},
             heads={'shared': 'sh', 'state_us': 'su'}, decoder={'legacy': 'leg', 'residual2': 'res2'},
             noise={'global': 'glob', 'local': 'loc'},
             us_error={'none': 'none', 'shared_factor': 'shf'},
             head_sharing={'shared': 'sh', 'pathogen': 'path', 'target': 'targ'},
             fit_partition={'all': 'all', 'pathogen': 'path', 'target': 'targ'})
assert set(CODES['loss_weights']) == set(LOSS_WEIGHTS)
assert set(CODES['count_transform']) == set(COUNT_TRANSFORMS) and set(CODES['ed_transform']) == set(ED_TRANSFORMS)


@dataclass(frozen=True)
class TrainingScenario:
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
    # Training draws per episode. Fair CRPS is already unbiased at 8 (the m(m-1)
    # Ferro divisor), so this is variance reduction, not a bias fix: the per-batch
    # gradient estimate falls as 1/sqrt(m). Evaluation dominates a fold at ~21s
    # against a few seconds of fitting, so the extra draws are close to free.
    members: int = 128
    lr: float = .001
    head_sharing: str = 'shared'
    annual_calendar: bool = True
    location_embedding: int = 0
    fit_partition: str = 'all'
    validation_members: int = 256
    weight_decay: float = 0.

    def __post_init__(self):
        for key, codes in CODES.items():
            if getattr(self, key) not in codes:
                raise ValueError(f'Invalid {key}: {getattr(self, key)}')
        if min(self.lookback, self.latent, self.width, self.epochs, self.batch_size) < 1 or self.members < 2:
            raise ValueError('Positive dimensions/epochs required; training members >= 2')
        if self.validation_members < 2 or self.location_embedding < 0 or self.weight_decay < 0:
            raise ValueError('Validation members >=2, nonnegative embedding and weight decay required')
        if self.patience < 0 or (self.patience and self.patience >= self.epochs):
            raise ValueError('Patience must be 0 (fixed epochs) or below the epoch cap')
        if not (0 < self.lr < float('inf') and float(f'{self.lr:g}') == self.lr):
            raise ValueError(f'lr must be positive and representable in the string: {self.lr}')

    @property
    def scenario_string(self):
        return 'b0:' + ':'.join(PREFIX[key] + encode(key, value) for key, value in asdict(self).items())

    @classmethod
    def from_string(cls, value):
        """Require a complete canonical string; typos never become defaults."""
        try:
            prefix, *tokens = value.split(':')
            if prefix != 'b0' or len(tokens) != len(PREFIX):
                raise ValueError()
            options = {}
            for field, token in zip(fields(cls), tokens):
                if not token.startswith(PREFIX[field.name]):
                    raise ValueError()
                text = token[len(PREFIX[field.name]):]
                if field.name in CODES:
                    options[field.name] = {code: name for name, code in CODES[field.name].items()}[text]
                elif field.type is bool:
                    options[field.name] = {'0': False, '1': True}[text]
                else:
                    options[field.name] = field.type(text)
            scenario = cls(**options)
            if scenario.scenario_string != value:
                raise ValueError()
            return scenario
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f'Invalid scenario string: {value}') from error

    @classmethod
    def from_config(cls, config):
        """Read a saved season-CV configuration, ignoring non-scenario settings."""
        return cls(**{field.name: config[field.name] for field in fields(cls)})

    def flags(self):
        flags = []
        for key, value in asdict(self).items():
            option = '--' + key.replace('_', '-')
            if isinstance(value, bool):
                if value:
                    flags.append(option)
                elif key == 'annual_calendar':
                    flags.append('--no-annual-calendar')
            else:
                flags.extend((option, str(value)))
        return flags


def encode(key, value):
    if key in CODES:
        return CODES[key][value]
    if isinstance(value, bool):
        return str(int(value))
    return f'{value:g}' if isinstance(value, float) else str(value)


def ofat(anchor, **options):
    """One-factor-at-a-time variations around an anchor, named `<field>_<value>`."""
    return {f'{key}_{encode(key, value)}': replace(anchor, **{key: value})
            for key, values in options.items() for value in values if value != getattr(anchor, key)}


ANCHOR = TrainingScenario()
BASELINE = TrainingScenario(lookback=8, count_transform='raw', geography=False, dynamics=False)

# Explicit names stay stable when the registry grows. Each candidate is compared
# with its stated control; do not pick later candidates adaptively at seed 42.
ESSENTIAL = {
    'baseline': (BASELINE, None, 'Historical raw-count B0 control'),
    'anchor': (ANCHOR, 'baseline', 'Existing fourth-root/geography/12-week/dynamics candidate'),
    'state_us': (replace(ANCHOR, heads='state_us'), 'anchor', 'Separate stochastic state and US heads'),
    'residual2': (replace(ANCHOR, decoder='residual2'), 'anchor', 'Two modulated blocks, latent held at 16'),
    'latent32': (replace(ANCHOR, latent=32), 'anchor', 'Latent dimension alone'),
    'residual2_z32': (replace(ANCHOR, decoder='residual2', latent=32), 'residual2', 'Designed decoder with latent 32'),
    'mlp_h8': (replace(ANCHOR, lookback=8, dynamics=False), 'mlp_h12', 'Eight weeks, same MLP recipe'),
    'mlp_h12': (replace(ANCHOR, dynamics=False), 'anchor', 'Twelve weeks; dynamics ablation'),
    'mlp_h26': (replace(ANCHOR, lookback=26, dynamics=False), 'mlp_h12', 'Twenty-six weeks, same MLP recipe'),
    'balanced': (replace(ANCHOR, loss_weights='balanced_admissions'), 'anchor', 'Equal admission loss weights'),
    'flu_only': (replace(ANCHOR, loss_weights='flu_only'), 'anchor', 'Flu supervision, all six input channels'),
    'conv_h12': (replace(ANCHOR, encoder='conv'), 'anchor', 'Shared temporal filters at twelve weeks'),
    'mlp_h26_dynamics': (replace(ANCHOR, lookback=26), 'mlp_h26', 'Dynamics at twenty-six weeks; convolution control'),
    'conv_h26': (replace(ANCHOR, encoder='conv', lookback=26), 'mlp_h26_dynamics', 'Shared temporal filters at twenty-six weeks'),
}

# Levels explored around each reference. Fixed: geography features,
# target weights [1,1,1,.5,.5,.5], width 64, batch 8, 128 training
# draws, learning rate .001. Add or remove a level here and `crosses` follows;
# this is the single registry of what the experiment varies.
AXES = dict(lookback=(8, 12), dynamics=(False, True), encoder=('mlp', 'conv'), decoder=('legacy', 'residual2'),
            latent=(16, 32), spatial=('none', 'attention'), noise=('global', 'local'), heads=('shared', 'state_us'),
            count_transform=('raw', 'rate', 'sqrt', 'fourth_root', 'log1p'),
            ed_transform=('linear', 'logit', 'fourth_root'), geography=(False, True),
            # Correlated national error: a per-episode, per-channel common mode
            # shared by every location; native US is still predicted directly.
            us_error=('none', 'shared_factor'),
            # (epochs, patience): fixed 50/100/300 epochs isolate training length;
            # patience 20 up to 300 adds validation checkpoint selection on top.
            stopping=((50, 0), (100, 0), (300, 0), (300, 20)))


# Three deliberately chosen reference recipes, not selected from sweep scores.
# Hold the scoring objective and seeds fixed while testing local changes.
CROSS_ANCHORS = {
    'raw': replace(BASELINE, loss_weights='objective'),
    'anchor': replace(ANCHOR, loss_weights='objective'),
    'conv': replace(ANCHOR, loss_weights='objective', encoder='conv', decoder='residual2',
                    latent=32, spatial='attention', noise='local', heads='state_us',
                    ed_transform='logit'),
}


def crosses():
    """One-factor changes around three references; shared configurations run once.

    Every level in `AXES` is visited on every reference, so adding a level there
    extends the experiment without touching this function. Stopping is a paired
    policy. This screens local effects; it does not estimate interactions.
    """
    scenarios = dict(CROSS_ANCHORS)
    seen = set(scenarios.values())
    for name, anchor in CROSS_ANCHORS.items():
        for field, values in AXES.items():
            for value in values:
                options = dict(zip(('epochs', 'patience'), value)) if field == 'stopping' else {field: value}
                candidate = replace(anchor, **options)
                if candidate not in seen:
                    label = '_'.join(map(str, value)) if field == 'stopping' else encode(field, value)
                    scenarios[f'{name}__{field}_{label}'] = candidate
                    seen.add(candidate)
    return scenarios


# Add a named suite here for each new exploration, e.g.
#   'capacity': {'anchor': ANCHOR, **ofat(ANCHOR, width=(32, 128), epochs=(100,))},
SUITES = {
    'essential': lambda: {name: value[0] for name, value in ESSENTIAL.items()},
    'crosses': crosses,
    'B0.1': lambda: __import__('tapestry.models.b01_suite', fromlist=['scenarios']).scenarios(),
}


def get_training_scenario(name):
    if name in ESSENTIAL:
        return ESSENTIAL[name][0]
    if name.startswith('B0.1/'):
        return get_scenarios('B0.1')[name]
    return TrainingScenario.from_string(name)


def get_scenarios(suite='essential'):
    if suite not in SUITES:
        raise ValueError(f'Unknown suite: {suite}')
    return SUITES[suite]()
