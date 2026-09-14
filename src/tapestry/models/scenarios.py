"""Influpaint-style immutable scenarios, readable strings, and essential/grid sets.

Adapted from the pattern in influpaint/influpaint/batch/scenarios.py:
TrainingScenario, scenario_string, and get_essential_scenarios. No MLflow needed.
"""
from dataclasses import asdict, dataclass, replace
from itertools import product

from .experiments import LOSS_WEIGHTS


@dataclass(frozen=True)
class TrainingScenario:
    lookback: int = 12
    count_transform: str = 'fourth_root'
    geography: bool = True
    dynamics: bool = True
    loss_weights: str = 'influenza_first'
    encoder: str = 'mlp'
    heads: str = 'shared'
    decoder: str = 'legacy'
    latent: int = 16

    def __post_init__(self):
        choices = dict(count_transform=('raw', 'sqrt', 'fourth_root'),
                       loss_weights=tuple(LOSS_WEIGHTS), encoder=('mlp', 'conv'),
                       heads=('shared', 'state_us'), decoder=('legacy', 'residual2'))
        for key, values in choices.items():
            if getattr(self, key) not in values:
                raise ValueError(f'Invalid {key}: {getattr(self, key)}')
        if self.lookback < 1 or self.latent < 1:
            raise ValueError('lookback and latent must be positive')

    @property
    def scenario_string(self):
        return 'b0::' + '::'.join(f'{key}={int(value) if isinstance(value, bool) else value}'
                                  for key, value in asdict(self).items())

    @classmethod
    def from_string(cls, value):
        """Require a complete canonical string; typos never become defaults."""
        try:
            prefix, *parts = value.split('::')
            pairs = [part.split('=') for part in parts]
            options = dict(pairs)
            if prefix != 'b0' or len(options) != len(parts) or set(options) != set(cls.__dataclass_fields__):
                raise ValueError()
            for key in ('lookback', 'latent'):
                options[key] = int(options[key])
            for key in ('geography', 'dynamics'):
                if options[key] not in ('0', '1'):
                    raise ValueError()
                options[key] = options[key] == '1'
            scenario = cls(**options)
            if scenario.scenario_string != value:
                raise ValueError()
            return scenario
        except (TypeError, ValueError) as error:
            raise ValueError(f'Invalid scenario string: {value}') from error

    def flags(self):
        flags = []
        for key, value in asdict(self).items():
            option = '--' + key.replace('_', '-')
            if isinstance(value, bool):
                if value:
                    flags.append(option)
            else:
                flags.extend((option, str(value)))
        return flags


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


def get_training_scenario(name):
    if name in ESSENTIAL:
        return ESSENTIAL[name][0]
    return TrainingScenario.from_string(name)


def get_scenarios(suite='essential'):
    if suite == 'essential':
        return {name: value[0] for name, value in ESSENTIAL.items()}
    if suite != 'grid':
        raise ValueError(f'Unknown suite: {suite}')
    # Full interactions, with representation fixed; spatial attention belongs to B1.
    keys = ('lookback', 'dynamics', 'loss_weights', 'encoder', 'heads', 'decoder', 'latent')
    values = ((8, 12, 26), (False, True), tuple(LOSS_WEIGHTS), ('mlp', 'conv'),
              ('shared', 'state_us'), ('legacy', 'residual2'), (16, 32))
    grid = {'baseline': BASELINE}
    for setting in product(*values):
        scenario = replace(ANCHOR, **dict(zip(keys, setting)))
        grid[scenario.scenario_string] = scenario
    return grid
