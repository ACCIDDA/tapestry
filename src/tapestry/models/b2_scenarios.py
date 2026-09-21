"""B1 formulation B with explicit covariate subsets and input information states."""
from dataclasses import asdict, dataclass, fields, replace
import hashlib
from itertools import combinations

from .b1_scenarios import B1Scenario, B0_TOP4_BASE

SOURCE_GROUPS = ('inpatient', 'outpatient', 'ww_wval_like', 'ww_pct_rank', 'kinsa')
FACTORIAL_GROUPS = SOURCE_GROUPS[:4]
INPUT_MODES = ('finalized', 'wednesday')


def _subsets(groups):
    return ('none',) + tuple('+'.join(group) for size in range(1, len(groups) + 1)
                             for group in combinations(groups, size))


# Any subset of the source groups is a valid scenario; each suite enumerates its own.
COVARIATE_SETS = _subsets(SOURCE_GROUPS)
SUITE_SETS = {
    # The original factorial and screen predate Kinsa and keep their four claims/wastewater groups.
    'B2-covariates': _subsets(FACTORIAL_GROUPS),
    'B2-screen': ('none', *FACTORIAL_GROUPS, 'inpatient+outpatient',
                  'ww_wval_like+ww_pct_rank', '+'.join(FACTORIAL_GROUPS)),
    'B2-kinsa': ('none', 'kinsa'),
}


@dataclass(frozen=True)
class B2Scenario(B1Scenario):
    pipeline: str = 'direct_finalflag'
    covariate_set: str = 'none'
    input_mode: str = 'wednesday'

    def __post_init__(self):
        super().__post_init__()
        if self.pipeline != 'direct_finalflag':
            raise ValueError('B2 uses formulation B: direct forecasting with supplied-final flags')
        if self.covariate_set not in COVARIATE_SETS or self.input_mode not in INPUT_MODES:
            raise ValueError('Unknown B2 covariate subset or input mode')

    @property
    def scenario_string(self):
        base = B1Scenario(**{f.name: getattr(self, f.name) for f in fields(B1Scenario)})
        return base.scenario_string.replace('b1:v2:', 'b2:v1:', 1) + f':cov_{self.covariate_set}:input_{self.input_mode}'

    @classmethod
    def from_string(cls, value):
        try:
            base, covariates, mode = value.rsplit(':', 2)
            if not base.startswith('b2:v1:') or not covariates.startswith('cov_') or not mode.startswith('input_'):
                raise ValueError()
            scenario = cls(**asdict(B1Scenario.from_string(base.replace('b2:v1:', 'b1:v2:', 1))),
                           covariate_set=covariates[4:], input_mode=mode[6:])
            if scenario.scenario_string != value:
                raise ValueError()
            return scenario
        except (ValueError, TypeError) as error:
            raise ValueError(f'Invalid B2 scenario string: {value}') from error

    @property
    def run_id(self):
        digest = hashlib.sha256(self.scenario_string.encode()).hexdigest()[:12]
        return f'b2-{self.fit_partition}-{self.input_mode}-{self.covariate_set}-{digest}'


def scenarios(covariate_sets=SUITE_SETS['B2-covariates'], **budget):
    """Each listed subset, including matched no-covariate controls, for both B1 anchors."""
    anchors = {
        'target_gap': B2Scenario(**B0_TOP4_BASE, epochs=100, fit_partition='target',
                                 mask_rate=.5, mask_recent=0., mask_gap=1., mask_outage=0.),
        'pathogen_mixed': B2Scenario(**B0_TOP4_BASE, epochs=300, fit_partition='pathogen',
                                    mask_rate=.5, mask_recent=.5, mask_gap=.3, mask_outage=.2),
    }
    overrides = {key: value for key, value in budget.items() if value is not None}
    return {f'{name}__{mode}__{subset}': replace(anchor, input_mode=mode, covariate_set=subset, **overrides)
            for name, anchor in anchors.items() for mode in INPUT_MODES for subset in covariate_sets}


def add_scenario_args(parser):
    from .b1_scenarios import add_scenario_args as add_b1_args
    add_b1_args(parser)
    parser.add_argument('--covariate-set', choices=COVARIATE_SETS)
    parser.add_argument('--input-mode', choices=INPUT_MODES)


def resolve(args):
    if args.preset:
        raise ValueError('Use a complete B2 scenario or explicit options, not a B1 preset')
    base = B2Scenario.from_string(args.scenario) if args.scenario else B2Scenario()
    return replace(base, **{f.name: getattr(args, f.name) for f in fields(B2Scenario)
                            if getattr(args, f.name, None) is not None})
