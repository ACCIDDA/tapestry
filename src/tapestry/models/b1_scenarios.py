"""Canonical B1 formulations and masking settings, using B0's string vocabulary."""
import argparse
from dataclasses import asdict, dataclass, fields, replace
import hashlib
import json
import math

from .scenarios import TrainingScenario, PREFIX, CODES

PREFIXES = {**PREFIX, 'pipeline': 'pipe_', 'mask_rate': 'mask', 'mask_recent': 'mr', 'mask_gap': 'mg', 'mask_outage': 'mo', 'nowcast_weight': 'nw', 'validation_mode': 'val', 'revision_rate': 'rev'}
OPTIONS = {**CODES, 'pipeline': {'direct': 'direct', 'two_stage': 'two', 'direct_finalflag': 'flag', 'joint_aux025': 'aux025', 'joint_aux': 'aux', 'gated_revision': 'gate'},
           'validation_mode': {'recipe': 'recipe', 'natural_forecast': 'natural'}}
NEW_FIELDS = ('nowcast_weight', 'validation_mode', 'revision_rate')
MODEL_FIELDS = ('encoder', 'spatial', 'decoder', 'heads', 'noise', 'head_sharing', 'count_transform',
                'ed_transform', 'geography', 'dynamics', 'annual_calendar', 'location_embedding', 'us_error')


@dataclass(frozen=True)
class B1Scenario(TrainingScenario):
    ed_transform: str = 'logit'
    fit_partition: str = 'target'
    epochs: int = 100
    patience: int = 30
    pipeline: str = 'two_stage'
    mask_rate: float = .5
    mask_recent: float = .5
    mask_gap: float = .3
    mask_outage: float = .2
    nowcast_weight: float = .25
    validation_mode: str = 'recipe'
    revision_rate: float = 0.

    def __post_init__(self):
        for key, choices in OPTIONS.items():
            if getattr(self, key) not in choices:
                raise ValueError(f'Invalid {key}: {getattr(self, key)}')
        if self.count_transform not in ('sqrt', 'fourth_root', 'log1p') or self.loss_weights != 'objective':
            raise ValueError('B1 uses transformed admissions and the scientific objective')
        if min(self.lookback, self.width, self.latent, self.epochs, self.batch_size) < 1 or self.lookback < 2 or self.width < 3:
            raise ValueError('Positive dimensions/epochs required, lookback >=2 and width >=3')
        if self.spatial != 'none' and self.width % 4:
            raise ValueError('Spatial attention requires width divisible by four')
        if min(self.members, self.validation_members) < 2 or self.patience < 0 or self.location_embedding < 0:
            raise ValueError('Members >=2 and nonnegative patience/location embedding required')
        if not math.isfinite(self.lr) or self.lr <= 0 or not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError('Finite positive learning rate and nonnegative weight decay required')
        if not math.isfinite(self.mask_rate) or not 0 <= self.mask_rate <= 1:
            raise ValueError('mask_rate must be between zero and one')
        if not math.isfinite(self.nowcast_weight) or self.nowcast_weight < 0:
            raise ValueError('Nowcast coefficient must be finite and nonnegative')
        if not math.isfinite(self.revision_rate) or not 0 <= self.revision_rate <= 1:
            raise ValueError('Revision augmentation probability must be in [0, 1]')
        if self.pipeline in ('joint_aux', 'gated_revision') and self.validation_mode != 'natural_forecast':
            raise ValueError('New revision formulations require natural forecast selection')
        mix = (self.mask_recent, self.mask_gap, self.mask_outage)
        if any(not math.isfinite(p) or p < 0 for p in mix) or not math.isclose(sum(mix), 1., abs_tol=1e-10):
            raise ValueError('Conditional recent/gap/outage probabilities must sum to one')

    @property
    def mask_probabilities(self):
        return (1 - self.mask_rate, self.mask_rate * self.mask_recent,
                self.mask_rate * self.mask_gap, self.mask_rate * self.mask_outage)

    @property
    def scenario_string(self):
        def encode(key, value):
            if key in OPTIONS:
                return OPTIONS[key][value]
            if isinstance(value, bool):
                return str(int(value))
            field = next(f for f in fields(self) if f.name == key)
            return repr(float(value)) if field.type is float else str(value)
        legacy = all(getattr(self, k) == next(f.default for f in fields(self) if f.name == k) for k in NEW_FIELDS)
        items = [(k, v) for k, v in asdict(self).items() if not legacy or k not in NEW_FIELDS]
        return ('b1:v2:' if legacy else 'b1:v3:') + ':'.join(PREFIXES[k] + encode(k, v) for k, v in items)

    @classmethod
    def from_string(cls, value):
        """Complete strings only; every field round-trips without float truncation."""
        try:
            version, schema, *tokens = value.split(':')
            selected_fields = [f for f in fields(cls) if schema != 'v2' or f.name not in NEW_FIELDS]
            if version != 'b1' or schema not in ('v2', 'v3') or len(tokens) != len(selected_fields):
                raise ValueError()
            options = {}
            for field, token in zip(selected_fields, tokens):
                prefix = PREFIXES[field.name]
                if not token.startswith(prefix):
                    raise ValueError()
                raw = token[len(prefix):]
                if field.name in OPTIONS:
                    options[field.name] = {code: name for name, code in OPTIONS[field.name].items()}[raw]
                elif field.type is bool:
                    options[field.name] = {'0': False, '1': True}[raw]
                else:
                    options[field.name] = field.type(raw)
            candidate = cls(**options)
            if candidate.scenario_string != value:
                raise ValueError()
            return candidate
        except (ValueError, TypeError, KeyError) as error:
            raise ValueError(f'Invalid B1 scenario string: {value}') from error

    @property
    def run_id(self):
        """Readable short directory name; full canonical string remains in manifests."""
        digest = hashlib.sha256(self.scenario_string.encode()).hexdigest()[:12]
        return (f'{self.encoder}-{self.fit_partition}-{self.pipeline}-mask{self.mask_rate:g}-{digest}')

    def flags(self):
        return ['--scenario', self.scenario_string]

    def model_options(self):
        return dict({k: getattr(self, k) for k in MODEL_FIELDS},
                    supplied_final=self.pipeline != 'direct',
                    parallel_recent=self.pipeline in ('joint_aux025', 'joint_aux', 'gated_revision'),
                    revision_bridge=self.pipeline == 'gated_revision')


# B0.1's four best configurations, read from
# data/experiments/B0.1/ranking-509b07b0d243/configuration_ranking.csv
# (combined location-relative WIS .8834/.8948/.8949/.8968). All four share
# h12/4rt/logit/geo/dyn/mlp/no-spatial/z16/w64/lr.001 and differ only in how the
# six targets are partitioned and in the epoch cap, so the B1 comparison changes
# the task and the inputs, not the architecture. These are B0's winners carried
# over, not a claim that they win under Wednesday inputs.
B0_TOP4 = (
    dict(fit_partition='target', epochs=100),
    dict(fit_partition='pathogen', epochs=300),
    dict(fit_partition='target', epochs=300),
    dict(fit_partition='pathogen', epochs=100),
)
B0_TOP4_BASE = dict(lookback=12, count_transform='fourth_root', ed_transform='logit',
                    geography=True, dynamics=True, loss_weights='objective', encoder='mlp',
                    spatial='none', heads='shared', decoder='legacy', noise='global',
                    us_error='none', latent=16, width=64, patience=30, batch_size=8,
                    members=128, lr=.001, head_sharing='shared', annual_calendar=True,
                    location_embedding=0, validation_members=256, weight_decay=0.)


def b0_top4(pipeline, mask_rate, **overrides):
    """B0's four best configurations at one pipeline and masking level.

    `overrides` replaces training-budget fields (width, members, and the epoch
    cap when a smoke run needs one); the architecture fields are the point of the
    suite and are not overridable.
    """
    scenarios = []
    for recipe in B0_TOP4:
        options = dict(B0_TOP4_BASE, **recipe, pipeline=pipeline, mask_rate=float(mask_rate))
        options.update({k: v for k, v in overrides.items() if v is not None})
        scenarios.append(B1Scenario(**options))
    return scenarios


# Exact B0 formulation choices, adapted to the B1 data/decoder contract. Scores
# motivated inclusion; they do not establish these architectures as B1 winners.
PRESETS = {
    'target_mlp': B1Scenario(),
    'pathogen_mlp': B1Scenario(fit_partition='pathogen'),
    'target_conv': B1Scenario(encoder='conv'),
    'target_multiscale': B1Scenario(encoder='multiscale_conv'),
    'spatial_conv': B1Scenario(encoder='conv', spatial='attention', fit_partition='all'),
    'joint_mlp': B1Scenario(spatial='joint_location_target', head_sharing='pathogen', fit_partition='all', epochs=300),
}


def add_scenario_args(parser):
    parser.add_argument('--preset', choices=tuple(PRESETS), help='B0-derived formulation; default target_mlp')
    parser.add_argument('--scenario', help='Complete canonical b1:v2 or b1:v3 string')
    for field in fields(B1Scenario):
        option = '--' + field.name.replace('_', '-')
        kwargs = dict(default=None)
        if field.type is bool:
            kwargs['action'] = argparse.BooleanOptionalAction
        else:
            kwargs['type'] = field.type
            if field.name in OPTIONS:
                choices = tuple(OPTIONS[field.name])
                if field.name == 'count_transform':
                    choices = ('sqrt', 'fourth_root', 'log1p')
                elif field.name == 'loss_weights':
                    choices = ('objective',)
                kwargs['choices'] = choices
        if field.name == 'mask_rate':
            kwargs['help'] = 'Probability an episode receives artificial masking, not cell dropout (default .5)'
        parser.add_argument(option, **kwargs)


def resolve(args):
    if args.scenario and args.preset:
        raise ValueError('Choose a preset or a scenario string, not both')
    base = B1Scenario.from_string(args.scenario) if args.scenario else PRESETS[args.preset or 'target_mlp']
    overrides = {f.name: getattr(args, f.name) for f in fields(B1Scenario) if getattr(args, f.name) is not None}
    return replace(base, **overrides)


def comparison_grid(base, presets=None, mask_rates=(0., .25, .5), pipelines=None):
    """Masking levels per formulation, for the requested pipelines.

    `pipelines=None` keeps the full contrast: an unmasked direct control plus a
    two-stage model at each masking rate. Naming one pipeline isolates a single
    factor — `two_stage` with `mask_rates=[0]` adds nowcasting to B0's
    information state, `direct` with `mask_rates=[.5]` adds masking to B0's task.
    """
    if not mask_rates:
        raise ValueError('At least one masking rate is required')
    pipelines = tuple(pipelines) if pipelines else ('direct', 'two_stage')
    if any(p not in ('direct', 'two_stage') for p in pipelines):
        raise ValueError('Pipelines must be direct and/or two_stage')
    recipes = [base]
    if presets:
        # Presets control architecture and fit grouping. Global training settings,
        # chosen transforms/calendar and mask mixture come from the resolved base,
        # except fields explicitly constituting each named preset below.
        varying = ('encoder', 'spatial', 'head_sharing', 'fit_partition')
        recipes = [replace(base, **{k: getattr(PRESETS[name], k) for k in varying}) for name in presets]
    candidates = {}
    for recipe in recipes:
        for pipeline in pipelines:
            for rate in mask_rates:
                candidate = replace(recipe, pipeline=pipeline, mask_rate=float(rate))
                candidates[candidate.scenario_string] = candidate
    return list(candidates.values())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_scenario_args(parser)
    args = parser.parse_args(argv)
    scenario = resolve(args)
    print(json.dumps(dict(run_id=scenario.run_id, scenario=scenario.scenario_string,
                         config=asdict(scenario), mask_probabilities=scenario.mask_probabilities), indent=2))


if __name__ == '__main__':
    main()
