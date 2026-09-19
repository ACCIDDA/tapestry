"""What actually differs between B0 and B1, so the manager itself has no model branches.

A backend supplies four things: how to expand a suite into scenarios, which extra
experiment settings to record, how to build the fitting commands for one seed, and
which artifacts prove a seed finished. Everything else — planning, attempts,
resume, status, ranking, comparison, dispatch — is one implementation shared by
both models, and scoring is `tapestry.evaluation.totals` for both.
"""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import shutil
import sys

from .scenarios import SUITES, get_scenarios, get_training_scenario
from .provenance import SEASONS
from .quantiles import LEVELS
from tapestry.model_data.wednesday import DEFAULT_DATASET as B1_DATASET

FROZEN = 'data/evaluation/b0_hub_comparison_q23'
LOCATIONS = 'data/metadata/locations.csv'
DATASETS = {'Forward': 'data/processed/forward_2025.npz', 'B0': 'data/processed/build_b_finalized.npz', 'B1': B1_DATASET}
EVAL_MEMBERS = {'Forward': 256, 'B0': 256, 'B1': 256}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_denominators(frozen):
    """Every frozen location must have positive ensemble WIS before any fitting.

    A nonpositive denominator would make a relative score undefined; catching it
    at planning time costs seconds, while catching it at ranking time costs a
    whole experiment. Shared by both models, which divide by the same totals.
    """
    import numpy as np
    import pandas as pd
    from tapestry.evaluation.totals import case_totals, frozen_cases
    from tapestry.evaluation.scoring import match_forecasts
    frozen = Path(frozen)
    denominators = []
    for case in frozen_cases(frozen):
        units = pd.read_parquet(frozen / case['directory'] / 'units.parquet')
        quantiles = pd.read_parquet(frozen / case['directory'] / 'quantiles.parquet')
        ensemble = match_forecasts(quantiles[quantiles.model == case['ensemble']], units, case['target'])
        totals = case_totals(ensemble, ensemble, case)
        for location, value in totals.groupby('location').ensemble_wis.sum().items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'Nonpositive frozen ensemble WIS: {case["directory"]}/{location}')
            denominators.append(dict(case=case['directory'], location=location, ensemble_wis=float(value)))
    if not denominators:
        raise ValueError('No frozen ensemble support')
    return denominators


def snapshot(folder, settings, extra=()):
    """Pin inputs and save a runnable source copy, so a launch never races an edit.

    The dispatcher puts this copy on PYTHONPATH, so workers run the code the
    experiment was planned with even while the working tree moves on.
    """
    from .provenance import save, git_state
    root = Path(__file__).resolve().parents[3]
    files = list((root / 'src').rglob('*.py')) + list((root / 'src').rglob('*.R'))
    files += [root / 'scripts/jlessler.sbatch', root / 'scripts/notify.sbatch',
              root / 'pyproject.toml', *extra]
    notifications = folder / 'notifications'
    notifications.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / 'scripts/b01_notify.py', notifications / 'b01_notify.py')
    destination = folder / 'code'
    if destination.exists():
        shutil.rmtree(destination)
    hashes = {}
    for source in files:
        target = destination / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(source.relative_to(root))] = sha(source)
    frozen = Path(settings['frozen'])
    inputs = {settings['dataset'], settings['population_file'], str(frozen / 'manifest.json')}
    if settings.get('model') == 'B1':
        from tapestry.model_data.wednesday import WednesdayDataset
        calendar_source = WednesdayDataset.load(settings['dataset']).metadata.get('calendar_source')
        if calendar_source:
            inputs.add(calendar_source['path'])
    inputs |= {str(p) for p in list(frozen.rglob('units.parquet')) + list(frozen.rglob('quantiles.parquet'))}
    settings['input_sha256'] = {p: sha(p) for p in sorted(inputs)}
    settings['source_snapshot_sha256'] = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    save(folder / 'experiment.json', settings)
    save(folder / 'preflight.json', dict(source_sha256=hashes, input_sha256=settings['input_sha256'],
                                         frozen_location_denominators=frozen_denominators(frozen), **git_state()))


def check_frozen(frozen):
    """Fail before any fitting when runs could not be scored on the current quantile grid."""
    expected = [f'q{q:g}' for q in LEVELS]
    try:
        quantiles = json.loads((Path(frozen) / 'manifest.json').read_text()).get('quantiles')
    except (OSError, ValueError) as error:
        raise ValueError(f'No frozen scoring support at {frozen}; build it with scripts/b0_prepare.sbatch') from error
    if quantiles != expected:
        raise ValueError(f'{frozen} holds quantiles {quantiles}; rebuild frozen support for the {len(expected)}-level grid')


def model_of(value):
    """The model that owns a scenario string or an experiment's settings."""
    if isinstance(value, dict):
        return value.get('model', 'B0')
    return 'Forward' if str(value).startswith('forward:') else 'B1' if str(value).startswith('b1:') else 'B0'


class B0Backend:
    model = 'B0'
    output_dir = 'cv'

    def scenarios(self, args):
        if args.scenario:
            return {name: get_training_scenario(name) for name in args.scenario}
        return get_scenarios(args.suite)

    def settings(self, args):
        return {}

    def check_inputs(self, settings):
        check_frozen(settings['frozen'])

    def prepare(self, folder, settings):
        from .provenance import save
        snapshot(folder, settings, extra=[Path(__file__).parents[3] / 'docs/design/b0.1.md'])
        if settings.get('suite') == 'B0.1':
            from .b01_suite import manifest
            save(folder / 'design.json', manifest())

    def fit_commands(self, scenario, seed, settings, output):
        """One season-CV process fits all three folds."""
        return [[sys.executable, '-m', 'tapestry.models.season_cv',
                 '--dataset', settings['dataset'], '--population-file', settings['population_file'],
                 '--eval-members', str(settings['eval_members']), '--device', settings['device'],
                 '--seed', str(seed), '--output', str(output), *scenario.flags()]]

    def collect_manifest(self, output):
        """season_cv already writes the run-level manifest itself."""

    def required_artifacts(self, output):
        required = ['manifest.json', 'scores.csv', 'totals.csv']
        required += [f'eval_{s}/{n}' for s in SEASONS
                     for n in ('model.pt', 'forecasts.npz', 'training.json', 'scores.csv')]
        return required

    def complete(self, output):
        try:
            manifest = json.loads((output / 'manifest.json').read_text())
            return sorted(f['eval_season'] for f in manifest['folds']) == sorted(SEASONS)
        except (KeyError, ValueError, TypeError):
            return False


class B1Backend:
    model = 'B1'
    output_dir = 'b1'
    STRESS = ('natural', 'recent', 'gap', 'outage')

    # Defaults for the open B1 grid. A named suite carries its own values and
    # only uses these when the flag is given explicitly.
    BUDGET = dict(epochs=300, patience=50, members=128, validation_members=256, width=64, batch_size=8)

    def add_plan_args(self, parser):
        group = parser.add_argument_group('B1 planning (--suite B1)')
        from .b1_scenarios import PRESETS
        group.add_argument('--presets', choices=tuple(PRESETS), nargs='+', default=list(PRESETS))
        group.add_argument('--mask-rates', type=float, nargs='+', default=[0., .25, .5])
        group.add_argument('--pipelines', choices=('direct', 'two_stage'), nargs='+',
                           help='Restrict the grid to these pipelines; default expands both')
        for key, value in self.BUDGET.items():
            group.add_argument('--' + key.replace('_', '-'), type=int, default=None,
                               help=f'default {value}; a named suite keeps its own unless given')
        group.add_argument('--retrospective', action='store_true')

    # B0's four best configurations under Wednesday inputs, the two-stage B1
    # design, or artificial masking. Label/source support can also differ from B0.
    SUITES = {
        # Inputs plus nowcasting: the two-stage recent->future path, still with
        # natural availability and no artificial masking.
        'B1-onlynowcast': dict(pipeline='two_stage', mask_rate=0.),
        # Inputs plus masking: B0's direct task under artificial missingness.
        'B1-onlymask': dict(pipeline='direct', mask_rate=.5),
        'B1-decisive-A': dict(pipeline='direct', mask_rate=.5),
        'B1-direct-finalflag': dict(pipeline='direct_finalflag', mask_rate=.5),
        'B1-joint-aux025': dict(pipeline='joint_aux025', mask_rate=.5),
        'B1-overnight': {},
        'B1-formulations': {},
        'B1-revisions': {},
    }

    def scenarios(self, args):
        from .b1_scenarios import B1Scenario, b0_top4, comparison_grid
        if args.scenario:
            return {s.run_id: s for s in map(B1Scenario.from_string, args.scenario)}
        # Unset budget flags stay None so a suite keeps its own recipe values;
        # an explicit flag overrides, which is what makes a cheap smoke run of
        # the real suite possible without redefining it.
        budget = {key: getattr(args, key) for key in self.BUDGET}
        if args.suite == 'B1-revisions':
            from .b1_revision_suite import scenarios
            return scenarios(**budget)
        if args.suite in ('B1-overnight', 'B1-formulations'):
            from .b1_overnight import scenarios
            return scenarios(formulations_only=args.suite == 'B1-formulations', **budget)
        if args.suite in self.SUITES:
            recipes = b0_top4(**self.SUITES[args.suite], **budget)
            if args.suite in ('B1-decisive-A', 'B1-direct-finalflag', 'B1-joint-aux025'):
                recipes = recipes[:1]
            return {s.run_id: s for s in recipes}
        base = B1Scenario(**{key: value if value is not None else self.BUDGET[key]
                             for key, value in budget.items()})
        grid = comparison_grid(base, args.presets, args.mask_rates, getattr(args, 'pipelines', None))
        return {s.run_id: s for s in grid}

    def settings(self, args):
        # B1 always uses B0's leave-one-season-out protocol; there is no other split.
        # `_refit_v1` marks B0's select-then-refit training procedure.
        return dict(model='B1', protocol='season_cv_refit_v1', retrospective=args.retrospective)

    def check_inputs(self, settings):
        from tapestry.model_data.wednesday import WednesdayDataset
        from .b1_run import populations
        from .b1_seasons import fold
        check_frozen(settings['frozen'])
        data = WednesdayDataset.load(settings['dataset'])
        calendar_source = data.metadata.get('calendar_source')
        if calendar_source and sha(calendar_source['path']) != calendar_source['sha256']:
            raise ValueError('The B0 calendar dataset changed; rebuild B1 against it and use a new experiment')
        populations(settings['population_file'], data.locations)
        for held in SEASONS:
            fold(data, held)  # every fold must have fitting, validation, refit and evaluation episodes
        if settings['eval_members'] < 2:
            raise ValueError('Fair CRPS requires at least two evaluation members')
        for path, expected in settings.get('input_sha256', {}).items():
            if sha(path) != expected:
                raise ValueError(f'B1 input changed since planning: {path}; use a new experiment')

    def prepare(self, folder, settings):
        root = Path(__file__).parents[3]
        extra = [root / 'docs/design/b1.md']
        if settings['suite'] == 'B1-overnight':
            extra.append(root / 'docs/workflows/b1-overnight.md')
        if settings['suite'] == 'B1-formulations':
            extra.append(root / 'docs/workflows/b1-300.md')
        if settings['suite'] == 'B1-revisions':
            extra.append(root / 'docs/workflows/b1-revisions.md')
        snapshot(folder, settings, extra=extra)

    def fit_commands(self, scenario, seed, settings, output):
        """One process per leave-one-season-out fold, as B0 fits three folds."""
        base = [sys.executable, '-m', 'tapestry.models', 'b1', 'train', *scenario.flags(),
                '--dataset', settings['dataset'], '--population-file', settings['population_file'],
                '--eval-members', str(settings['eval_members']), '--device', settings['device'],
                '--seed', str(seed)]
        if settings['retrospective']:
            base.append('--retrospective')
        return [[*base, '--held-out-season', held, '--output', str(output / f'eval_{held}')]
                for held in SEASONS]

    def collect_manifest(self, output):
        """Fold into one run-level manifest, as season_cv writes for B0.

        Every fold must report the configuration and seed it was actually asked
        for; a mismatch means the folds are not one run and must not be pooled.
        """
        from .provenance import save
        folds, common = {}, None
        for held in SEASONS:
            fold = json.loads((output / f'eval_{held}' / 'manifest.json').read_text())
            if fold.get('held_out_season') != held:
                raise ValueError(f'Fold {held} checkpoint reports {fold.get("held_out_season")}')
            identity = (fold['scenario'], fold['seed'])
            if common is not None and identity != common:
                raise ValueError(f'Folds disagree on configuration/seed: {identity} != {common}')
            common = identity
            folds[held] = fold
        base = folds[SEASONS[0]]
        save(output / 'manifest.json', dict(base, held_out_season=None, seasons=list(SEASONS),
                                            folds={held: folds[held]['records'] for held in SEASONS}))

    def required_artifacts(self, output):
        try:
            manifest = json.loads((output / 'manifest.json').read_text())
        except (OSError, ValueError):
            return ['manifest.json']
        prefix = f'{manifest["run_id"]}-s{manifest["seed"]}'
        required = ['manifest.json', 'totals.csv']
        for held in SEASONS:
            required.append(f'eval_{held}/model.pt')
            required += [f'eval_{held}/{kind}-{prefix}-{stress}.npz'
                         for kind in ('forecasts', 'evaluation-masks') for stress in self.STRESS]
        return required

    def complete(self, output):
        try:
            manifest = json.loads((output / 'manifest.json').read_text())
            return manifest.get('model') == 'B1' and sorted(manifest['seasons']) == sorted(SEASONS)
        except (KeyError, ValueError, TypeError):
            return False


from .forward import ForwardBackend

BACKENDS = {'Forward': ForwardBackend(), 'B0': B0Backend(), 'B1': B1Backend()}


def backend_for(value):
    return BACKENDS[model_of(value)]
