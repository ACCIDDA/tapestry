"""Manage B0 experiments by name: list, plan, run/resume, status, and compare."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from .scenarios import ESSENTIAL, get_scenarios, get_training_scenario

SEASONS = ('2023-2024', '2024-2025', '2025-2026')


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def frozen_hashes(folder):
    folder = Path(folder)
    return {str(p.relative_to(folder)): sha256(p) for p in sorted(folder.rglob('*'))
            if p.name in ('manifest.json', 'units.parquet', 'quantiles.parquet')}


def experiment_folder(root, name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', name):
        raise ValueError('Experiment name must be 1–80 letters, numbers, dots, underscores or hyphens; start with a letter/number')
    return Path(root).resolve() / name


@contextmanager
def locked(folder):
    """One runner per experiment; OS releases the lock even after a killed run."""
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another manager is writing {folder}') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def protocol(args):
    package = Path(__file__).parents[1]
    files = [p for directory in ('models', 'model_data', 'evaluation')
             for p in (package / directory).rglob('*') if p.suffix in ('.py', '.R')]
    return {
        'schema': 1,
        'dataset': str(Path(args.dataset).resolve()), 'dataset_sha256': sha256(args.dataset),
        'population_file': str(Path(args.population_file).resolve()),
        'population_sha256': sha256(args.population_file),
        'frozen': str(Path(args.frozen).resolve()),
        'frozen_files_sha256': frozen_hashes(args.frozen),
        'code_sha256': {str(p.relative_to(package)): sha256(p) for p in sorted(files)},
        'runtime': {key: getattr(args, key) for key in
                    ('epochs', 'width', 'batch_size', 'members', 'eval_members', 'lr', 'device')},
        'seasons': list(SEASONS),
        'assumptions': [
            'Exploratory finalized-data leave-one-season-out CV; these seasons have already been examined.',
            'Each scenario/seed trains three folds; all seeds run every selected scenario.',
            'Native-unit loss Q95 is fitted per fold independently of task weights.',
            'Five saved quantiles follow the current repository scoring protocol.',
            'No calibration or early stopping. Any future calibration requires inner out-of-sample predictions and refitted scalers.',
            'B0 remains local. Spatial attention is a separate B1 experiment.',
            'Failed/interrupted attempts restart all three folds; previous artifacts are retained.',
        ],
    }


def prepare(folder, definition, scenarios, seeds):
    path = folder / 'protocol.json'
    if path.exists() and json.loads(path.read_text()) != definition:
        raise ValueError('Experiment protocol/code/data changed; use a new experiment name')
    save(path, definition)
    registry = folder / 'runs.json'
    runs = json.loads(registry.read_text()) if registry.exists() else {}
    selected = []
    for name, scenario in scenarios.items():
        for seed in seeds:
            key = f'{scenario.scenario_string}::s{seed}'
            selected.append(key)
            if key not in runs:
                runs[key] = dict(name=name, scenario_string=scenario.scenario_string,
                                 config=asdict(scenario), seed=seed, status='planned', attempts=[])
    save(registry, runs)
    return runs, list(dict.fromkeys(selected))


def complete_artifacts(output):
    """A top-level scores file alone is insufficient evidence of a complete CV."""
    required = ['manifest.json', 'scores.csv']
    required += [f'eval_{season}/{name}' for season in SEASONS
                 for name in ('model.pt', 'forecasts.npz', 'training.json', 'scores.csv')]
    if not all((output / name).is_file() and (output / name).stat().st_size for name in required):
        return False
    try:
        manifest = json.loads((output / 'manifest.json').read_text())
        return sorted(f['eval_season'] for f in manifest['folds']) == sorted(SEASONS)
    except (KeyError, ValueError, TypeError):
        return False


def execute(folder, definition, runs, selected, keep_going=False):
    failures = []
    for key in selected:
        record = runs[key]
        if record['status'] == 'complete':
            if not complete_artifacts(Path(record['output'])):
                raise ValueError(f'Completed artifacts missing or incomplete: {key}')
            print(f'Reusing {record["name"]}, seed {record["seed"]}', flush=True)
            continue
        # A stale running state indicates the previous manager exited before saving
        # its final status. Preserve that attempt and start a fresh CV directory.
        for previous in record['attempts']:
            if previous['status'] == 'running':
                previous.update(status='interrupted', finished=now())
        attempt = folder / key / f'attempt-{len(record["attempts"]) + 1:03d}'
        attempt.mkdir(parents=True, exist_ok=False)
        output = attempt / 'cv'
        command = [sys.executable, '-m', 'tapestry.models.season_cv',
                   '--dataset', definition['dataset'], '--population-file', definition['population_file'],
                   '--output', str(output), '--seed', str(record['seed'])]
        for option, value in definition['runtime'].items():
            command.extend(('--' + option.replace('_', '-'), str(value)))
        command.extend(get_training_scenario(record['scenario_string']).flags())
        current = dict(status='running', started=now(), command=command, output=str(output), log=str(attempt / 'run.log'))
        record['attempts'].append(current)
        record.update(status='running', output=str(output))
        save(folder / 'runs.json', runs)
        save(attempt / 'run.json', dict(scenario=record['config'], seed=record['seed'], **current))
        print(f'Running {record["name"]}, seed {record["seed"]}: {attempt}', flush=True)
        try:
            with (attempt / 'run.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            if not complete_artifacts(output):
                raise RuntimeError('CV exited without all three fitted-fold artifacts')
            manifest = json.loads((output / 'manifest.json').read_text())
            manifest['scenario_string'] = record['scenario_string']
            manifest['scenario_name'] = record['name']
            save(output / 'manifest.json', manifest)
        except (Exception, KeyboardInterrupt) as error:
            current.update(status='failed', error=str(error), finished=now())
            record['status'] = 'failed'
            save(folder / 'runs.json', runs)
            save(attempt / 'run.json', current)
            failures.append(key)
            if not keep_going or isinstance(error, KeyboardInterrupt):
                raise
        else:
            current.update(status='complete', finished=now())
            record['status'] = 'complete'
            save(attempt / 'run.json', current)
            save(folder / 'runs.json', runs)
            print(f'Completed {record["name"]}, seed {record["seed"]}', flush=True)
    return failures


def compare(folder):
    """Use the existing EpiBench pipeline and all registered completed runs."""
    definition = json.loads((folder / 'protocol.json').read_text())
    if frozen_hashes(definition['frozen']) != definition['frozen_files_sha256']:
        raise ValueError('Frozen scoring inputs changed since experiment planning')
    package = Path(__file__).parents[1]
    for name, checksum in definition['code_sha256'].items():
        if name.startswith('evaluation/') and sha256(package / name) != checksum:
            raise ValueError('Evaluation code changed since experiment planning')
    runs = json.loads((folder / 'runs.json').read_text())
    incomplete = [key for key, record in runs.items() if record['status'] != 'complete']
    if not runs or incomplete:
        raise ValueError(f'Complete all registered runs before comparing ({len(incomplete)} incomplete)')
    outputs = [record['output'] for record in runs.values()]
    if not all(complete_artifacts(Path(output)) for output in outputs):
        raise ValueError('Missing completed CV artifacts')
    # Different compared sets get different destinations; never mix partial rankings.
    signature = hashlib.sha256(json.dumps(sorted(outputs)).encode()).hexdigest()[:12]
    destination = folder / f'comparison-{signature}'
    command = [sys.executable, '-m', 'tapestry.evaluation.sweep', '--runs', *outputs,
               '--frozen', definition['frozen'], '--output', str(destination)]
    save(folder / 'comparison.json', dict(status='running', command=command, output=str(destination)))
    try:
        subprocess.run(command, check=True)
    except (Exception, KeyboardInterrupt) as error:
        save(folder / 'comparison.json', dict(status='failed', command=command, error=str(error), output=str(destination)))
        raise
    save(folder / 'comparison.json', dict(status='complete', command=command, output=str(destination)))
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['list', 'plan', 'run', 'status', 'compare'])
    parser.add_argument('-e', '--experiment', help='Persistent experiment name')
    parser.add_argument('--root', default='data/experiments')
    parser.add_argument('--suite', choices=['essential', 'grid'], default='essential')
    parser.add_argument('-s', '--scenario', nargs='+', help='Named aliases or full scenario strings; overrides suite')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--dataset', default='data/processed/build_b_finalized.npz')
    parser.add_argument('--population-file', default='data/metadata/b0_locations.csv')
    parser.add_argument('--frozen', default='data/evaluation/b0_hub_comparison')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--members', type=int, default=8)
    parser.add_argument('--eval-members', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--keep-going', action='store_true', help='Run remaining scenarios after a failed CV')
    args = parser.parse_args(argv)
    if min(args.epochs, args.width, args.batch_size, args.eval_members) < 1 or args.members < 2 or not 0 < args.lr < float('inf'):
        parser.error('Positive runtime dimensions/lr required; training members >=2')
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error('Seeds must be distinct nonnegative integers')
    scenarios = ({name: get_training_scenario(name) for name in args.scenario}
                 if args.scenario else get_scenarios(args.suite))
    count = len(set(scenarios.values()))
    counts = dict(configurations=count, seeds=len(args.seeds), runs=count * len(args.seeds),
                  season_fits=count * len(args.seeds) * len(SEASONS))
    if args.command == 'list':
        for name, scenario in scenarios.items():
            info = ESSENTIAL.get(name)
            print(f'{name}\t{scenario.scenario_string}' + (f'\tcontrol={info[1]}; {info[2]}' if info else ''))
        print(json.dumps(counts))
        return
    if not args.experiment:
        parser.error('--experiment is required')
    folder = experiment_folder(args.root, args.experiment)
    if args.command == 'status':
        runs = json.loads((folder / 'runs.json').read_text())
        for key, record in runs.items():
            print(f'{record["status"]}\t{record["name"]}\t{key}')
        print(json.dumps({status: sum(r['status'] == status for r in runs.values())
                          for status in ('planned', 'running', 'failed', 'complete')}))
        return
    with locked(folder):
        if args.command == 'compare':
            print(compare(folder))
            return
        if not (Path(args.frozen) / 'manifest.json').is_file():
            parser.error('Frozen comparison manifest is required')
        definition = protocol(args)
        runs, selected = prepare(folder, definition, scenarios, args.seeds)
        print(json.dumps(dict(experiment=str(folder), **counts)), flush=True)
        if args.command == 'run':
            if execute(folder, definition, runs, selected, args.keep_going):
                raise SystemExit(1)


if __name__ == '__main__':
    main()
