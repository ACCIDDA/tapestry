"""Shard an existing manager registry without changing its frozen protocol."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from tapestry.models.manager import complete_artifacts, locked, save


def read(path):
    return json.loads(path.read_text())


def prepare(folder):
    with locked(folder):
        runs = read(folder / 'runs.json')
        definition = read(folder / 'protocol.json')
        pending = []
        for key, record in runs.items():
            if record['status'] == 'complete':
                if not complete_artifacts(Path(record['output'])):
                    raise ValueError(f'Incomplete saved artifacts: {key}')
            else:
                pending.append(key)
        if not pending:
            raise ValueError('No unfinished runs to distribute')
        distributed = folder / 'distributed'
        distributed.mkdir(exist_ok=False)
        save(distributed / 'original-runs.json', runs)
        save(distributed / 'protocol.json', definition)
        save(distributed / 'tasks.json', pending)
        for task, key in enumerate(pending):
            shard = distributed / str(task) / folder.name
            shard.mkdir(parents=True)
            save(shard / 'protocol.json', definition)
            save(shard / 'runs.json', {key: runs[key]})
        print(json.dumps({'tasks': len(pending), 'reused': len(runs) - len(pending),
                          'array': f'0-{len(pending)-1}%6'}))


def worker(folder, task):
    distributed = folder / 'distributed'
    key = read(distributed / 'tasks.json')[task]
    definition = read(distributed / 'protocol.json')
    original = read(distributed / 'original-runs.json')[key]
    shard_root = distributed / str(task)
    command = [sys.executable, '-m', 'tapestry.models.manager', 'run',
               '-e', folder.name, '--root', str(shard_root),
               '--scenario', original['scenario_string'], '--seeds', str(original['seed'])]
    for option in ('dataset', 'population_file', 'frozen'):
        command.extend(('--' + option.replace('_', '-'), definition[option]))
    for option, value in definition['runtime'].items():
        command.extend(('--' + option.replace('_', '-'), str(value)))
    save(shard_root / 'allocation.json', {
        name: os.environ.get(name) for name in
        ('SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID',
         'SLURMD_NODENAME', 'CUDA_VISIBLE_DEVICES')})
    subprocess.run(command, check=True)


def merged_records(folder):
    distributed = folder / 'distributed'
    original = read(distributed / 'original-runs.json')
    definition = read(distributed / 'protocol.json')
    if read(folder / 'protocol.json') != definition:
        raise ValueError('Central protocol changed')
    merged = dict(original)
    for task, key in enumerate(read(distributed / 'tasks.json')):
        shard = distributed / str(task) / folder.name
        with locked(shard):
            if read(shard / 'protocol.json') != definition:
                raise ValueError(f'Shard {task} protocol differs')
            records = read(shard / 'runs.json')
            if set(records) != {key}:
                raise ValueError(f'Shard {task} has unexpected runs')
            record = records[key]
            for field in ('name', 'config', 'seed', 'scenario_string'):
                if record[field] != original[key][field]:
                    raise ValueError(f'Shard {task} changed {field}')
            merged[key] = record
    return merged


def collect(folder):
    # Collect after the array has ended.
    with locked(folder):
        merged = merged_records(folder)
        if set(read(folder / 'runs.json')) != set(merged):
            raise ValueError('Central registry selection changed')
        for key, record in merged.items():
            if record['status'] == 'complete' and not complete_artifacts(Path(record['output'])):
                raise ValueError(f'Missing complete artifacts: {key}')
        save(folder / 'runs.json', merged)
        incomplete = [key for key, record in merged.items() if record['status'] != 'complete']
        print(json.dumps({'complete': len(merged) - len(incomplete), 'incomplete': len(incomplete)}))
        if incomplete:
            raise ValueError('Inspect failed array tasks and retry them before comparison')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'worker', 'collect', 'status'))
    parser.add_argument('--experiment', type=Path, default=Path('data/experiments/b0-rebuilt'))
    parser.add_argument('--task', type=int)
    args = parser.parse_args()
    folder = args.experiment.resolve()
    if args.command == 'prepare':
        prepare(folder)
    elif args.command == 'worker':
        if args.task is None or args.task < 0:
            parser.error('worker requires a nonnegative --task')
        worker(folder, args.task)
    elif args.command == 'collect':
        collect(folder)
    else:
        # Atomic manager writes permit read-only monitoring while workers run.
        from collections import Counter
        distributed = folder / 'distributed'
        runs = read(distributed / 'original-runs.json')
        for task, key in enumerate(read(distributed / 'tasks.json')):
            runs[key] = read(distributed / str(task) / folder.name / 'runs.json')[key]
        print(json.dumps(dict(Counter(record['status'] for record in runs.values()))))


if __name__ == '__main__':
    main()
