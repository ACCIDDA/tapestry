"""Manage B0 experiments by name: list, plan, run, status, rank, and compare.

One `jobs.csv` row (one Slurm array task) is one scenario with all of its seeds.
Each seed attempt writes only its own folder, so tasks never share a registry;
`status` rebuilds `runs.csv` by scanning attempts. Nothing is locked: every
attempt records its git commit and whether the checkout had uncommitted changes.
"""
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys

from .quantiles import LEVELS
from .scenarios import ESSENTIAL, SUITES, TrainingScenario, get_scenarios, get_training_scenario

SEASONS = ('2023-2024', '2024-2025', '2025-2026')
JOB_FIELDS = ['task', 'name', 'scenario', 'seeds']
SLURM = ('SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID', 'SLURM_ARRAY_TASK_ID', 'SLURMD_NODENAME', 'CUDA_VISIBLE_DEVICES')
ARRAY_CHUNK = 1000
FROZEN = 'data/evaluation/b0_hub_comparison_q23'


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def git_state():
    """Commit of the checkout running this code; None outside a git checkout."""
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                                         text=True, stderr=subprocess.DEVNULL).strip()
        changes = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'],
                                          text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return dict(git_commit=None, git_dirty=None)
    return dict(git_commit=commit, git_dirty=bool(changes))


def environment():
    return dict(host=socket.gethostname(), slurm={name: os.environ.get(name) for name in SLURM}, **git_state())


def experiment_folder(root, name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', name):
        raise ValueError('Experiment name must be 1–80 letters, numbers, dots, underscores or hyphens; start with a letter/number')
    # Paths stay relative to the repository root, so folders move between machines.
    return Path(root) / name


def read_jobs(folder):
    if not (folder / 'jobs.csv').is_file():
        raise ValueError(f'No jobs.csv in {folder}; run plan first')
    with (folder / 'jobs.csv').open() as stream:
        return [dict(row, task=int(row['task']), seeds=[int(seed) for seed in row['seeds'].split()])
                for row in csv.DictReader(stream)]


def write_csv(path, rows, fieldnames):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_jobs(folder, jobs):
    write_csv(folder / 'jobs.csv', [dict(job, seeds=' '.join(map(str, job['seeds']))) for job in jobs], JOB_FIELDS)


def plan(folder, scenarios, seeds, settings):
    """Append scenarios and seeds; existing task numbers never change."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'experiment.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    changed = sorted(key for key, value in settings.items() if key in previous and previous[key] != value)
    if changed:
        print(f'Updated experiment settings {changed}; each attempt records the settings it used', flush=True)
    save(path, {**previous, **settings})
    jobs = read_jobs(folder) if (folder / 'jobs.csv').exists() else []
    by_scenario = {job['scenario']: job for job in jobs}
    for name, scenario in scenarios.items():
        key = scenario.scenario_string
        if key not in by_scenario:
            by_scenario[key] = dict(task=len(jobs), name=name, scenario=key, seeds=[])
            jobs.append(by_scenario[key])
        by_scenario[key]['seeds'] = sorted(set(by_scenario[key]['seeds']) | set(seeds))
    write_jobs(folder, jobs)
    return jobs


def check_frozen(frozen):
    """Fail before any fitting when runs could not be scored on the current quantile grid."""
    expected = [f'q{q:g}' for q in LEVELS]
    try:
        quantiles = json.loads((Path(frozen) / 'manifest.json').read_text()).get('quantiles')
    except (OSError, ValueError) as error:
        raise ValueError(f'No frozen scoring support at {frozen}; build it with scripts/b0_prepare.sbatch') from error
    if quantiles != expected:
        raise ValueError(f'{frozen} holds quantiles {quantiles}; rebuild frozen support for the {len(expected)}-level grid')


def complete_artifacts(output):
    """A top-level scores file alone is insufficient evidence of a complete, scored CV."""
    required = ['manifest.json', 'scores.csv', 'totals.csv']
    required += [f'eval_{season}/{name}' for season in SEASONS
                 for name in ('model.pt', 'forecasts.npz', 'training.json', 'scores.csv')]
    if not all((output / name).is_file() and (output / name).stat().st_size for name in required):
        return False
    try:
        manifest = json.loads((output / 'manifest.json').read_text())
        return sorted(f['eval_season'] for f in manifest['folds']) == sorted(SEASONS)
    except (KeyError, ValueError, TypeError):
        return False


def attempts(folder, scenario, seed):
    return sorted((folder / scenario / f's{seed}').glob('attempt-*'))


def seed_state(folder, scenario, seed):
    """(attempt, record, done): the latest complete attempt, else the latest attempt."""
    found = []
    for attempt in attempts(folder, scenario, seed):
        try:
            found.append((attempt, json.loads((attempt / 'run.json').read_text())))
        except (OSError, ValueError):
            found.append((attempt, dict(status='unknown')))
    for attempt, record in reversed(found):
        if record.get('status') == 'complete' and complete_artifacts(attempt / 'cv'):
            return attempt, record, True
    return found[-1] + (False,) if found else (None, dict(status='planned'), False)


def execute(command, log):
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def run_seed(folder, job, seed, settings):
    if seed_state(folder, job['scenario'], seed)[2]:
        print(f'Reusing {job["name"]}, seed {seed}', flush=True)
        return True
    number = len(attempts(folder, job['scenario'], seed)) + 1
    attempt = folder / job['scenario'] / f's{seed}' / f'attempt-{number:03d}'
    # A duplicate task running the same seed concurrently fails here instead of sharing a folder.
    attempt.mkdir(parents=True, exist_ok=False)
    output = attempt / 'cv'
    scenario = TrainingScenario.from_string(job['scenario'])
    command = [sys.executable, '-m', 'tapestry.models.season_cv',
               '--dataset', settings['dataset'], '--population-file', settings['population_file'],
               '--eval-members', str(settings['eval_members']), '--device', settings['device'],
               '--seed', str(seed), '--output', str(output), *scenario.flags()]
    scoring = [sys.executable, '-m', 'tapestry.evaluation.totals', 'score',
               '--run', str(output), '--frozen', settings['frozen']]
    record = dict(status='running', name=job['name'], scenario=job['scenario'], seed=seed,
                  config=asdict(scenario), settings=settings, command=command, scoring_command=scoring,
                  started=now(), **environment())
    save(attempt / 'run.json', record)
    print(f'Running {job["name"]}, seed {seed}: {attempt}', flush=True)
    try:
        with (attempt / 'run.log').open('w') as log:
            execute(command, log)
            execute(scoring, log)
        if not complete_artifacts(output):
            raise RuntimeError('CV exited without all three fitted-fold artifacts and totals')
        manifest = json.loads((output / 'manifest.json').read_text())
        manifest.update(scenario_string=job['scenario'], scenario_name=job['name'])
        save(output / 'manifest.json', manifest)
    except (Exception, KeyboardInterrupt) as error:
        record.update(status='failed', error=str(error), finished=now())
        save(attempt / 'run.json', record)
        if isinstance(error, KeyboardInterrupt):
            raise
        print(f'Failed {job["name"]}, seed {seed}: {error}', flush=True)
        return False
    record.update(status='complete', finished=now())
    save(attempt / 'run.json', record)
    print(f'Completed {job["name"]}, seed {seed}', flush=True)
    return True


def run(folder, tasks=None, device=None, keep_going=False):
    settings = json.loads((folder / 'experiment.json').read_text())
    if device:
        settings['device'] = device
    check_frozen(settings['frozen'])
    jobs = read_jobs(folder)
    if tasks is not None:
        unknown = set(tasks) - {job['task'] for job in jobs}
        if unknown:
            raise ValueError(f'Unknown tasks {sorted(unknown)} in {folder / "jobs.csv"}')
        jobs = [job for job in jobs if job['task'] in tasks]
    failures = 0
    for job in jobs:
        for seed in job['seeds']:
            if not run_seed(folder, job, seed, settings):
                failures += 1
                if not keep_going:
                    return failures
    return failures


def collect(folder):
    """Rebuild runs.csv from attempt folders. A 'running' status may be a killed job."""
    rows = []
    for job in read_jobs(folder):
        scenario = TrainingScenario.from_string(job['scenario'])
        for seed in job['seeds']:
            attempt, record, done = seed_state(folder, job['scenario'], seed)
            status = 'complete' if done else 'incomplete' if record.get('status') == 'complete' else record.get('status')
            rows.append(dict(task=job['task'], name=job['name'], seed=seed, status=status,
                             attempt=attempt.relative_to(folder).as_posix() if attempt else '',
                             attempts=len(attempts(folder, job['scenario'], seed)),
                             git_commit=record.get('git_commit'), git_dirty=record.get('git_dirty'),
                             started=record.get('started'), finished=record.get('finished'),
                             scenario=job['scenario'], **asdict(scenario)))
    if rows:
        write_csv(folder / 'runs.csv', rows, list(rows[0]))
    return rows


def pending_tasks(rows):
    return sorted({row['task'] for row in rows if row['status'] != 'complete'})


def ranges(values):
    """Compact Slurm array syntax: [0, 1, 2, 5] -> '0-2,5'."""
    spans = []
    for value in sorted(values):
        if spans and value == spans[-1][1] + 1:
            spans[-1][1] = value
        else:
            spans.append([value, value])
    return ','.join(str(a) if a == b else f'{a}-{b}' for a, b in spans)


def array_commands(tasks, experiment, chunk=ARRAY_CHUNK):
    """Shared-partition submissions, chunked so array indices stay below `chunk`."""
    commands = []
    for offset in sorted({task // chunk * chunk for task in tasks}):
        indices = [task - offset for task in tasks if offset <= task < offset + chunk]
        commands.append(f'sbatch --array={ranges(indices)} --export=ALL,OFFSET={offset} scripts/b0_sweep.sbatch {experiment}')
    return commands


def completed_runs(folder, allow_incomplete):
    rows = collect(folder)
    done = [row for row in rows if row['status'] == 'complete']
    if not done or (len(done) < len(rows) and not allow_incomplete):
        raise ValueError(f'{len(rows) - len(done)} of {len(rows)} runs incomplete; finish them or pass --allow-incomplete')
    versions = {(row['git_commit'], row['git_dirty']) for row in done}
    if len(versions) > 1 or any(row['git_dirty'] is not False for row in done):
        print(f'Warning: runs come from {len(versions)} commit/dirty states: {sorted(versions, key=str)}', flush=True)
    return done, versions


def rank(folder, allow_incomplete=False):
    """Rank completed runs by total-WIS ratios to the ensemble (`tapestry.evaluation.totals`)."""
    from tapestry.evaluation.totals import rank as rank_runs
    done, _ = completed_runs(folder, allow_incomplete)
    attempts_used = sorted(row['attempt'] for row in done)
    # Different ranked sets get different destinations; never mix partial rankings.
    destination = folder / f'ranking-{hashlib.sha256(json.dumps(attempts_used).encode()).hexdigest()[:12]}'
    runs = [dict(config_id=row['scenario'], name=row['name'], seed=row['seed'], path=folder / row['attempt'] / 'cv')
            for row in sorted(done, key=lambda row: row['attempt'])]
    ranking = rank_runs(runs, destination)
    print(ranking.head(20).to_string(index=False), flush=True)
    return destination


def compare(folder, workers=2, allow_incomplete=False):
    """Score completed runs through the existing EpiBench sweep; warn when code versions differ."""
    settings = json.loads((folder / 'experiment.json').read_text())
    done, versions = completed_runs(folder, allow_incomplete)
    runs = sorted(row['attempt'] for row in done)
    # Different compared sets get different destinations; never mix partial rankings.
    destination = folder / f'comparison-{hashlib.sha256(json.dumps(runs).encode()).hexdigest()[:12]}'
    command = [sys.executable, '-m', 'tapestry.evaluation.sweep', '--runs', *[str(folder / run / 'cv') for run in runs],
               '--frozen', settings['frozen'], '--output', str(destination), '--score-workers', str(workers)]
    record = dict(status='running', command=command, output=destination.name, runs=len(runs),
                  run_versions=sorted(map(list, versions), key=str), started=now(), **environment())
    save(folder / 'comparison.json', record)
    try:
        subprocess.run(command, check=True)
    except (Exception, KeyboardInterrupt) as error:
        record.update(status='failed', error=str(error), finished=now())
        save(folder / 'comparison.json', record)
        raise
    record.update(status='complete', finished=now())
    save(folder / 'comparison.json', record)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['list', 'plan', 'run', 'status', 'rank', 'compare'])
    parser.add_argument('-e', '--experiment', help='Persistent experiment name')
    parser.add_argument('--root', default='data/experiments')
    parser.add_argument('--suite', choices=list(SUITES), default='essential')
    parser.add_argument('-s', '--scenario', nargs='+', help='Named aliases or full scenario strings; overrides suite')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    parser.add_argument('--dataset', default='data/processed/build_b_finalized.npz')
    parser.add_argument('--population-file', default='data/metadata/b0_locations.csv')
    parser.add_argument('--frozen', default=FROZEN, help='Frozen ensemble-supported tasks on the 23-quantile grid')
    parser.add_argument('--eval-members', type=int, default=2048)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'],
                        help='plan: saved default (cpu); run: override for this invocation')
    parser.add_argument('-t', '--task', nargs='+', type=int, help='run: jobs.csv task numbers (default: all)')
    parser.add_argument('--keep-going', action='store_true', help='run: continue with remaining seeds after a failure')
    parser.add_argument('--workers', type=int, default=2, help='compare: concurrent EpiBench cases')
    parser.add_argument('--allow-incomplete', action='store_true', help='rank/compare: use only completed runs')
    args = parser.parse_args(argv)
    if args.eval_members < 1 or args.workers < 1:
        parser.error('eval-members and workers must be positive')
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error('Seeds must be distinct nonnegative integers')
    if args.command in ('list', 'plan'):
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
    if args.command == 'plan':
        settings = dict(dataset=args.dataset, population_file=args.population_file, frozen=args.frozen,
                        eval_members=args.eval_members, device=args.device or 'cpu')
        plan(folder, scenarios, args.seeds, settings)
        print(json.dumps(dict(experiment=str(folder), **counts)), flush=True)
    elif args.command == 'run':
        if run(folder, args.task, args.device, args.keep_going):
            raise SystemExit(1)
        return
    elif args.command == 'rank':
        print(rank(folder, args.allow_incomplete))
        return
    elif args.command == 'compare':
        print(compare(folder, args.workers, args.allow_incomplete))
        return
    rows = collect(folder)
    if args.command == 'status':
        for row in rows:
            print(f'{row["status"]}\t{row["task"]}\t{row["name"]}\ts{row["seed"]}\t{row["attempt"]}')
        print(json.dumps({status: sum(row['status'] == status for row in rows) for status in sorted({r['status'] for r in rows})}))
    pending = pending_tasks(rows)
    if pending:
        root = '' if args.root == 'data/experiments' else f' --root {args.root}'
        print('Pending tasks (check squeue -a before resubmitting). Sweep launcher:')
        for command in array_commands(pending, args.experiment):
            print(command + root)
        if max(pending) < ARRAY_CHUNK:
            print(f'Partition jlessler: sbatch --array={ranges(pending)}%6 scripts/b0_array.sbatch {args.experiment}{root}')
        print(f'Locally: python -m tapestry.models.manager run -e {args.experiment}{root}')


if __name__ == '__main__':
    main()
