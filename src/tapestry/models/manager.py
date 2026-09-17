"""Manage B0 and B1 experiments: list, plan, run, status, rank, and compare.

One `jobs.csv` row (one Slurm array task) is one scenario with all of its seeds.
Each seed attempt writes only its own folder, so tasks never share a registry;
`status` rebuilds `runs.csv` by scanning attempts. Nothing is locked: every
attempt records its git commit and whether the checkout had uncommitted changes.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import threading

from .backends import BACKENDS, DATASETS, EVAL_MEMBERS, FROZEN, LOCATIONS, backend_for, model_of
from .provenance import SEASONS, environment, git_state, now, save
from .quantiles import LEVELS
from .scenarios import ESSENTIAL, SUITES, TrainingScenario

JOB_FIELDS = ['task', 'name', 'scenario', 'seeds']
ARRAY_CHUNK = 1000


def parse_scenario(value):
    if value.startswith('b1:'):
        from .b1_scenarios import B1Scenario
        return B1Scenario.from_string(value)
    return TrainingScenario.from_string(value)


def scenario_directory(value):
    # Complete B1 strings exceed a filesystem component's 255-byte limit.
    return parse_scenario(value).run_id if value.startswith('b1:') else value


def output_directory(scenario):
    return backend_for(scenario).output_dir


def check_inputs(settings):
    backend_for(settings).check_inputs(settings)


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
    if previous and model_of(previous) != model_of(settings):
        raise ValueError('Use separate experiment names for B0 and B1')
    changed = sorted(key for key, value in settings.items() if key in previous and previous[key] != value)
    # Data, scoring support and protocol define what the scores mean; changing one
    # mid-experiment would silently pool incomparable runs. Device is a machine detail.
    protocol = set(changed) - {'device', 'suite'}
    if protocol:
        raise ValueError(f'Experiment settings changed: {sorted(protocol)}; use a new experiment name')
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


def complete_artifacts(output, backend=None):
    """A top-level scores file alone is insufficient evidence of a complete, scored CV."""
    backend = backend or next(b for b in BACKENDS.values() if b.output_dir == output.name)
    required = backend.required_artifacts(output)
    if not all((output / name).is_file() and (output / name).stat().st_size for name in required):
        return False
    return backend.complete(output)


def attempts(folder, scenario, seed):
    return sorted((folder / scenario_directory(scenario) / f's{seed}').glob('attempt-*'))


def seed_state(folder, scenario, seed):
    """(attempt, record, done): the latest complete attempt, else the latest attempt."""
    found = []
    for attempt in attempts(folder, scenario, seed):
        try:
            found.append((attempt, json.loads((attempt / 'run.json').read_text())))
        except (OSError, ValueError):
            found.append((attempt, dict(status='unknown')))
    backend = backend_for(scenario)
    for attempt, record in reversed(found):
        if record.get('status') == 'complete' and complete_artifacts(attempt / backend.output_dir, backend):
            return attempt, record, True
    return found[-1] + (False,) if found else (None, dict(status='planned'), False)


def execute(command, log):
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)


def run_seed(folder, job, seed, settings):
    if seed_state(folder, job['scenario'], seed)[2]:
        print(f'Reusing {job["name"]}, seed {seed}', flush=True)
        return True
    number = len(attempts(folder, job['scenario'], seed)) + 1
    backend = backend_for(job['scenario'])
    attempt = folder / scenario_directory(job['scenario']) / f's{seed}' / f'attempt-{number:03d}'
    # A duplicate task running the same seed concurrently fails here instead of sharing a folder.
    attempt.mkdir(parents=True, exist_ok=False)
    output = attempt / backend.output_dir
    scenario = parse_scenario(job['scenario'])
    # `command` is a list of fitting commands, one per leave-one-season-out fold.
    command = backend.fit_commands(scenario, seed, settings, output)
    # One scorer for every model: ensemble-relative WIS on the frozen Hub tasks.
    scoring = [sys.executable, '-m', 'tapestry.evaluation.totals', 'score',
               '--run', str(output), '--frozen', settings['frozen']]
    record = dict(status='running', name=job['name'], scenario=job['scenario'], seed=seed,
                  config=asdict(scenario), settings=settings, command=command, scoring_command=scoring,
                  started=now(), **environment())
    save(attempt / 'run.json', record)
    print(f'Running {job["name"]}, seed {seed}: {attempt}', flush=True)
    try:
        with (attempt / 'run.log').open('w') as log:
            for fit in command:
                execute(fit, log)
            # The run-level manifest must exist before scoring: the scorer reads
            # its `model` field to choose the forecast exporter.
            backend.collect_manifest(output)
            execute(scoring, log)
        if not complete_artifacts(output, backend):
            raise RuntimeError('Run exited without its complete fitted and scored artifacts')
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


def run(folder, tasks=None, device=None, keep_going=False, fit_workers=1):
    """Fit the selected tasks, up to `fit_workers` CONFIGURATIONS at a time.

    A worker owns one configuration and fits its seeds in sequence, so seeds of a
    configuration never overlap while `fit_workers` different configurations run
    side by side on one card. This is the shape that keeps a GPU busy: a task has
    only three seeds, so parallelising seeds caps concurrency at three, while
    parallelising configurations has no such ceiling.

    One fit uses a small fraction of a GPU (28k-142k parameters, batch 8), so the
    card idles between kernel launches and several fits share it well. Each seed is
    a separate subprocess writing its own attempt folder, created with
    `exist_ok=False`, so concurrent fits cannot share state or race for a folder.
    CPU is the binding resource: each fit pins two torch threads, so keep
    `fit_workers` at or below half the allocated cores. GPU memory is the other
    limit: the heaviest configuration peaks near 6 GiB, so ~6 lanes fit a 44 GiB L40.
    """
    settings = json.loads((folder / 'experiment.json').read_text())
    if device:
        settings['device'] = device
    check_inputs(settings)
    jobs = read_jobs(folder)
    if tasks is not None:
        unknown = set(tasks) - {job['task'] for job in jobs}
        if unknown:
            raise ValueError(f'Unknown tasks {sorted(unknown)} in {folder / "jobs.csv"}')
        jobs = [job for job in jobs if job['task'] in tasks]
    if fit_workers < 1:
        raise ValueError('fit_workers must be positive')
    # One unit of work is a whole configuration, not a seed: that is what keeps a
    # configuration's seeds in sequence while several configurations run at once.
    # The pool's max_workers does the throttling: submit everything and let it queue.
    # Without --keep-going a failure stops configurations that have not started;
    # lanes already running finish their current seed, and a skipped seed stays
    # 'planned' for the next run to pick up.
    failures, stop = 0, threading.Event()

    def fit(job):
        """Fit one configuration's seeds in order; returns the number that failed."""
        failed = 0
        for seed in job['seeds']:
            if stop.is_set():
                break
            if not run_seed(folder, job, seed, settings):
                failed += 1
                if not keep_going:
                    stop.set()
                    break
        return failed

    with ThreadPoolExecutor(max_workers=fit_workers) as pool:
        for future in as_completed([pool.submit(fit, job) for job in jobs]):
            failures += future.result()
    return failures


def collect(folder):
    """Rebuild runs.csv from attempt folders. A 'running' status may be a killed job."""
    rows = []
    for job in read_jobs(folder):
        scenario = parse_scenario(job['scenario'])
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
    """Rank completed runs by season-equal location-relative WIS, B0 or B1.

    A B1 experiment additionally gets a nowcast ranking under `nowcast/`, scored
    against preliminary-value persistence. Forecast and nowcast scores share
    weights but not support and are never combined.
    """
    from tapestry.evaluation.totals import SCORE_VERSION, rank as rank_runs
    from tapestry.evaluation import nowcast
    settings = json.loads((folder / 'experiment.json').read_text())
    backend = backend_for(settings)
    done, _ = completed_runs(folder, allow_incomplete)
    attempts_used = sorted(row['attempt'] for row in done)
    # Different ranked sets get different destinations; never mix partial rankings.
    fingerprint = dict(attempts=attempts_used, score_version=SCORE_VERSION)
    destination = folder / f'ranking-{hashlib.sha256(json.dumps(fingerprint).encode()).hexdigest()[:12]}'
    runs = [dict(config_id=row['scenario'], name=row['name'], seed=row['seed'],
                 path=folder / row['attempt'] / backend.output_dir)
            for row in sorted(done, key=lambda row: row['attempt'])]
    ranking = rank_runs(runs, destination)
    print(ranking.head(20).to_string(index=False), flush=True)
    nowcasts = nowcast.rank(runs, destination / 'nowcast')
    if nowcasts is not None:
        print(f'\nNowcast, relative to {nowcast.BASELINE}:', flush=True)
        print(nowcasts.head(20).to_string(index=False), flush=True)
    return destination


def compare(folder, workers=2, allow_incomplete=False):
    """Score completed runs through the existing EpiBench sweep; warn when code versions differ."""
    settings = json.loads((folder / 'experiment.json').read_text())
    backend = backend_for(settings)
    done, versions = completed_runs(folder, allow_incomplete)
    runs = sorted(row['attempt'] for row in done)
    # Different compared sets get different destinations; never mix partial rankings.
    destination = folder / f'comparison-{hashlib.sha256(json.dumps(runs).encode()).hexdigest()[:12]}'
    command = [sys.executable, '-m', 'tapestry.evaluation.sweep',
               '--runs', *[str(folder / run / backend.output_dir) for run in runs],
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
    parser.add_argument('command', choices=['list', 'plan', 'run', 'status', 'rank', 'compare', 'decisive'])
    parser.add_argument('-e', '--experiment', help='Persistent experiment name')
    parser.add_argument('--root', default='data/experiments')
    parser.add_argument('--suite', choices=[*SUITES, 'B1', *BACKENDS['B1'].SUITES], default='essential')
    parser.add_argument('-s', '--scenario', nargs='+', help='Named aliases or full scenario strings; overrides suite')
    parser.add_argument('--seeds', nargs='+', type=int, default=None)
    parser.add_argument('--dataset')
    parser.add_argument('--population-file')
    parser.add_argument('--frozen', help='Frozen ensemble-supported tasks on the 23-quantile grid')
    parser.add_argument('--eval-members', type=int)
    BACKENDS['B1'].add_plan_args(parser)
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'],
                        help='plan: saved default (cpu); run: override for this invocation')
    parser.add_argument('-t', '--task', nargs='+', type=int, help='run: jobs.csv task numbers (default: all)')
    parser.add_argument('--keep-going', action='store_true', help='run: continue with remaining seeds after a failure')
    parser.add_argument('--workers', type=int, default=2, help='compare: concurrent EpiBench cases')
    parser.add_argument('--fit-workers', type=int, default=1,
                        help='run: configurations fitted concurrently on one GPU, each running its '
                             'seeds in sequence; every fit pins two torch threads')
    parser.add_argument('--report-output', default='data/experiments/B1-decisive-report',
                        help='decisive: output for paired seeds, mixtures, calibration and uncertainty')
    parser.add_argument('--allow-incomplete', action='store_true', help='rank/compare: use only completed runs')
    args = parser.parse_args(argv)
    b1_suites = {'B1', *BACKENDS['B1'].SUITES}
    model = 'B1' if args.suite in b1_suites or (args.scenario and all(s.startswith('b1:') for s in args.scenario)) else 'B0'
    backend = BACKENDS[model]
    args.dataset = args.dataset or DATASETS[model]
    args.population_file = args.population_file or LOCATIONS
    args.eval_members = args.eval_members if args.eval_members is not None else EVAL_MEMBERS[model]
    # Both models are ranked on the frozen ensemble-supported tasks, so the
    # frozen support is required for either; a run that cannot be scored failed.
    args.frozen = args.frozen or FROZEN
    if args.seeds is None:
        if args.suite == 'B0.1':
            from .b01_suite import expand
            args.seeds = expand()[0]['seeds']
        else:
            args.seeds = [42, 43, 44]
    if args.eval_members < 1 or args.workers < 1:
        parser.error('eval-members and workers must be positive')
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error('Seeds must be distinct nonnegative integers')
    if args.command in ('list', 'plan'):
        scenarios = backend.scenarios(args)
        count = len(set(scenarios.values()))
        # Both models fit one run per configuration/seed and one process per fold.
        counts = dict(configurations=count, seeds=len(args.seeds), runs=count * len(args.seeds),
                      season_fits=count * len(args.seeds) * len(SEASONS))
        if model == 'B1':
            # A B1 fold fits one model per independently fitted component group.
            counts['component_fits'] = sum({'all': 1, 'pathogen': 3, 'target': 6}[s.fit_partition]
                                           for s in set(scenarios.values())) * len(args.seeds) * len(SEASONS)
    if args.command == 'list':
        for name, scenario in scenarios.items():
            info = ESSENTIAL.get(name)
            print(f'{name}\t{scenario.scenario_string}' + (f'\tcontrol={info[1]}; {info[2]}' if info else ''))
        print(json.dumps(counts))
        return
    if not args.experiment:
        parser.error('--experiment is required')
    folder = experiment_folder(args.root, args.experiment)
    if args.command == 'decisive':
        from tapestry.evaluation.decisive import compare as decisive_compare
        decisive_compare(Path(args.root), Path(args.report_output), args.device or 'cpu')
        return
    if args.command == 'plan':
        settings = dict(dataset=args.dataset, population_file=args.population_file, frozen=args.frozen,
                        eval_members=args.eval_members, device=args.device or 'cpu',
                        suite=args.suite, **backend.settings(args))
        check_inputs(settings)
        plan(folder, scenarios, args.seeds, settings)
        backend.prepare(folder, settings)
        print(json.dumps(dict(experiment=str(folder), **counts)), flush=True)
    elif args.command == 'run':
        if run(folder, args.task, args.device, args.keep_going, args.fit_workers):
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
        # A source snapshot means the shared dispatcher drains one queue across
        # the patron GPUs, rather than Slurm slicing static array tasks.
        if (folder / 'code').is_dir():
            settings = json.loads((folder / 'experiment.json').read_text())
            if settings.get('device') == 'cpu':
                print(f'CPU launch: sbatch --job-name={args.experiment} scripts/cpu.sbatch {args.experiment}')
                print(f'Locally: python -m tapestry.models.manager run -e {args.experiment}{root} --device cpu')
                return
            print(f'Shared GPU queue: sbatch --job-name={args.experiment} --array=0-3 '
                  f'scripts/jlessler.sbatch {args.experiment}')
            print(f'Locally: python -m tapestry.models.manager run -e {args.experiment}{root}')
            return
        print('Pending tasks (check squeue -a before resubmitting). Sweep launcher:')
        for command in array_commands(pending, args.experiment):
            print(command + root)
        if max(pending) < ARRAY_CHUNK:
            print(f'Partition jlessler: sbatch --array={ranges(pending)}%6 scripts/b0_array.sbatch {args.experiment}{root}')
        print(f'Locally: python -m tapestry.models.manager run -e {args.experiment}{root}')


if __name__ == '__main__':
    main()
