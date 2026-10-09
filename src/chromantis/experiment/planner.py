"""Plan, run, resume, replay and rank experiments: one manager for every scenario.

`plan` takes scenario strings (`-s`) or a study file (`--study experiments/<name>.json`:
{"description": ..., "candidates": {name: scenario}, "seeds": [...]}). `run` fits each
scenario/seed with `experiment/fit.py` across its held-out seasons; a seed whose last
attempt stopped (time limit, failure) with saved fold models is continued in that attempt
(`fit --resume`), and the continuation command is recorded in its `run.json`. `replay`
evaluates completed fits on other inputs without refitting. `rank` scores runs with
`evaluation/ranking.py` and, for a complete experiment, writes the report page
(`evaluation/report.py`).

`plan` records in `experiment.json` the dataset path, the frozen-support path and
the sha256 of `panel.npz`, of the frozen manifest and of the one population file
(`LOCATIONS`); `run` (and the Slurm dispatcher) refuse to fit when any hash no
longer matches. The Slurm launcher runs the code snapshot pinned by `plan`
(`<experiment>/code`); a local `planner run` runs the working tree.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
import shutil
from pathlib import Path
import subprocess
import sys
import threading

from chromantis.dataset.build import PANEL_DATASET
from chromantis.model.scenario import Scenario
from .fit import fit, replay, LOCATIONS
from .provenance import save, now, environment, git_state, sha256

# Forecast samples per episode for every reported score (user decision 2026-10-05), so
# extreme quantiles have the same sampling noise in every experiment.
EVAL_MEMBERS = 512

JOB_FIELDS = ['task', 'name', 'scenario', 'seeds']
FROZEN = 'data/evaluation/b0_hub_comparison_q23'


def pinned_inputs(settings):
    """Hashes of the dataset, frozen-support manifest and population file an experiment was planned against.

    The population file (`LOCATIONS`) and the frozen support are git-ignored, not
    synced with the code: copy them to the cluster (docs/longleaf-setup.md)."""
    return dict(ili_sha256=sha256(settings['ili_path']) if settings.get('ili_path') else None, dataset_sha256=sha256(settings['dataset']),
                frozen_manifest_sha256=sha256(Path(settings['frozen']) / 'manifest.json'),
                population_sha256=sha256(LOCATIONS))


def check_pinned_inputs(settings):
    current = pinned_inputs(settings)
    changed = [k for k, v in current.items() if settings.get(k) != v]
    if changed:
        raise ValueError(f'{changed} differ from plan time ({settings["dataset"]}, {settings["frozen"]}, {LOCATIONS}); '
                         'rebuilding data, frozen support or populations needs a new experiment name')


def scenario_directory(value):
    return Scenario.from_string(value).run_id


def read_jobs(folder):
    if not (folder / 'jobs.csv').is_file():
        raise ValueError(f'No jobs.csv in {folder}; run plan first')
    with (folder / 'jobs.csv').open() as stream:
        return [dict(row, task=int(row['task']), seeds=[int(s) for s in row['seeds'].split()])
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


def snapshot_code(folder):
    """Copy src/ into `folder/code` and the notifier into `folder/notifications`, so
    Slurm jobs run the code the experiment was planned with even while the working
    tree moves on (scripts/jlessler.sbatch puts `code/src` first on PYTHONPATH). The
    launchers themselves are not copied: Slurm runs scripts/*.sbatch from the working
    tree. Refreshed on every accepted `plan`: re-planning an experiment re-pins its
    code. A local `planner run` does not use the snapshot; it runs the working tree."""
    root = Path(__file__).resolve().parents[3]
    destination = folder / 'code'
    if destination.exists():
        shutil.rmtree(destination)
    files = list((root / 'src').rglob('*.py')) + list((root / 'src').rglob('*.json')) + [root / 'pyproject.toml']
    for source in files:
        target = destination / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (folder / 'notifications').mkdir(exist_ok=True)
    shutil.copy2(root / 'scripts/notify.py', folder / 'notifications' / 'notify.py')
    save(destination / 'git.json', git_state())


def read_study(path):
    """{name: Scenario} and seeds from a study file; the file is copied into the experiment."""
    study = json.loads(Path(path).read_text())
    return {name: Scenario.from_string(value) for name, value in study['candidates'].items()}, study.get('seeds')


def plan(folder, scenarios, seeds, settings):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'experiment.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    changed = sorted(k for k, v in previous.items() if k in settings and settings[k] != v)
    protocol = set(changed) - {'device'}
    if protocol:
        raise ValueError(f'Experiment settings changed: {sorted(protocol)}; use a new experiment name')
    snapshot_code(folder)  # only after the check: a rejected re-plan keeps the pinned code
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


def attempts(folder, scenario, seed):
    return sorted((folder / scenario_directory(scenario) / f's{seed}').glob('attempt-*'))


def complete_artifacts(output):
    try:
        manifest = json.loads((output / 'manifest.json').read_text())
        seasons = Scenario.from_string(manifest.get('scenario', '')).scored_seasons
    except (OSError, ValueError, TypeError):
        return False
    required = ['manifest.json'] + [f'eval_{s}/{n}' for s in seasons for n in ('model.pt', 'nowcaster.pkl', 'forecasts.npz')]
    return (all((output / n).is_file() and (output / n).stat().st_size for n in required)
            and sorted(manifest.get('folds', [])) == sorted(seasons))


def seed_state(folder, scenario, seed):
    found = []
    for attempt in attempts(folder, scenario, seed):
        try:
            found.append((attempt, json.loads((attempt / 'run.json').read_text())))
        except (OSError, ValueError):
            found.append((attempt, dict(status='unknown')))
    for attempt, record in reversed(found):
        if record.get('status') == 'complete' and complete_artifacts(attempt):
            return attempt, record, True
    return found[-1] + (False,) if found else (None, dict(status='planned'), False)


def resumable(attempt, scenario):
    """An unfinished attempt with at least one saved fold model (prescribed-revision fits only)."""
    return (attempt is not None and Scenario.from_string(scenario).evaluation_inputs == 'prescribed'
            and any((attempt / f'eval_{s}' / 'model.pt').exists() for s in Scenario.from_string(scenario).scored_seasons))


def run_seed(folder, job, seed, settings):
    attempt, record, done = seed_state(folder, job['scenario'], seed)
    if done:
        print(f'Reusing {job["name"]}, seed {seed}', flush=True)
        return True
    check_pinned_inputs(settings)
    if resumable(attempt, job['scenario']):
        # Continue the stopped attempt: saved fold models and complete forecast views are reused.
        command = record['command'] + ['--resume']
        record = dict(record, status='running', resumed=now(), resume_command=command,
                      resumed_after=record.get('status'))
        log_name = 'resume.log'
    else:
        number = len(attempts(folder, job['scenario'], seed)) + 1
        attempt = folder / scenario_directory(job['scenario']) / f's{seed}' / f'attempt-{number:03d}'
        attempt.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, '-m', 'chromantis.experiment.planner', 'fit', '--scenario', job['scenario'],
                   '--seed', str(seed), '--device', settings['device'],
                   '--eval-members', str(settings['eval_members']), '--dataset', settings['dataset'],
                   '--frozen', settings['frozen'], '--output', str(attempt)]
        record = dict(status='running', name=job['name'], scenario=job['scenario'], seed=seed,
                      settings=settings, command=command, started=now(), **environment())
        log_name = 'run.log'
    save(attempt / 'run.json', record)
    print(f'Running {job["name"]}, seed {seed}: {attempt}', flush=True)
    try:
        with (attempt / log_name).open('a') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        if not complete_artifacts(attempt):
            raise RuntimeError('Run exited without complete fitted/scored artifacts')
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


def run(folder, tasks=None, device=None, keep_going=False, fit_workers=1, seeds=None):
    settings = json.loads((folder / 'experiment.json').read_text())
    if device:
        settings['device'] = device
    jobs = read_jobs(folder)
    if tasks is not None:
        jobs = [job for job in jobs if job['task'] in tasks]
    if seeds is not None:
        jobs = [dict(job, seeds=[s for s in job['seeds'] if s in seeds]) for job in jobs]
        jobs = [job for job in jobs if job['seeds']]
    failures, stop = 0, threading.Event()

    def worker(job):
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
        for future in as_completed([pool.submit(worker, job) for job in jobs]):
            failures += future.result()
    return failures


def collect(folder):
    rows = []
    for job in read_jobs(folder):
        for seed in job['seeds']:
            attempt, record, done = seed_state(folder, job['scenario'], seed)
            status = 'complete' if done else 'incomplete' if record.get('status') == 'complete' else record.get('status')
            rows.append(dict(task=job['task'], name=job['name'], seed=seed, status=status,
                             attempt=attempt.relative_to(folder).as_posix() if attempt else '',
                             scenario=job['scenario']))
    if rows:
        write_csv(folder / 'runs.csv', rows, list(rows[0]))
    return rows


def completed_runs(folder, allow_incomplete=False, seeds=None):
    """(completed runs to rank, whether they are every planned run of the experiment)."""
    planned = collect(folder)
    rows = planned if seeds is None else [row for row in planned if row['seed'] in seeds]
    done = [row for row in rows if row['status'] == 'complete']
    if not done or (len(done) < len(rows) and not allow_incomplete):
        raise ValueError(f'{len(rows) - len(done)} of {len(rows)} runs incomplete; finish them or pass --allow-incomplete')
    return done, len(done) == len(planned)


def replay_folder(folder, attempt, inputs):
    """Where `planner replay` writes the replay of one completed attempt."""
    return Path(folder) / f'replay-{inputs}' / Path(attempt).relative_to(folder)


def ranking_runs(folder, allow_incomplete=False, seeds=None, inputs=None):
    """Completed runs as ranking rows, and whether they are every planned run."""
    done, every_run = completed_runs(folder, allow_incomplete, seeds)
    runs = [dict(config_id=row['scenario'], name=row['name'], seed=row['seed'], path=folder / row['attempt'])
            for row in sorted(done, key=lambda r: r['attempt'])]
    if inputs:
        runs = [dict(r, path=replay_folder(folder, r['path'], inputs)) for r in runs]
        missing = [str(r['path']) for r in runs if not (r['path'] / 'manifest.json').exists()]
        if missing:
            raise ValueError(f'{len(missing)} runs not replayed on {inputs} inputs; run planner replay first')
    return runs, every_run


def rank(folder, allow_incomplete=False, seeds=None, inputs=None, report=True):
    """Score completed runs into `ranking-<hash>/` (`evaluation/ranking.py`).

    The report page (`docs/experiments/<experiment>/index.md`) is written only for the
    complete ranking of the fitted runs: every planned run complete and included."""
    from chromantis.evaluation.ranking import rank as rank_runs
    runs, every_run = ranking_runs(folder, allow_incomplete, seeds, inputs)
    key = dict(attempts=sorted(str(r['path'].relative_to(folder)) for r in runs), score='headline-v1')
    destination = folder / f"ranking-{hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]}"
    settings = json.loads((folder / 'experiment.json').read_text())
    rank_runs(runs, settings['frozen'], destination)
    if report and every_run and not inputs:
        from chromantis.evaluation.report import write_report
        print(write_report(folder, destination, runs), flush=True)
    elif report:
        print('Report not written: ' + ('replayed inputs are ranked in their own folder' if inputs else
                                        'not every planned run is ranked (--seeds subset or incomplete runs)'), flush=True)
    return destination


def replay_runs(folder, inputs, device, seeds=None):
    """Evaluate every completed run on `inputs` without refitting (`fit.replay`), fold by fold."""
    settings = json.loads((folder / 'experiment.json').read_text())
    runs, _ = ranking_runs(folder, allow_incomplete=True, seeds=seeds)
    for run in runs:
        target = replay_folder(folder, run['path'], inputs)
        manifest = json.loads((run['path'] / 'manifest.json').read_text())
        for season in manifest['folds']:
            if not (target / f'eval_{season}' / 'manifest.json').exists():
                replay(run['path'] / f'eval_{season}', target / f'eval_{season}', inputs, settings['dataset'], device)
        save(target / 'manifest.json', dict(manifest, replay_of=str(run['path']), inputs=inputs, refit=False))
        print(f'Replayed {run["name"]}, seed {run["seed"]} on {inputs} inputs: {target}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit_parser = sub.add_parser('fit', help='Fit and evaluate one scenario/seed across its held-out seasons')
    fit_parser.add_argument('--scenario', required=True)
    fit_parser.add_argument('--seed', type=int, default=42)
    fit_parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    fit_parser.add_argument('--eval-members', type=int, default=EVAL_MEMBERS)
    fit_parser.add_argument('--dataset', default=PANEL_DATASET)
    fit_parser.add_argument('--frozen', default=FROZEN)
    fit_parser.add_argument('--output', required=True)
    fit_parser.add_argument('--resume', action='store_true', help='Reuse saved prescribed-revision fold models and completed forecast views')
    for name in ('plan', 'run', 'status', 'rank', 'replay'):
        p = sub.add_parser(name)
        p.add_argument('-e', '--experiment', required=True)
        p.add_argument('--root', default='data/experiments')
        if name == 'plan':
            p.add_argument('-s', '--scenario', nargs='+', help='Full scenario strings to plan (named by run id)')
            p.add_argument('--study', help='Study file: {"candidates": {name: scenario}, "seeds": [...]}')
            p.add_argument('--seeds', nargs='+', type=int, default=None,
                           help='Default: the study file\'s seeds, else 42 43 (two-seed screen, 2026-10-05)')
            p.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
            p.add_argument('--dataset', default=PANEL_DATASET)
            p.add_argument('--frozen', default=FROZEN)
        if name == 'run':
            p.add_argument('-t', '--task', nargs='+', type=int, default=None)
            p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default=None)
            p.add_argument('--keep-going', action='store_true')
            p.add_argument('--fit-workers', type=int, default=1)
        if name in ('run', 'rank', 'status', 'replay'):
            p.add_argument('--seeds', nargs='+', type=int, default=None)
        if name in ('rank', 'replay'):
            p.add_argument('--inputs', choices=['reported', 'synthetic'], default=None if name == 'rank' else 'reported',
                           help='replay: evaluation inputs; rank: rank those replays instead of the fits')
        if name == 'replay':
            p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
        if name == 'rank':
            p.add_argument('--allow-incomplete', action='store_true')
            p.add_argument('--no-report', action='store_true', help='Write score tables only')
    args = parser.parse_args(argv)
    if args.command == 'fit':
        scenario = Scenario.from_string(args.scenario)
        output = Path(args.output)
        for held_out in scenario.scored_seasons:
            fit(scenario, args.seed, held_out, args.eval_members, args.device, output / f'eval_{held_out}',
                args.dataset, resume=args.resume)
        folds = {held: json.loads((output / f'eval_{held}' / 'manifest.json').read_text()) for held in scenario.scored_seasons}
        save(output / 'manifest.json', dict(scenario=scenario.scenario_string, run_id=scenario.run_id,
                                            seed=args.seed, folds=list(scenario.scored_seasons), fold_manifests=folds,
                                            eval_members=args.eval_members, dataset=args.dataset, frozen=args.frozen))
        if scenario.evaluation_seasons != 'production':
            from chromantis.evaluation.ranking import cache_scores
            cache_scores(output, args.frozen)
        return
    folder = Path(args.root) / args.experiment
    if args.command == 'plan':
        if bool(args.scenario) == bool(args.study):
            parser.error('Give either --scenario strings or one --study file')
        if args.study:
            scenarios, study_seeds = read_study(args.study)
        else:
            scenarios, study_seeds = {Scenario.from_string(s).run_id: Scenario.from_string(s) for s in args.scenario}, None
        seeds = args.seeds or study_seeds or [42, 43]
        settings = dict(device=args.device, eval_members=EVAL_MEMBERS, dataset=args.dataset, frozen=args.frozen)
        ili_paths = {s.ili_path for s in scenarios.values() if s.ili_training != 'none'}
        if len(ili_paths) > 1:
            raise ValueError('One historical ILI dataset per experiment')
        if ili_paths:
            settings['ili_path'] = next(iter(ili_paths))
        settings.update(pinned_inputs(settings))
        jobs = plan(folder, scenarios, seeds, settings)
        if args.study:
            shutil.copy2(args.study, folder / 'study.json')
        print(json.dumps(dict(experiment=str(folder), configurations=len(scenarios), seeds=len(seeds),
                              runs=len(jobs) and len(scenarios) * len(seeds))))
    elif args.command == 'run':
        if run(folder, args.task, args.device, args.keep_going, args.fit_workers, args.seeds):
            raise SystemExit(1)
    elif args.command == 'status':
        rows = collect(folder)
        if args.seeds is not None:
            rows = [row for row in rows if row['seed'] in args.seeds]
        for row in rows:
            print(f"{row['status']}\t{row['task']}\t{row['name']}\ts{row['seed']}\t{row['attempt']}")
        unfinished = sum(row['status'] != 'complete' for row in rows)
        if unfinished:
            root = '' if args.root == 'data/experiments' else f' --root {args.root}'
            print(f'{unfinished} of {len(rows)} runs not complete. Resume on Longleaf (stopped prescribed-revision '
                  f'runs continue from their saved folds): sbatch --job-name={args.experiment} '
                  f'--array=0-3 scripts/jlessler.sbatch {args.experiment} --retry-failed\n'
                  f'or locally: .venv/bin/python -m chromantis.experiment.planner run -e {args.experiment}{root}')
    elif args.command == 'rank':
        print(rank(folder, args.allow_incomplete, args.seeds, args.inputs, report=not args.no_report))
    elif args.command == 'replay':
        replay_runs(folder, args.inputs, args.device, args.seeds)


if __name__ == '__main__':
    main()
