"""Plan, run, status, rank and compare experiments for the unified `Scenario`.

A shared manager fits standalone nowcasters, standalone forecasters and a
cross-fitted two-stage pipeline. The `plan`, `run`, `status` and `rank` commands
serve all three tasks. Forecast ranking uses hub-relative WIS; standalone
nowcasting uses its training-normalized native-unit fair CRPS.

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
import os
import shutil
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np
from tapestry.dataset.build import PANEL_DATASET
from tapestry.dataset.cv import SEASONS
from tapestry.evaluation.totals import US_SCORE_WEIGHT, ADMISSIONS_WEIGHT, ED_WEIGHT
from tapestry.model.scenario import Scenario
from .fitting import fit, LOCATIONS
from .provenance import save, now, environment, git_state, sha256

JOB_FIELDS = ['task', 'name', 'scenario', 'seeds']
FROZEN = 'data/evaluation/b0_hub_comparison_q23'


def pinned_inputs(settings):
    """Hashes of the dataset, frozen-support manifest and population file an experiment was planned against.

    The population file (`LOCATIONS`) and the frozen support are git-ignored, not
    synced with the code: copy them to the cluster (docs/longleaf-setup.md)."""
    return dict(dataset_sha256=sha256(settings['dataset']),
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
    required = ['manifest.json'] + [f'eval_{s}/model.pt' for s in seasons] + [f'eval_{s}/forecasts.npz' for s in seasons]
    if not all((output / n).is_file() and (output / n).stat().st_size for n in required):
        return False
    try:
        manifest = json.loads((output / 'manifest.json').read_text())
        task = Scenario.from_string(manifest.get('scenario', '')).task
        extra = (['nowcast_scores.json'] if task == 'nowcast' else
                 ['nowcaster.pt', 'forecaster.pt', 'nowcast/forecasts.npz', 'nowcast/nowcast_scores.json']
                 if task == 'pipeline' else [])
        return (sorted(manifest.get('folds', [])) == sorted(seasons)
                and all((output / f'eval_{season}' / name).is_file() for season in seasons for name in extra))
    except (KeyError, ValueError, TypeError):
        return False


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


def run_seed(folder, job, seed, settings):
    if seed_state(folder, job['scenario'], seed)[2]:
        print(f'Reusing {job["name"]}, seed {seed}', flush=True)
        return True
    check_pinned_inputs(settings)
    number = len(attempts(folder, job['scenario'], seed)) + 1
    attempt = folder / scenario_directory(job['scenario']) / f's{seed}' / f'attempt-{number:03d}'
    attempt.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-m', 'tapestry.experiment.planner', 'fit', '--scenario', job['scenario'],
               '--seed', str(seed), '--device', settings['device'],
               '--eval-members', str(settings['eval_members']), '--dataset', settings['dataset'],
               '--frozen', settings['frozen'], '--output', str(attempt)]
    record = dict(status='running', name=job['name'], scenario=job['scenario'], seed=seed,
                 settings=settings, command=command, started=now(), **environment())
    save(attempt / 'run.json', record)
    print(f'Running {job["name"]}, seed {seed}: {attempt}', flush=True)
    try:
        with (attempt / 'run.log').open('w') as log:
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


def rank(folder, allow_incomplete=False, seeds=None, us_weight=US_SCORE_WEIGHT,
         admissions_weight=ADMISSIONS_WEIGHT, ed_weight=ED_WEIGHT, make_plots=True):
    """Score completed runs into `ranking-<hash>/` (hash of the runs and the score weights), then plot.

    The report page (`docs/results/<experiment>/index.md`) is written only for the
    complete ranking: every planned run complete and included, default score weights.
    A subset (`--seeds`, `--allow-incomplete` with missing runs) or non-default weights
    still gets its ranking folder and figures, not the report."""
    from tapestry.evaluation.totals import rank as rank_runs
    from tapestry.evaluation.plots import plot_experiment, write_report
    if not 0 <= us_weight <= 1 or min(admissions_weight, ed_weight) < 0 or not admissions_weight + ed_weight:
        raise ValueError('Need 0 <= us_weight <= 1 and nonnegative target weights, not both zero')
    done, every_run = completed_runs(folder, allow_incomplete, seeds)
    nowcasts = [r for r in done if Scenario.from_string(r['scenario']).task == 'nowcast']
    if nowcasts:
        if len(nowcasts) != len(done):
            raise ValueError('Rank nowcasts and forecasts in separate experiments: their scores differ')
        if (us_weight, admissions_weight, ed_weight) != (US_SCORE_WEIGHT, ADMISSIONS_WEIGHT, ED_WEIGHT):
            raise ValueError('Nowcast ranking uses the fixed scientific objective weights')
        rows = []
        for row in nowcasts:
            scores = [json.loads((folder / row['attempt'] / f'eval_{s}' / 'nowcast_scores.json').read_text())
                      ['normalized_crps'] for s in SEASONS]
            rows.append(dict(scenario=row['scenario'], seed=row['seed'], normalized_crps=float(np.mean(scores))))
        rows.sort(key=lambda r: r['normalized_crps'])
        destination = folder / 'nowcast-ranking.csv'
        write_csv(destination, rows, list(rows[0]))
        print(json.dumps(rows, indent=2), flush=True)
        return destination
    key = dict(attempts=sorted(r['attempt'] for r in done), us_weight=us_weight,
               admissions_weight=admissions_weight, ed_weight=ed_weight)
    destination = folder / f"ranking-{hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]}"
    runs = [dict(config_id=row['scenario'], name=row['name'], seed=row['seed'], path=folder / row['attempt'])
            for row in sorted(done, key=lambda r: r['attempt'])]
    ranking = rank_runs(runs, destination, us_weight, admissions_weight, ed_weight)
    print(ranking.head(20).to_string(index=False), flush=True)
    if not make_plots:
        return destination
    for path in plot_experiment(folder, destination):
        print(path, flush=True)
    default_weights = (us_weight, admissions_weight, ed_weight) == (US_SCORE_WEIGHT, ADMISSIONS_WEIGHT, ED_WEIGHT)
    if every_run and default_weights:
        print(write_report(folder, destination), flush=True)
    else:
        print('Report not written: ' + ' and '.join(
            reason for reason, applies in (('not every planned run is ranked (--seeds subset or incomplete runs)',
                                            not every_run),
                                           ('score weights differ from the defaults', not default_weights)) if applies),
              flush=True)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    fit_parser = sub.add_parser('fit', help='Fit and evaluate one scenario/seed across all three seasons')
    fit_parser.add_argument('--scenario', required=True)
    fit_parser.add_argument('--seed', type=int, default=42)
    fit_parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    fit_parser.add_argument('--eval-members', type=int, default=256)
    fit_parser.add_argument('--dataset', default=PANEL_DATASET)
    fit_parser.add_argument('--frozen', default=FROZEN)
    fit_parser.add_argument('--output', required=True)
    for name in ('plan', 'run', 'status', 'rank', 'plots'):
        p = sub.add_parser(name)
        p.add_argument('-e', '--experiment', required=True)
        p.add_argument('--root', default='data/experiments')
        if name == 'plan':
            p.add_argument('-s', '--scenario', nargs='+', required=True, help='Full scenario strings to plan')
            p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
            p.add_argument('--eval-members', type=int, default=256)
            p.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
            p.add_argument('--dataset', default=PANEL_DATASET)
            p.add_argument('--frozen', default=FROZEN)
        if name == 'run':
            p.add_argument('-t', '--task', nargs='+', type=int, default=None)
            p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default=None)
            p.add_argument('--keep-going', action='store_true')
            p.add_argument('--fit-workers', type=int, default=1)
            p.add_argument('--seeds', nargs='+', type=int, default=None)
        if name in ('rank', 'status'):
            p.add_argument('--seeds', nargs='+', type=int, default=None)
        if name == 'rank':
            p.add_argument('--allow-incomplete', action='store_true')
            p.add_argument('--no-plots', action='store_true', help='Write score tables only, without plots or a report')
            p.add_argument('--us-weight', type=float, default=US_SCORE_WEIGHT,
                           help='US share of the score (states/DC share the rest equally)')
            p.add_argument('--admissions-weight', type=float, default=ADMISSIONS_WEIGHT)
            p.add_argument('--ed-weight', type=float, default=ED_WEIGHT)
        if name == 'plots':
            p.add_argument('--ranking', help='Ranking folder (default: the most recent ranking-*)')
            p.add_argument('--configs', nargs='+', help="Scenario strings for fans/heatmaps ('' or default = "
                                                        'the default scenario), any number; default: the best ranked')
            p.add_argument('--dates', nargs='+', help='Fan reference dates (default: every 4 weeks of each held-out season)')
    args = parser.parse_args(argv)
    if args.command == 'fit':
        scenario = Scenario.from_string(args.scenario)
        output = Path(args.output)
        for held_out in scenario.scored_seasons:
            fit(scenario, args.seed, held_out, args.eval_members, args.device, output / f'eval_{held_out}', args.dataset)
        folds = {held: json.loads((output / f'eval_{held}' / 'manifest.json').read_text()) for held in scenario.scored_seasons}
        save(output / 'manifest.json', dict(scenario=scenario.scenario_string, run_id=scenario.run_id,
                                            seed=args.seed, folds=list(scenario.scored_seasons), fold_manifests=folds,
                                            eval_members=args.eval_members, dataset=args.dataset, frozen=args.frozen))
        if scenario.task != 'nowcast':
            from tapestry.evaluation.totals import score_run
            score_run(output, args.frozen)
        return
    folder = Path(args.root) / args.experiment
    if args.command == 'plan':
        scenarios = {Scenario.from_string(s).run_id: Scenario.from_string(s) for s in args.scenario}
        settings = dict(device=args.device, eval_members=args.eval_members, dataset=args.dataset,
                        frozen=args.frozen)
        settings.update(pinned_inputs(settings))
        jobs = plan(folder, scenarios, args.seeds, settings)
        print(json.dumps(dict(experiment=str(folder), configurations=len(scenarios), seeds=len(args.seeds),
                              runs=len(jobs) and len(scenarios) * len(args.seeds))))
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
            print(f'{unfinished} of {len(rows)} runs not complete. Resume on Longleaf: sbatch --job-name={args.experiment} '
                  f'--array=0-3 scripts/jlessler.sbatch {args.experiment} --retry-failed\n'
                  f'or locally: .venv/bin/python -m tapestry.experiment.planner run -e {args.experiment}{root}')
    elif args.command == 'rank':
        print(rank(folder, args.allow_incomplete, args.seeds, args.us_weight, args.admissions_weight, args.ed_weight,
                   make_plots=not args.no_plots))
    elif args.command == 'plots':
        from tapestry.evaluation.plots import plot_experiment
        rankings = sorted(folder.glob('ranking-*'), key=lambda p: p.stat().st_mtime)
        ranking = Path(args.ranking) if args.ranking else rankings[-1] if rankings else None
        if ranking is None:
            parser.error(f'No ranking-* folder in {folder}; run rank first')
        for path in plot_experiment(folder, ranking, args.configs, args.dates):
            print(path)


if __name__ == '__main__':
    main()
