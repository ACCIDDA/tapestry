"""Shared seed queue for heterogeneous Slurm GPUs.

Moved unchanged from `models/dispatch.py`: model-agnostic already. NFS advisory
locking serializes short queue updates. Independent seeds may run concurrently;
each seed has exactly one owner. No static node slices.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

from .planner import read_jobs, run_seed, save, seed_state


@contextmanager
def queue_lock(folder):
    with (folder / 'dispatch.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def initialize(folder, retry_failed=False):
    with queue_lock(folder):
        jobs = read_jobs(folder)
        path = folder / 'dispatch.json'
        state = json.loads(path.read_text()) if path.exists() else {'tasks': {}}
        for job in jobs:
            entry = state['tasks'].setdefault(str(job['task']), dict(active={}, seeds={}))
            for seed in job['seeds']:
                if str(seed) in entry['active']:
                    continue
                previous = entry['seeds'].get(str(seed))
                if seed_state(folder, job['scenario'], seed)[2]:
                    status = 'complete'
                else:
                    status = 'failed' if previous == 'failed' and not retry_failed else 'pending'
                entry['seeds'][str(seed)] = status
        save(folder / 'dispatch.json', state)


class Queue:
    def __init__(self, folder, owner, lanes=6, gpu_count=6, retry_failed=False, seeds=None):
        self.folder, self.owner = folder, owner
        self.lanes, self.gpu_count = lanes, gpu_count
        self.seeds = None if seeds is None else {str(seed) for seed in seeds}
        self.jobs = {str(job['task']): job for job in read_jobs(folder)}
        from chromantis.problem import Problem
        problem = Problem.load(json.loads((folder / 'experiment.json').read_text())['problem'])
        self.cost = {}
        for task, job in self.jobs.items():
            from chromantis.model.scenario import Scenario
            s = Scenario.from_string(job['scenario'])
            components = (1 if s.fit_partition == 'all' else len(set(problem.target_groups))
                          if s.fit_partition == 'pathogen' else len(problem.targets))
            encoder = {'mlp': 1., 'conv': 1.5, 'multiscale_conv': 2., 'series_mlp': .5, 'series_mixer': .6}[s.encoder]
            decoder = {'legacy': 1., 'residual2': 2., 'quantile': .5, 'quantile_small': .5}[s.decoder]
            exchange = 1.4 if s.spatial == 'joint_location_target' else 1.
            self.cost[task] = components * s.epochs * (s.width / 64)**2 * (s.lookback / 12)**.5 * encoder * decoder * exchange
        self.local = threading.Lock()
        initialize(folder, retry_failed)
        with queue_lock(folder):
            state = json.loads((folder / 'dispatch.json').read_text())
            state.setdefault('owners', {})[owner] = lanes
            save(folder / 'dispatch.json', state)

    def claim(self, lane):
        with self.local, queue_lock(self.folder):
            state = json.loads((self.folder / 'dispatch.json').read_text())
            eligible = []
            loads = {owner: 0. for owner in state['owners']}
            counts = {owner: 0 for owner in state['owners']}
            for task, entry in state['tasks'].items():
                for active in entry['active'].values():
                    owner = active['owner']
                    loads[owner] = loads.get(owner, 0.) + self.cost[task]
                    counts[owner] = counts.get(owner, 0) + 1
            idle_peer = any(owner != self.owner and counts[owner] < lanes
                            for owner, lanes in state['owners'].items())
            pending = False
            for task, entry in state['tasks'].items():
                todo = [seed for seed, status in entry['seeds'].items() if status == 'pending'
                        and (self.seeds is None or seed in self.seeds)]
                active = [seed for seed in entry['active'] if self.seeds is None or seed in self.seeds]
                pending |= bool(todo) or bool(active)
                if not todo:
                    continue
                if (idle_peer and entry.get('last_owner') == self.owner
                        and time.time() - entry.get('finished', 0) < 10):
                    continue
                cost = self.cost[task]
                eligible.append((cost * len(todo), cost, -int(task), task, todo[0]))
            if not eligible:
                return None, pending
            average = sum(loads.values()) / self.gpu_count
            chosen = min(eligible) if loads.get(self.owner, 0) > average else max(eligible)
            _, _, _, task, seed = chosen
            entry = state['tasks'][task]
            entry['active'][seed] = dict(owner=self.owner, lane=lane, seed=seed, started=time.time())
            entry['seeds'][seed] = 'running'
            save(self.folder / 'dispatch.json', state)
            return (task, int(seed)), True

    def finish(self, task, seed, success):
        with self.local, queue_lock(self.folder):
            state = json.loads((self.folder / 'dispatch.json').read_text())
            entry = state['tasks'][task]
            entry['active'].pop(str(seed))
            entry.update(last_owner=self.owner, finished=time.time())
            entry['seeds'][str(seed)] = 'complete' if success else 'failed'
            save(self.folder / 'dispatch.json', state)

    def reclaim(self):
        try:  # an unreachable Slurm controller must not block dispatch (7 October 2026 outage)
            result = subprocess.run(['squeue', '-a', '-r', '-h', '-u', os.environ['USER'], '-o', '%i'],
                                    text=True, capture_output=True, timeout=20)
        except subprocess.TimeoutExpired:
            return
        if result.returncode:
            return
        live = set(result.stdout.split())
        with queue_lock(self.folder):
            state = json.loads((self.folder / 'dispatch.json').read_text())
            changed = False
            for owner in list(state.get('owners', {})):
                if owner not in live:
                    del state['owners'][owner]
                    changed = True
            for task, entry in state['tasks'].items():
                for seed, active in list(entry['active'].items()):
                    if active['owner'] not in live:
                        done = seed_state(self.folder, self.jobs[task]['scenario'], int(seed))[2]
                        entry['seeds'][seed] = 'complete' if done else 'pending'
                        del entry['active'][seed]
                        changed = True
            if changed:
                save(self.folder / 'dispatch.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-e', '--experiment', required=True)
    parser.add_argument('--lanes', type=int, required=True)
    parser.add_argument('--gpu-count', type=int, default=6)
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if args.lanes < 1 or args.gpu_count < 1:
        parser.error('Positive lanes and gpu-count required')
    if 'SLURM_JOB_ID' not in os.environ:
        parser.error('The shared GPU dispatcher requires a Slurm allocation')
    folder = Path('data/experiments') / args.experiment
    settings = json.loads((folder / 'experiment.json').read_text())
    settings['device'] = 'cuda'
    owner = (os.environ['SLURM_ARRAY_JOB_ID'] + '_' + os.environ['SLURM_ARRAY_TASK_ID']
             if 'SLURM_ARRAY_JOB_ID' in os.environ else os.environ['SLURM_JOB_ID'])
    queue = Queue(folder, owner, args.lanes, args.gpu_count, args.retry_failed, args.seeds)
    stop = threading.Event()
    print(json.dumps(dict(dispatch_owner=owner, host=socket.gethostname(), lanes=args.lanes)), flush=True)

    def worker(lane):
        failures = 0
        while not stop.is_set():
            claimed, pending = queue.claim(lane)
            if claimed is None:
                if not pending:
                    return failures
                stop.wait(5)
                continue
            task, seed = claimed
            success = False
            print(json.dumps(dict(lane=lane, task=int(task), seed=seed)), flush=True)
            try:
                success = run_seed(folder, queue.jobs[task], seed, settings)
            finally:
                queue.finish(task, seed, success)
            failures += not success
        return failures

    def recover():
        while not stop.wait(60):
            queue.reclaim()

    queue.reclaim()
    watcher = threading.Thread(target=recover, daemon=True)
    watcher.start()
    try:
        with ThreadPoolExecutor(max_workers=args.lanes) as pool:
            failures = sum(f.result() for f in [pool.submit(worker, i) for i in range(args.lanes)])
    finally:
        stop.set()
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
