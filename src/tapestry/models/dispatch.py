"""Shared seed queue for heterogeneous Slurm GPUs.

NFS advisory locking serializes short queue updates. A configuration has at most
one active seed; its next seed may move to another GPU. No static node slices.
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

from .manager import check_frozen, read_jobs, run_seed, save, seed_state
from .scenarios import TrainingScenario


def signature(s):
    """Fields affecting tensor size; transforms share a conservative memory margin."""
    names = ('lookback', 'width', 'latent', 'encoder', 'decoder', 'head_sharing', 'heads',
             'spatial', 'noise', 'dynamics', 'annual_calendar', 'location_embedding', 'us_error')
    return '|'.join(str(getattr(s, key)) for key in names)


@contextmanager
def queue_lock(folder):
    with (folder / 'dispatch.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def initialize(folder):
    with queue_lock(folder):
        if (folder / 'dispatch.json').exists():
            return
        jobs = read_jobs(folder)
        state = {'tasks': {str(job['task']): {'active': None, 'seconds': [],
                    'seeds': {str(seed): 'complete' if seed_state(folder, job['scenario'], seed)[2] else 'pending'
                              for seed in job['seeds']}} for job in jobs}}
        save(folder / 'dispatch.json', state)


class Queue:
    def __init__(self, folder, owner):
        self.folder, self.owner = folder, owner
        self.jobs = {str(job['task']): job for job in read_jobs(folder)}
        self.cost = {}
        for task, job in self.jobs.items():
            s = TrainingScenario.from_string(job['scenario'])
            components = {'all': 1, 'pathogen': 3, 'target': 6}[s.fit_partition]
            # Scheduling estimate only: caps are not actual selected epoch counts.
            encoder = {'mlp': 1., 'conv': 1.5, 'multiscale_conv': 2.}[s.encoder]
            decoder = {'legacy': 1., 'residual2': 2., 'stochastic_trend': .8}[s.decoder]
            exchange = 1.4 if s.spatial == 'joint_location_target' else 1.
            self.cost[task] = components * s.epochs * (s.width / 64)**2 * (s.lookback / 12)**.5 * encoder * decoder * exchange
        self.local = threading.Lock()
        initialize(folder)

    def claim(self, lane):
        with self.local, queue_lock(self.folder):
            state = json.loads((self.folder / 'dispatch.json').read_text())
            eligible = []
            pending = False
            for task, entry in state['tasks'].items():
                todo = [seed for seed, status in entry['seeds'].items() if status == 'pending']
                pending |= bool(todo) or entry['active'] is not None
                if not todo or entry['active']:
                    continue
                cost = self.cost[task]
                # Long remaining chains start first; ties start longer individual fits.
                eligible.append((cost * len(todo), cost, -int(task), task, todo[0]))
            if not eligible:
                return None, pending
            _, _, _, task, seed = max(eligible)
            entry = state['tasks'][task]
            entry['active'] = dict(owner=self.owner, lane=lane, seed=seed, started=time.time())
            entry['seeds'][seed] = 'running'
            save(self.folder / 'dispatch.json', state)
            return (task, int(seed)), True

    def finish(self, task, seed, success, seconds):
        with self.local, queue_lock(self.folder):
            state = json.loads((self.folder / 'dispatch.json').read_text())
            entry = state['tasks'][task]
            entry['active'] = None
            entry['seeds'][str(seed)] = 'complete' if success else 'failed'
            if success:
                entry['seconds'].append(seconds)
            save(self.folder / 'dispatch.json', state)

    def reclaim(self):
        """Release seeds whose Slurm allocation disappeared; never guess from age."""
        result = subprocess.run(['squeue', '-r', '-h', '-u', os.environ['USER'], '-o', '%i'],
                                text=True, capture_output=True)
        if result.returncode:
            return
        live = set(result.stdout.split())
        with queue_lock(self.folder):
            state = json.loads((self.folder / 'dispatch.json').read_text())
            changed = False
            for task, entry in state['tasks'].items():
                active = entry['active']
                if active and active['owner'] not in live:
                    seed = active['seed']
                    done = seed_state(self.folder, self.jobs[task]['scenario'], int(seed))[2]
                    entry['seeds'][seed] = 'complete' if done else 'pending'
                    entry['active'] = None
                    changed = True
            if changed:
                save(self.folder / 'dispatch.json', state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-e', '--experiment', default='B0.1')
    parser.add_argument('--lanes', type=int, required=True)
    args = parser.parse_args()
    if args.lanes < 1:
        parser.error('Positive lanes required')
    if 'SLURM_JOB_ID' not in os.environ:
        parser.error('The shared GPU dispatcher requires a Slurm allocation')
    folder = Path('data/experiments') / args.experiment
    settings = json.loads((folder / 'experiment.json').read_text())
    settings['device'] = 'cuda'
    check_frozen(settings['frozen'])
    owner = (os.environ['SLURM_ARRAY_JOB_ID'] + '_' + os.environ['SLURM_ARRAY_TASK_ID']
             if 'SLURM_ARRAY_JOB_ID' in os.environ else os.environ['SLURM_JOB_ID'])
    queue = Queue(folder, owner)
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
            started, success = time.perf_counter(), False
            print(json.dumps(dict(lane=lane, task=int(task), seed=seed)), flush=True)
            try:
                success = run_seed(folder, queue.jobs[task], seed, settings)
            finally:
                queue.finish(task, seed, success, time.perf_counter() - started)
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
