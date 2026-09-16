"""Publish aggregate experiment status to ntfy after Slurm allocations terminate."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import Request, urlopen

from tapestry.models.manager import read_jobs, seed_state


def summary(folder, job_ids):
    # Include hidden partitions and expand arrays so owner IDs match run metadata.
    live_result = subprocess.run(['squeue', '-a', '-r', '-h', '-u', os.environ['USER'], '-o', '%i'],
                                 capture_output=True, text=True, check=True)
    live = set(live_result.stdout.split())
    counts, configurations_done = Counter(), 0
    jobs = read_jobs(folder)
    for job in jobs:
        completed = 0
        for seed in job['seeds']:
            _, record, done = seed_state(folder, job['scenario'], seed)
            status = 'complete' if done else record.get('status', 'unknown')
            if status == 'running':
                slurm = record.get('slurm', {})
                owner = (f"{slurm['SLURM_ARRAY_JOB_ID']}_{slurm['SLURM_ARRAY_TASK_ID']}"
                         if slurm.get('SLURM_ARRAY_JOB_ID') else slurm.get('SLURM_JOB_ID'))
                if owner not in live:
                    status = 'interrupted'
            if status == 'planned':
                status = 'pending'
            elif status == 'complete' and not done:
                status = 'incomplete'
            counts[status] += 1
            completed += done
        configurations_done += completed == len(job['seeds'])
    accounting = subprocess.run(['sacct', '-X', '-n', '-P', '-j', ','.join(job_ids),
                                 '--format=JobID,State,ExitCode'], capture_output=True, text=True)
    allocation_states = {}
    if accounting.returncode == 0:
        for line in accounting.stdout.splitlines():
            fields = line.split('|')
            if len(fields) >= 3 and any(fields[0] == j or fields[0].startswith(j + '_') for j in job_ids):
                allocation_states[fields[0]] = {'state': fields[1], 'exit_code': fields[2]}
    return dict(seed_counts=dict(counts), total_seeds=sum(counts.values()),
                complete_configurations=configurations_done, total_configurations=len(jobs),
                allocations=allocation_states, accounting_available=accounting.returncode == 0)


def message(experiment, result, event, job_ids):
    counts, total = result['seed_counts'], result['total_seeds']
    done = counts.get('complete', 0)
    opening = ('Notifications enabled; GPU jobs are still running.' if event == 'enabled'
               else 'Job terminated: all monitored GPU allocations have ended.')
    states = Counter(v['state'] for v in result['allocations'].values())
    lines = [f'{experiment}: {opening}',
             f'CV evaluations completed: {done}/{total}; unfinished: {total - done}.',
             'Failed: {failed}; interrupted: {interrupted}; incomplete: {incomplete}; '
             'pending: {pending}; running: {running}; unknown: {unknown}.'.format(
                 **{key: counts.get(key, 0) for key in ('failed', 'interrupted', 'incomplete', 'pending', 'running', 'unknown')}),
             f"Configurations fully complete: {result['complete_configurations']}/{result['total_configurations']}.",
             f"Slurm arrays: {', '.join(job_ids)}.",
             'Allocation outcomes: ' + (', '.join(f'{state}={count}' for state, count in sorted(states.items()))
                                       if states else 'accounting not yet available') + '.']
    if event == 'enabled':
        lines.append('One final summary will follow when both arrays end, including cancellation or timeout.')
    elif done < total:
        lines.append('Work remains; check the experiment manager before resuming.')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-e', '--experiment', default='B0.1')
    parser.add_argument('--jobs', nargs='+', required=True)
    parser.add_argument('--url', default='https://ntfy.sh/tapestry')
    parser.add_argument('--event', choices=['enabled', 'terminated'], default='terminated')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    folder = Path('data/experiments') / args.experiment
    result = summary(folder, args.jobs)
    body = message(args.experiment, result, args.event, args.jobs)
    print(body, flush=True)
    if args.dry_run:
        return
    complete = result['seed_counts'].get('complete', 0) == result['total_seeds']
    headers = {'Title': f'{args.experiment}: ' + ('notifications enabled' if args.event == 'enabled' else 'job terminated'),
               'Priority': '3',
               'Tags': 'bell' if args.event == 'enabled' else 'white_check_mark' if complete else 'warning',
               'Content-Type': 'text/plain; charset=utf-8'}
    for attempt in range(3):
        try:
            request = Request(args.url, data=body.encode('utf-8'), headers=headers, method='POST')
            with urlopen(request, timeout=20) as response:
                receipt = json.load(response)
            break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    output = folder / 'notifications'
    output.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    (output / f'{stamp}-{args.event}.json').write_text(json.dumps(
        dict(url=args.url, message=body, summary=result, receipt=receipt), indent=2) + '\n')
    print(f"Published ntfy message {receipt.get('id')}", flush=True)


if __name__ == '__main__':
    main()
