"""Run Emily's create configs against pinned local snapshots, keeping originals intact.

Only hub/output paths are adapted. Dates, targets and the -3-day vintage rule
remain Emily's; failures are recorded rather than silently changing the protocol.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import yaml

ROOT = Path('data/epibench/emily').resolve()
SOURCES = Path('fromEmily/configs')


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(Path('data/evaluation/b0_hub_comparison/manifest.json').read_text())
    snapshots = {}
    for hub, info in frozen['hubs'].items():
        # EpiBench date validation keys off the directory name, so retain the hub name.
        output = ROOT / 'hubs' / Path(info['url']).name
        mirror = Path('data/mirrors') / f'hub_{hub}_current.git'
        truth = 'target-data/time-series.csv' if hub == 'flusight' else 'target-data/time-series.parquet'
        for name in ('hub-config/admin.json', 'hub-config/tasks.json', truth):
            dest = output / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                with dest.open('wb') as stream:
                    subprocess.run(['git', f'--git-dir={mirror}', 'show', f"{info['commit']}:{name}"], stdout=stream, check=True)
        snapshots[hub] = output
    results = []
    for source in sorted(SOURCES.glob('*.yaml')):
        config = yaml.safe_load(source.read_text())
        hub = 'flusight' if 'flu ' in config['target'] else ('covid' if 'covid' in config['target'] else 'rsv')
        config['hub_path'] = str(snapshots[hub])
        config['output_path'] = str(ROOT / 'created' / source.stem)
        path = ROOT / 'configs' / source.name
        path.parent.mkdir(exist_ok=True)
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        command = [sys.executable, '-m', 'epibench', 'create', '--config-path', str(path)]
        log = ROOT / f'{source.stem}.log'
        with log.open('w') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        tasks = list(Path(config['output_path']).glob('*/task_list.csv'))
        row = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   config=str(path), command=command, target=config['target'],
                   hub_commit=frozen['hubs'][hub]['commit'], requested_dates=len(config['dates']),
                   first_date=config['dates'][0], last_date=config['dates'][-1],
                   status='created' if result.returncode == 0 else 'failed', log=str(log),
                   task_lists=[str(p) for p in tasks], n_created_dates=sum(len(pd.read_csv(p)) for p in tasks),
                   error_tail=log.read_text().splitlines()[-5:] if result.returncode else [])
        results.append(row)
        (ROOT / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
