"""Score saved B0 forecasts with nine concurrent EpiBench cases."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

from tapestry.models import manager
from tapestry.evaluation import sweep

WORKERS = 9


class ScoringPool(ThreadPoolExecutor):
    def __init__(self, max_workers=None):
        super().__init__(max_workers=WORKERS)


def export_rows(frames, model, output, csv=False):
    """Return the row count for existing Hubverse exports."""
    return sum(len(frame) * len(sweep.QCOLS) for frame in frames.values())


def main():
    folder = Path('data/experiments/b0-rebuilt').resolve()
    with manager.locked(folder):
        runs = json.loads((folder / 'runs.json').read_text())
        if not runs or any(run['status'] != 'complete' for run in runs.values()):
            raise ValueError('Complete all registered runs before scoring')
        protocol = json.loads((folder / 'protocol.json').read_text())
        outputs = [run['output'] for run in runs.values()]
        signature = hashlib.sha256(json.dumps(sorted(outputs)).encode()).hexdigest()[:12]
        output = folder / f'comparison-{signature}'
        output.mkdir(exist_ok=True)
        command = [sys.executable, '-m', 'tapestry.evaluation.sweep', '--runs', *outputs,
                   '--frozen', protocol['frozen'], '--output', str(output)]
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        for case in (output / 'epibench').glob('*'):
            if case.is_dir() and not ((case / 'provenance.json').is_file() and
                                      (case / 'output/EpiBenchmark_scores.csv').is_file()):
                archive = output / 'interrupted-scoring' / stamp
                archive.mkdir(parents=True, exist_ok=True)
                case.rename(archive / case.name)
        execution = dict(script_sha256=manager.sha256(__file__), workers=WORKERS,
                         job_id=os.environ.get('SLURM_JOB_ID'),
                         node=os.environ.get('SLURMD_NODENAME'), started=manager.now())
        manager.save(output / f'parallel-execution-{stamp}.json', execution)
        state = dict(status='running', command=command, output=str(output), execution=execution)
        manager.save(folder / 'comparison.json', state)
        try:
            print(f'Loading forecasts with {WORKERS} workers', flush=True)
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                frames = dict(zip(outputs, pool.map(sweep.export_b0, outputs)))
            # Keep execution settings outside the protocol-hashed package files.
            sweep.export_b0 = lambda run: frames[str(run)]
            sweep.ThreadPoolExecutor = ScoringPool
            if (output / 'hubverse').is_dir():
                sweep.hubverse = export_rows
            sys.argv = ['tapestry.evaluation.sweep', *command[3:]]
            sweep.main()
        except BaseException as error:
            state.update(status='failed', error=str(error))
            manager.save(folder / 'comparison.json', state)
            raise
        state.update(status='complete', finished=manager.now())
        manager.save(folder / 'comparison.json', state)


if __name__ == '__main__':
    main()
