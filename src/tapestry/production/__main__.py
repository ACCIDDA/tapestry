"""Weekly submission commands; nothing is published (opening the Hub PR is a separate, human step).

    python -m tapestry.production run --release production/releases/b7-20261007.json \
        --dataset data/operational-<date>/processed/panel.npz --issuance <wednesday>
    python -m tapestry.production record production/output/<wednesday>/<release>/<reference>-ACCIDDA-<model>.csv
    python -m tapestry.production intervals A.csv [B.csv --labels A B]
    python -m tapestry.production peers A.csv [B.csv ...]
    python -m tapestry.production diff SUBMITTED.csv NEW.csv OUTDIR [--labels A B]

`run` replays every released checkpoint (`forecast.py`), combines and exports the Hub file
(`export.py`), then draws the interval PDFs (`intervals.py`) and the peer comparison
(`peers.py`). Output: production/output/<issuance>/<release name>/.

`record` keeps what was submitted, after the Hub PR is pushed: the CSV (renamed with
`--as` when the Hub file name differs), its export record, the interval PDFs and the peer
comparison PDF, in production/submissions/<reference date>/ (`--superseded`: its
superseded/ subfolder, for a file that was replaced). The PDFs are drawn from the local
Hub clone, so refresh it before submitting and record right after.
"""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from .forecast import read_release, forecast_release
from .export import export


def run(argv):
    p = argparse.ArgumentParser(prog='python -m tapestry.production run')
    p.add_argument('--release', type=Path, required=True)
    p.add_argument('--dataset', required=True, help='Operational panel.npz')
    p.add_argument('--issuance', required=True, help='Wednesday issuance, e.g. 2026-10-14')
    p.add_argument('--out', type=Path, help='Default: production/output/<issuance>/<release name>')
    p.add_argument('--hub', type=Path, default=Path('production/hubs/FluSight-forecast-hub'))
    p.add_argument('--device', default='cpu')
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--compare-csv', nargs='*', type=Path, default=[], help='Additional same-week local candidates in the peer comparison')
    p.add_argument('--no-plots', action='store_true')
    args = p.parse_args(argv)
    release = read_release(args.release)
    out = args.out or Path('production/output') / args.issuance / release['name']
    members = forecast_release(release, args.dataset, args.issuance, out / 'forecasts', args.device, args.jobs)
    csv = export(release, members, args.hub, out)
    if not args.no_plots:
        from . import intervals, peers
        intervals.main([str(csv), '--hub', str(args.hub)])
        peers.main([str(csv), *map(str, args.compare_csv), '--hub', str(args.hub)])
    print(csv)


def record(argv):
    p = argparse.ArgumentParser(prog='python -m tapestry.production record')
    p.add_argument('csv', type=Path, help='The submitted file')
    p.add_argument('--as', dest='name', help='File name used on the Hub, when it differs')
    p.add_argument('--superseded', action='store_true', help='A file that was replaced by a later one')
    p.add_argument('--hub', type=Path, default=Path('production/hubs/FluSight-forecast-hub'))
    p.add_argument('--hub-ref', default='HEAD', help='Hub commit for history and peers (default: the local clone)')
    args = p.parse_args(argv)
    from . import intervals, peers
    name = args.name or args.csv.name
    reference = name[:10]
    folder = Path('production/submissions') / reference / ('superseded' if args.superseded else '')
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    if target.exists() and target.read_bytes() != args.csv.read_bytes():
        raise ValueError(f'{target} exists with different content')
    shutil.copy2(args.csv, target)
    if args.csv.with_suffix('.json').exists():
        shutil.copy2(args.csv.with_suffix('.json'), target.with_suffix('.json'))
    intervals.main([str(target), '--hub', str(args.hub)])
    with tempfile.TemporaryDirectory() as scratch:
        peers.main([str(target), '--hub', str(args.hub), '--hub-ref', args.hub_ref, '--out', scratch])
        shutil.copy2(Path(scratch) / 'comparison.pdf', folder / f'{target.stem}-peers.pdf')
        shutil.copy2(Path(scratch) / 'manifest.json', folder / f'{target.stem}-peers.json')
    print('\n'.join(sorted(str(f) for f in folder.glob(f'{target.stem}*'))))


def main():
    commands = {'run': run, 'record': record}
    from . import intervals, peers, diff
    commands.update(intervals=intervals.main, peers=peers.main, diff=diff.main)
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        raise SystemExit(__doc__)
    commands[sys.argv[1]](sys.argv[2:])


if __name__ == '__main__':
    main()
