"""Materialize CSV companions for an existing Hubverse Parquet archive."""
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path, help='hubverse directory, one hub, or one model directory')
    args = parser.parse_args()
    files = sorted(args.archive.rglob('*.parquet'))
    if not files:
        parser.error('No Parquet forecast files found')
    for i, file in enumerate(files, 1):
        pd.read_parquet(file).to_csv(file.with_suffix('.csv'), index=False)
        if i % 450 == 0:
            print(f'Converted {i}/{len(files)} forecast files', flush=True)
    print(f'Complete: {len(files)} CSV companions', flush=True)


if __name__ == '__main__':
    main()
