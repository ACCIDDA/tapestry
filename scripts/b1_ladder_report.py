"""Paired seed differences across the B1 ladder, against the `B1-fromB0-refit` control.

The four experiments share B0's rank-1 configuration, the dataset, the folds, the
refit procedure and the frozen scoring support; they differ only in artificial
masking and the nowcast stage. Every comparison here is therefore **paired within a
seed**: the same seed fitted under two conditions, differenced, then summarized
across seeds. Unpaired means across a 3-seed sample would hide that the seed spread
within a configuration is comparable to the effects being measured.

Reported per arm:

- combined score under `natural` inputs, the headline;
- the same under each missing-input stress, because masking's benefit is supposed to
  appear when inputs go missing, not when they are all present;
- per-season and per-target paired differences;
- nowcast skill for the two-stage arms, which the scorer already computes with
  visible supplied finals excluded, so copying a known answer cannot score as
  reconstruction.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONTROL = 'B1-fromB0-refit'
ARMS = (CONTROL, 'B1-onlymask-refit', 'B1-onlynowcast-refit', 'B1-full-refit')
LABEL = {CONTROL: 'direct, no mask', 'B1-onlymask-refit': 'direct, mask .5',
         'B1-onlynowcast-refit': 'two-stage, no mask', 'B1-full-refit': 'two-stage, mask .5'}


def latest_ranking(experiment):
    """The most recent ranking directory written for an experiment."""
    folders = sorted(Path('data/experiments', experiment).glob('ranking-*'),
                     key=lambda p: p.stat().st_mtime)
    if not folders:
        raise FileNotFoundError(f'{experiment} has no ranking-* directory; run manager rank first')
    return folders[-1]


def run_scores(experiment, geography='all'):
    """Per-seed combined scores for one geography aggregation.

    `run_scores.csv` carries one row per seed *and* geography ('US', 'states_dc',
    'all'); selecting one keeps a seed a single observation, so paired differences
    are over seeds rather than over seed-geography pairs.
    """
    frame = pd.read_csv(latest_ranking(experiment) / 'run_scores.csv')
    frame = frame[frame.geography == geography].copy()
    if frame.seed.duplicated().any():
        raise ValueError(f'{experiment}: repeated seeds within geography={geography}')
    frame['experiment'] = experiment
    return frame


def season_scores(experiment):
    frame = pd.read_csv(latest_ranking(experiment) / 'season_composite_scores.csv')
    frame['experiment'] = experiment
    return frame


def paired(values, control, key):
    """Differences of `values` against `control` on shared seeds only.

    Returns (n_pairs, mean difference, sd of differences). An unpaired seed is
    dropped rather than compared against a different seed's control, which would
    add seed noise to the effect being measured.
    """
    shared = sorted(set(values[key]) & set(control[key]))
    if not shared:
        return 0, np.nan, np.nan
    a = values.set_index(key).loc[shared, 'combined']
    b = control.set_index(key).loc[shared, 'combined']
    diff = (a - b).astype(float)
    return len(shared), diff.mean(), diff.std(ddof=1) if len(shared) > 1 else np.nan


def stress_table(experiment):
    """Combined score per stress condition, per seed, from the saved score files.

    The ranking aggregates `natural` only, so the stress conditions are recomputed
    here from the per-fold score parquet using the same weights the ranker uses.
    """
    from tapestry.evaluation.totals import TARGET_WEIGHTS
    from tapestry.models.objective import US_WEIGHT
    rows = []
    for parquet in sorted(Path('data/experiments', experiment).glob(
            '*/s*/attempt-*/b1/eval_*/scores-*.parquet')):
        frame = pd.read_parquet(parquet)
        for (stress, seed), part in frame.groupby(['stress', 'seed']):
            forecast = part[part.task == 'forecast']
            if forecast.empty:
                continue
            weighted = []
            for geography, gw in (('states_dc', 1 - US_WEIGHT), ('US', US_WEIGHT)):
                g = forecast[forecast.geography == geography]
                if g.empty:
                    continue
                by_target = g.groupby('target').wis.mean()
                weighted.append(gw * by_target.mean())
            rows.append(dict(experiment=experiment, seed=int(seed), stress=stress,
                             season=parquet.parent.name.removeprefix('eval_'),
                             wis=float(np.sum(weighted))))
    return pd.DataFrame(rows)


def nowcast_support(experiment):
    """Nowcast skill for two-stage arms, as written by the fold scorer."""
    rows = []
    for path in sorted(Path('data/experiments', experiment).glob(
            '*/s*/attempt-*/b1/eval_*/nowcast-support.json')):
        payload = json.loads(path.read_text())
        rows.append(dict(experiment=experiment, season=path.parent.name.removeprefix('eval_'),
                         **{k: v for k, v in payload.items() if not isinstance(v, (list, dict))}))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--arms', nargs='+', default=list(ARMS))
    parser.add_argument('--control', default=CONTROL)
    parser.add_argument('--output', type=Path, help='Optional JSON summary path')
    args = parser.parse_args()

    # A planned-but-unranked experiment has a directory but nothing to read yet.
    available = [a for a in args.arms
                 if any(Path('data/experiments', a).glob('ranking-*'))]
    missing = [a for a in args.arms if a not in available]
    if missing:
        print(f'Not yet ranked, skipped: {", ".join(missing)}\n')
    if args.control not in available:
        raise SystemExit(f'Control {args.control} is not ranked yet; nothing to compare against.')
    control = run_scores(args.control)

    print('=' * 78)
    print('Combined score, natural inputs (lower is better; paired against control)')
    print('=' * 78)
    rows = []
    for arm in available:
        scores = run_scores(arm)
        n, mean, sd = paired(scores, control, 'seed')
        rows.append(dict(arm=arm, condition=LABEL.get(arm, ''), seeds=len(scores),
                         combined=scores.combined.mean(),
                         spread=scores.combined.std(ddof=1) if len(scores) > 1 else np.nan,
                         paired_n=n, paired_diff=0. if arm == args.control else mean,
                         paired_sd=np.nan if arm == args.control else sd))
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print('\n`paired_diff` is mean(arm - control) over shared seeds; negative favours the arm.')

    print('\n' + '=' * 78)
    print('Per-season paired differences (geography=all)')
    print('=' * 78)
    control_season = season_scores(args.control)
    for arm in available:
        if arm == args.control:
            continue
        frame = season_scores(arm)
        parts = []
        for season in sorted(frame.season.unique()):
            a = frame[(frame.season == season) & (frame.geography == 'all')]
            b = control_season[(control_season.season == season) & (control_season.geography == 'all')]
            n, mean, sd = paired(a, b, 'seed')
            parts.append(dict(season=season, pairs=n, diff=mean, sd=sd))
        print(f'\n{arm} ({LABEL.get(arm, "")})')
        print(pd.DataFrame(parts).to_string(index=False, float_format=lambda x: f'{x:+.4f}'))

    print('\n' + '=' * 78)
    print('Per-target paired differences (natural inputs)')
    print('=' * 78)
    targets = [c for c in control.columns if c.startswith('wk inc')]
    for arm in available:
        if arm == args.control:
            continue
        scores = run_scores(arm)
        shared = sorted(set(scores.seed) & set(control.seed))
        if not shared:
            continue
        parts = []
        for target in targets:
            a = scores.set_index('seed').loc[shared, target].astype(float)
            b = control.set_index('seed').loc[shared, target].astype(float)
            d = a - b
            parts.append(dict(target=target, diff=d.mean(),
                              sd=d.std(ddof=1) if len(shared) > 1 else np.nan))
        print(f'\n{arm} ({LABEL.get(arm, "")})')
        print(pd.DataFrame(parts).to_string(index=False, float_format=lambda x: f'{x:+.4f}'))

    print('\n' + '=' * 78)
    print('Missing-input stress: mean WIS by condition (masking should pay off here)')
    print('=' * 78)
    stress = pd.concat([stress_table(a) for a in available], ignore_index=True)
    if not stress.empty:
        pivot = stress.groupby(['experiment', 'stress']).wis.mean().unstack()
        pivot = pivot.reindex([a for a in available if a in pivot.index])
        print(pivot.to_string(float_format=lambda x: f'{x:.2f}'))
        base = pivot.loc[args.control]
        print('\nRelative to control (ratio; <1 favours the arm):')
        print((pivot / base).to_string(float_format=lambda x: f'{x:.3f}'))

    print('\n' + '=' * 78)
    print('Nowcast skill, visible supplied finals excluded by the scorer')
    print('=' * 78)
    nowcast = pd.concat([nowcast_support(a) for a in available], ignore_index=True)
    if nowcast.empty:
        print('No nowcast-support.json found; the direct arms do not nowcast.')
    else:
        print(nowcast.to_string(index=False))

    if args.output:
        args.output.write_text(json.dumps(dict(
            control=args.control, arms=available,
            combined=table.to_dict('records'),
            stress=stress.to_dict('records') if not stress.empty else [],
        ), indent=2) + '\n')
        print(f'\nWrote {args.output}')


if __name__ == '__main__':
    main()
