"""Item 2 of B4.refineTop2: combine saved forecasts of already-fitted models, no retraining.

Each member run saved 23 quantiles per issuance/horizon/target/location for three evaluation
views (raw reports, corrected newest admissions, 50/50 forecast mixture). An ensemble combines
members view by view with equal weight per member, by one of two rules:
  vincent  - average the quantiles level by level (quantile averaging);
  mixture  - equal-weight mixture of the members' predictive CDFs (each member's quantile
             function linearly interpolated, constant beyond the 1%/99% levels, as in
             `training.mixture_quantiles`).
Admission quantiles are re-rounded to integers. The combined runs are scored by the same
`rank_pilot` used for every pilot/B4 run. Members and groups are fixed below, chosen from
B4 results that already used both evaluation seasons; ensemble gains are development
results, not untouched-season results."""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from tapestry.evaluation.quantiles import LEVELS
from tapestry.experiment.pilot import rank_pilot
from tapestry.experiment.provenance import save

ROOT = Path('/proj/jlessler/projects/tapestry-all')
HEADS = ROOT / 'tapestry-b4-flu-heads-20261006/data/experiments/b4-flu-heads-20261006'
SWEEP = ROOT / 'tapestry-b4-flu-600-20261005/data/experiments/b4-flu-600-20261005'
REFINE = ROOT / 'tapestry-b4-refinetop2-20261006/data/experiments/b4-refinetop2-20261006'
POLISH = ROOT / 'tapestry-b5b4polish-20261006/data/experiments/b4-polish-20261006'
SEASONS = ('2024-2025', '2025-2026')
VIEWS = {'raw': '', 'half': 'half', 'corrected': 'corrected'}
U = (np.arange(256) + .5) / 256
W = np.stack([np.interp(U, LEVELS, np.eye(len(LEVELS))[k]) for k in range(len(LEVELS))], 1)  # [u, levels]


def runs(experiment, scenario, count=2):
    t = pd.read_csv(experiment / 'runs.csv')
    t = t[(t.scenario == scenario) & (t.status == 'complete')].sort_values('seed')
    assert len(t) == count, (experiment, scenario, len(t))
    return [experiment / a for a in t.attempt]


def combine(qs, rule):
    if rule == 'vincent':
        q = np.mean(qs, 0)
    else:
        draws = np.concatenate([np.tensordot(W, x, axes=(1, 0)) for x in qs], 0)
        q = np.quantile(draws, LEVELS, axis=0)
    q[:, :, :, :3] = np.floor(q[:, :, :, :3] + .5)
    return q


def member_file(m, season, sub):
    # nokinsa exists only for members with covariates; members without any covariate are unaffected.
    path = m / f'eval_{season}' / sub / 'forecasts.npz'
    return path if path.exists() or sub != 'nokinsa' else m / f'eval_{season}' / 'corrected' / 'forecasts.npz'


def build(name, members, rule, out, views=VIEWS):
    run = out / f'{name}-{rule}'; run.mkdir(parents=True, exist_ok=True)
    config = f'ensemble={name},rule={rule},forecast_targets=flu'
    first = json.loads((members[0] / 'manifest.json').read_text())
    save(run / 'manifest.json', dict(scenario=first['scenario'], folds=list(SEASONS), ensemble=name, rule=rule,
                                     members=[str(m) for m in members], definition=__doc__))
    for season in SEASONS:
        for view, sub in views.items():
            data = [dict(np.load(member_file(m, season, sub))) for m in members]
            # Members can differ by an issuance at the season edge (reconstruction training
            # changes the first usable origin); keep the shared issuances. The scorer still
            # requires every frozen Hub task, so a dropped scored task would fail loudly.
            common = sorted(set.intersection(*(set(d['context_end'].tolist()) for d in data)))
            for d in data:
                keep = np.isin(d['context_end'], common)
                for k, v in list(d.items()):
                    if k not in ('quantiles', 'quantile_levels', 'locations') and v.ndim and len(v) == len(keep):
                        d[k] = v[keep]
                d['quantiles'] = d['quantiles'][:, keep]
            for k in ('truth', 'mask', 'context_end', 'locations', 'target_dates'):
                assert all(np.array_equal(d[k], data[0][k]) for d in data), (name, season, view, k)
            merged = {k: data[0][k] for k in data[0] if k != 'flu_admission_sum_quantiles'}
            merged['quantiles'] = combine([d['quantiles'] for d in data], rule)
            folder = run / f'eval_{season}' / sub; folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / 'forecasts.npz', **merged)
        manifest = run / f'eval_{season}' / 'manifest.json'
        if not manifest.exists():
            manifest.write_text((members[0] / f'eval_{season}' / 'manifest.json').read_text())
    return dict(path=str(run), config_id=config, seed=0)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', default='data/experiments/b4-refinetop2-ensembles')
    p.add_argument('--set', choices=['b4', 'polish', 'polish5'], default='b4')
    p.add_argument('--frozen', default='data/evaluation/b0_hub_comparison_q23'); args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    heads = Path('docs/experiments/b4-flu-heads-20261006/scenarios.txt').read_text().split('\n')
    a, b_samples_total, b = runs(HEADS, heads[0]), runs(HEADS, heads[5]), runs(HEADS, heads[6])
    ranking = pd.read_csv('docs/experiments/b4-flu-600-20261005/final-analysis/pilot-rankings.csv')
    best = ranking[(ranking['count'] == 2) & (ranking.history == 'corrected') & (ranking.metric == 'flu_native')
                   & (ranking.season == 'equal_season_mean')].sort_values('mean')
    top = [runs(SWEEP, c) for c in best.config_id.head(10)]
    if args.set == 'polish5':
        # B4.polish part 2: seeds 44-48. "all" = B4 folds; "recent2" = two preceding seasons only.
        design = json.loads(Path('docs/experiments/b4-polish-20261006/design.json').read_text())['rows']
        get = lambda recipe, window: runs(POLISH, next(r['scenario'] for r in design if r['recipe'] == recipe
                                                      and r['training_window'] == window), 5)
        groups = {}
        for w in ('all', 'recent2'):
            A5, B5, F5 = get('A', w), get('B', w), get('B_finalized', w)
            groups.update({f'{w}_A5': A5, f'{w}_B5': B5, f'{w}_Bfinalized5': F5, f'{w}_A5_B5': A5 + B5,
                           f'{w}_A5_B15': A5 + B5 * 3, f'{w}_A5_B5_Bfinalized5': A5 + B5 + F5})
            # Seed-subset check of A's value: pairs of single seeds.
            for i in range(5):
                groups[f'{w}_A1_B1_seed{44 + i}'] = [A5[i], B5[i]]
                groups[f'{w}_B1_seed{44 + i}'] = [B5[i]]
                groups[f'{w}_A1_seed{44 + i}'] = [A5[i]]
    elif args.set == 'polish':
        # B4.polish stress tests. Weights by repetition: 25% A + 75% B repeats each B run 3 times.
        design = json.loads(Path('docs/experiments/b4-refinetop2-20261006/design.json').read_text())['rows']
        find = lambda recipe, treatment: runs(REFINE, next(r['scenario'] for r in design if r['recipe'] == recipe
                                                             and r['treatment'] == treatment and r['sum_wis_weight'] == 0))
        a_fin, b_fin = find('A', 'finalized'), find('B', 'finalized')
        groups = {'B_only': b, 'A25_B75': a + b * 3, 'A50_B50': a + b, 'Afinalized50_B50': a_fin + b,
                  'Afinalized25_B75': a_fin + b * 3, 'Bfinalized_only': b_fin, 'B_Bfinalized': b + b_fin,
                  'A_B_Bfinalized': a + b + b_fin}
    else:
      groups = {
        'A_two_seeds': a,
        'B_two_seeds': b,
        'Bsum_two_seeds': b_samples_total,
        'A_B': a + b,
        'A_B_Bsum': a + b + b_samples_total,
        'sweep_top5': [r for pair in top[:5] for r in pair],
        'sweep_top10': [r for pair in top for r in pair],
      }
    save(out / 'groups.json', {k: [str(m) for m in v] for k, v in groups.items()})
    views = dict(VIEWS, delayed='delayed', nokinsa='nokinsa') if args.set == 'polish5' else VIEWS
    rules = ('vincent',) if args.set == 'polish5' else ('vincent', 'mixture')
    built = [build(name, members, rule, out, views) for name, members in groups.items() for rule in rules]
    destination = out / 'ranking'; destination.mkdir(exist_ok=True)
    rank_pilot(built, args.frozen, destination)


if __name__ == '__main__':
    main()
