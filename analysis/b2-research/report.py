"""Matched exploratory research report from completed manager score tables.

Run after planner rank. Partial rankings are allowed: only existing matched seeds
enter effects, and both run and comparison completeness are reported explicitly.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_EXPERIMENT = 'b2-direct-research-v1'
DIMENSIONS = ['geography', 'season', 'target']


def read_scores(ranking):
    """Use manager-computed ratios; never reconstruct scientific score weights."""
    runs = pd.read_csv(ranking / 'run_scores.csv')
    seasons = pd.read_csv(ranking / 'season_composite_scores.csv')
    cells = pd.read_csv(ranking / 'season_scores.csv')
    tables = []
    for frame, keys in [(runs, ['config_id', 'seed', 'geography']),
                        (seasons, ['config_id', 'seed', 'geography', 'season'])]:
        if frame.duplicated(keys).any():
            raise ValueError(f'Duplicate manager score keys: {keys}')
        value_columns = ['combined'] + [c for c in frame if c.startswith('wk inc ')]
        long = frame.melt(id_vars=keys, value_vars=value_columns,
                          var_name='target', value_name='value')
        if 'season' not in long:
            long['season'] = 'all'
        # Per-season target ratios come from season_scores below, not twice.
        if 'season' in keys:
            long = long[long.target == 'combined']
        tables.append(long)
    keys = ['config_id', 'seed', 'geography', 'season', 'target']
    if cells.duplicated(keys).any():
        raise ValueError('Duplicate target-season score keys')
    tables.append(cells[keys + ['wis_ratio', 'n', 'locations']].rename(columns={'wis_ratio': 'value'}))
    scores = pd.concat(tables, ignore_index=True).dropna(subset=['value'])
    if scores.duplicated(keys).any() or not np.isfinite(scores.value).all():
        raise ValueError('Nonunique or nonfinite score rows')
    return runs, scores, cells


def comparisons(design):
    """Pair models differing in the named factor, using the frozen design map."""
    rows = design['rows']
    index = {}
    for row in rows:
        key = (row['backbone'], row['sources'], row['spatial'], row['location_embedding'])
        if key in index:
            raise ValueError(f'Duplicate design coordinates: {key}')
        index[key] = row
    output = []

    def add(kind, row, baseline, label):
        if baseline is None:
            raise ValueError(f'Missing planned comparator: {kind}, {row["scenario"]}')
        output.append(dict(comparison_id=f'c{len(output):03d}', kind=kind,
                           backbone=row['backbone'], sources=row['sources'], spatial=row['spatial'],
                           location_embedding=row['location_embedding'], label=label,
                           config_id=row['scenario'], baseline_config_id=baseline['scenario']))

    for row in rows:
        b, source, spatial, embedding = (row[k] for k in
                                         ['backbone', 'sources', 'spatial', 'location_embedding'])
        if source != 'none':
            add('source-addition', row, index.get((b, 'none', spatial, embedding)), source)
        if source.startswith('all-minus-'):
            # Negative means adding the omitted source helps conditional on others.
            add('source-conditional-addition', index.get((b, 'all', spatial, embedding)), row,
                source.removeprefix('all-minus-'))
        if spatial != 'none':
            add('spatial', row, index.get((b, source, 'none', embedding)), f'{source} / {spatial}')
        if embedding:
            add('embedding', row, index.get((b, source, spatial, 0)), f'{source} / {spatial}')
    return pd.DataFrame(output)


def pair_scores(scores, contrasts, seeds):
    pairs, statuses = [], []
    key = ['seed', *DIMENSIONS]
    for contrast in contrasts.to_dict('records'):
        candidate = scores[scores.config_id == contrast['config_id']].drop(columns='config_id')
        baseline = scores[scores.config_id == contrast['baseline_config_id']].drop(columns='config_id')
        pair = candidate.merge(baseline, on=key, suffixes=('', '_baseline'), validate='one_to_one')
        for col in ('n', 'locations'):
            check = pair[col].notna() & pair[f'{col}_baseline'].notna()
            if not np.allclose(pair.loc[check, col], pair.loc[check, f'{col}_baseline']):
                raise ValueError(f'Score support differs: {contrast["comparison_id"]}, {col}')
        if (pair.value_baseline <= 0).any():
            raise ValueError('Nonpositive baseline WIS ratio; relative effect undefined')
        pair['delta'] = pair.value - pair.value_baseline
        pair['percent'] = 100 * (pair.value / pair.value_baseline - 1)
        pair['improved'] = pair.delta < 0
        for name, value in contrast.items():
            pair[name] = value
        pairs.append(pair)
        for seed in seeds:
            present = lambda frame: bool(((frame.seed == seed) & (frame.geography == 'all') &
                                          (frame.season == 'all') & (frame.target == 'combined')).any())
            has_candidate, has_baseline = present(candidate), present(baseline)
            statuses.append(dict(**contrast, seed=seed, candidate_present=has_candidate,
                                 baseline_present=has_baseline, paired=has_candidate and has_baseline))
    paired = pd.concat(pairs, ignore_index=True)
    group = ['comparison_id', *DIMENSIONS]
    summary = paired.groupby(group, as_index=False).agg(
        paired_seeds=('seed', 'nunique'), candidate_mean=('value', 'mean'),
        baseline_mean=('value_baseline', 'mean'), delta_mean=('delta', 'mean'),
        delta_sd=('delta', 'std'), percent_mean=('percent', 'mean'),
        percent_sd=('percent', 'std'), percent_min=('percent', 'min'),
        percent_max=('percent', 'max'), seeds_improved=('improved', 'sum'))
    summary = summary.merge(contrasts, on='comparison_id', validate='many_to_one')
    summary['expected_seeds'] = len(seeds)
    summary['complete'] = summary.paired_seeds == len(seeds)
    summary['uncertainty'] = 'Across-seed SD; exploratory, not a confidence interval'
    return paired, summary, pd.DataFrame(statuses)


def figures(summary, out):
    main = summary[(summary.geography == 'all') & (summary.season == 'all') &
                   (summary.target == 'combined')]
    paths = []
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    for kind in ('source-addition', 'source-conditional-addition', 'spatial', 'embedding'):
        selected = main[main.kind == kind].copy()
        # Source attribution plot uses independent, no-ID models; CSV keeps all controls.
        if kind == 'source-addition':
            selected = selected[(selected.spatial == 'none') & (selected.location_embedding == 0)]
        if kind == 'spatial':
            selected = selected[selected.location_embedding == 0]
        if selected.empty:
            continue
        backbones = sorted(selected.backbone.unique())
        labels = sorted(selected.label.unique())
        fig, axes = plt.subplots(1, len(backbones), figsize=(5 * len(backbones), max(4, .34 * len(labels))),
                                 sharey=True, squeeze=False, layout='constrained')
        for ax, backbone in zip(axes[0], backbones):
            part = selected[selected.backbone == backbone].set_index('label')
            for y, label in enumerate(labels):
                if label not in part.index:
                    continue
                row = part.loc[label]
                color = '#2166ac' if row.percent_mean < 0 else '#b2182b'
                ax.errorbar(row.percent_mean, y, xerr=0 if pd.isna(row.percent_sd) else row.percent_sd,
                            fmt='o' if row.complete else 's', color=color, capsize=2, markersize=4)
                ax.annotate(f' n={int(row.paired_seeds)}', (row.percent_mean, y),
                            xytext=(5, 3), textcoords='offset points', fontsize=7)
            ax.axvline(0, color='gray', linewidth=.8)
            ax.set_title(backbone)
            ax.set_xlabel('Paired WIS change (%)\nNegative is better')
            ax.set_yticks(range(len(labels)), labels)
        axes[0, 0].invert_yaxis()
        fig.suptitle(f'{kind}: exploratory paired seed means ± seed SD\n'
                     'Squares: incomplete seed pairs. Bars are not confidence intervals.', fontsize=11)
        filename = f'{kind}.png'
        fig.savefig(out / filename, dpi=170)
        plt.close(fig)
        paths.append(filename)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default=DEFAULT_EXPERIMENT)
    parser.add_argument('--ranking', type=Path, help='Default: latest ranking under the experiment')
    parser.add_argument('--design', type=Path, help='Default: experiment/research-design.json')
    parser.add_argument('--out', type=Path, default=Path('docs/results/b2-direct-research-v1'))
    parser.add_argument('--schema-check', action='store_true', help='Read/validate ranking CSV schema only')
    args = parser.parse_args()
    folder = Path('data/experiments') / args.experiment
    ranking = args.ranking
    if ranking is None:
        rankings = list(folder.glob('ranking-*'))
        if not rankings:
            raise SystemExit('No ranking found; run planner rank first (use --allow-incomplete for progress).')
        ranking = max(rankings, key=lambda p: p.stat().st_mtime)
    runs, scores, cells = read_scores(ranking)
    if args.schema_check:
        print(json.dumps(dict(ranking=str(ranking), run_rows=len(runs), decomposed_score_rows=len(scores))))
        return
    design_path = args.design or folder / 'research-design.json'
    design = json.loads(design_path.read_text())
    seeds = design['seeds']
    wanted = {row['scenario'] for row in design['rows']}
    if set(runs.config_id) - wanted:
        raise ValueError('Ranking contains configurations outside the supplied frozen design')
    if set(runs.seed) - set(seeds):
        raise ValueError('Ranking contains seeds outside the supplied frozen design')
    contrasts = comparisons(design)
    paired, summary, comparison_status = pair_scores(scores, contrasts, seeds)
    present = set(map(tuple, runs[runs.geography == 'all'][['config_id', 'seed']].values))
    run_status = pd.DataFrame([dict(**{k: v for k, v in row.items() if k != 'families'}, seed=seed,
                                    complete=(row['scenario'], seed) in present)
                               for row in design['rows'] for seed in seeds])
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in [('comparisons', contrasts), ('paired_seed_scores', paired),
                        ('matched_summary', summary), ('comparison_status', comparison_status),
                        ('run_status', run_status)]:
        frame.to_csv(out / f'{name}.csv', index=False)
    coverage = cells[[c for c in cells if c in ['config_id', 'seed', 'geography', 'season', 'target']
                      or 'coverage_' in c]]
    coverage.to_csv(out / 'target_season_coverage.csv', index=False)
    charts = figures(summary, out)
    complete = int(run_status.complete.sum())
    status = 'COMPLETE' if run_status.complete.all() else 'INCOMPLETE — exploratory progress snapshot'
    fingerprint = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    provenance = dict(experiment=args.experiment, ranking=str(ranking), design=str(design_path),
                      design_sha256=fingerprint(design_path), complete_runs=complete,
                      planned_runs=len(run_status), expected_seeds=seeds, status=status,
                      inputs={p.name: fingerprint(p) for p in ranking.glob('*.csv')},
                      percent='Mean of 100*(candidate/baseline - 1), paired on seed and score support.',
                      uncertainty='Across-seed SD, not a confidence interval or epidemiological replication.')
    (out / 'research-provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    text = [f'# Covariates and information sharing: matched research report\n\n**{status}.** '
            f'{complete}/{len(run_status)} planned seed runs represented in this ranking. '
            f'Expected seeds: {", ".join(map(str, seeds))}. Evaluation uses {design["eval_members"]} samples.\n',
            'All effects compare existing matched seeds; an unfinished candidate or control contributes '
            'no pair. Missing pairs are listed explicitly. Partial comparisons may contain different seed '
            'subsets and should not be ranked against one another.\n',
            'Negative WIS change favors the candidate. Source additions compare against no external '
            'covariates with the same backbone, exchange and embedding. Conditional source additions '
            'compare all sources against all-minus-one, so negative means the omitted source helps when '
            'added back. Spatial effects compare against no exchange with sources and embedding fixed. '
            'Embedding effects compare 8 dimensions against zero with everything else fixed.\n',
            'Percent effects are means of seedwise ratios, not ratios of separately averaged scores. '
            'Bars show across-seed standard deviations, not confidence intervals. Three seeds describe '
            'optimization variability; these reused historical seasons are exploratory development data. '
            'Source conclusions are conditional on the operational availability assumptions inferred '
            'from 2025–26: actual vintages are preferred, but missing historical vintages can use '
            'finalized proxies under assumed availability. This can introduce revision optimism, '
            'particularly for Kinsa and other short archives. Interpret effects alongside the '
            '[study availability policy](../../design/b2-direct-research.md#reporting-availability-and-support) '
            'and source-specific substitution counts; this is not a fully reconstructed real-time backtest.\n',
            '[All planned comparisons](comparisons.csv) · [Run completeness](run_status.csv) · '
            '[Pair completeness](comparison_status.csv) · [Paired seed scores](paired_seed_scores.csv) · '
            '[Paired means and seed dispersion](matched_summary.csv) · '
            '[Target-season coverage](target_season_coverage.csv) · '
            '[Provenance](research-provenance.json).\n',
            'The paired tables decompose overall, states/DC and US scores, season composites, target '
            'means, and target-by-season scores. Target means are diagnostics and must not be averaged '
            'to reconstruct the season-weighted combined score. Available benchmark targets differ '
            'between seasons.\n']
    text += [f'![{path.removesuffix(".png")}](./{path})\n' for path in charts]
    if not charts:
        text.append('No complete candidate/control seed pair is available yet.\n')
    (out / 'research.md').write_text('\n'.join(text))
    print(json.dumps(dict(report=str(out / 'research.md'), complete_runs=complete,
                          paired_rows=len(paired), comparisons=len(contrasts), status=status)))


if __name__ == '__main__':
    main()
