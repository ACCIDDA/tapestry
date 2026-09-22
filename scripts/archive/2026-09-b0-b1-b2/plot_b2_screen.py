"""Summarize a completed B2 screen from manager ranking tables; no refitting."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tapestry.models.b2_scenarios import B2Scenario, SCREEN_COVARIATE_SETS


LABELS = ['None', 'Inpatient', 'Outpatient', 'Wastewater level', 'Wastewater rank',
          'Both claims', 'Both wastewater', 'All groups']
PANELS = [('target', 'finalized'), ('target', 'wednesday'),
          ('pathogen', 'finalized'), ('pathogen', 'wednesday')]


def annotate(ax, values, labels, columns, title, *, delta=False):
    if delta:
        limit = max(float(np.nanmax(np.abs(values))), .01)
        view = ax.imshow(values, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
    else:
        view = ax.imshow(values, cmap='viridis_r', aspect='auto')
    ax.set_xticks(range(len(columns)), columns, fontsize=9)
    ax.set_yticks(range(len(labels)), labels, fontsize=10)
    ax.set_title(title, loc='left', pad=14)
    for (i, j), value in np.ndenumerate(values):
        color = 'white' if delta and abs(value) > limit * .6 else 'black'
        ax.text(j, i, f'{value:+.1f}%' if delta else f'{value:.3f}', ha='center', va='center',
                fontsize=9, color=color,
                bbox=dict(facecolor='white', alpha=.7, edgecolor='none', pad=1) if not delta else None)
    return view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranking', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ranked = pd.read_csv(args.ranking / 'configuration_ranking.csv')
    meta = []
    for config_id in ranked.config_id:
        scenario = B2Scenario.from_string(config_id)
        meta.append(dict(config_id=config_id, recipe=scenario.fit_partition,
                         input_mode=scenario.input_mode, covariate_set=scenario.covariate_set))
    meta = pd.DataFrame(meta)
    ranked = ranked.merge(meta, on='config_id', validate='one_to_one')
    if len(ranked) != 32 or not ranked.seeds.eq(1).all():
        raise ValueError('This report expects the complete 32-configuration, single-seed screen')
    controls = ranked[ranked.covariate_set.eq('none')][['recipe', 'input_mode', 'combined_mean']]
    controls = controls.rename(columns={'combined_mean': 'control_score'})
    ranked = ranked.merge(controls, on=['recipe', 'input_mode'], validate='many_to_one')
    ranked['change_percent'] = 100 * (ranked.combined_mean / ranked.control_score - 1)
    ranked.to_csv(args.output / 'matched_controls.csv', index=False)
    columns = [f'{r.title()}\n{m.title()}' for r, m in PANELS]
    scores = np.column_stack([ranked[(ranked.recipe == r) & (ranked.input_mode == m)]
                              .set_index('covariate_set').loc[list(SCREEN_COVARIATE_SETS), 'combined_mean']
                              for r, m in PANELS])
    delta = np.column_stack([ranked[(ranked.recipe == r) & (ranked.input_mode == m)]
                             .set_index('covariate_set').loc[list(SCREEN_COVARIATE_SETS), 'change_percent']
                             for r, m in PANELS])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), layout='constrained')
    annotate(axes[0], scores, LABELS, columns, 'Frozen-Hub relative WIS · lower is better')
    annotate(axes[1], delta, LABELS, columns, 'Change versus matched no-covariate control', delta=True)
    fig.suptitle('B2 covariate screen · seed 42\nSeason-equal objective; states/DC 80%, US 20%', fontsize=14)
    fig.savefig(args.output / 'covariate-screen.png', dpi=180)
    fig.savefig(args.output / 'covariate-screen.pdf')
    plt.close(fig)

    seasons = pd.read_csv(args.ranking / 'season_composite_scores.csv')
    seasons = seasons[seasons.geography.eq('all')].merge(meta, on='config_id', validate='many_to_one')
    baseline = seasons[seasons.covariate_set.eq('none')][['recipe', 'input_mode', 'seed', 'season', 'combined']]
    seasons = seasons.merge(baseline.rename(columns={'combined': 'control_score'}),
                            on=['recipe', 'input_mode', 'seed', 'season'], validate='many_to_one')
    seasons['change_percent'] = 100 * (seasons.combined / seasons.control_score - 1)
    seasons.to_csv(args.output / 'season_matched_controls.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), layout='constrained')
    for ax, (recipe, mode) in zip(axes.flat, PANELS):
        part = seasons[(seasons.recipe == recipe) & (seasons.input_mode == mode)]
        table = part.pivot(index='covariate_set', columns='season', values='change_percent')
        table = table.loc[list(SCREEN_COVARIATE_SETS)].sort_index(axis=1)
        annotate(ax, table.to_numpy(), LABELS, list(table.columns), f'{recipe.title()} · {mode}', delta=True)
    fig.suptitle('Per-season change versus matched control (%)\nNegative values improve WIS; panels use separate color scales', fontsize=14)
    fig.savefig(args.output / 'season-effects.png', dpi=180)
    fig.savefig(args.output / 'season-effects.pdf')
    plt.close(fig)

    (args.output / 'manifest.json').write_text(json.dumps(dict(ranking=str(args.ranking), configurations=32,
                                                              seeds=[42], plots=['covariate-screen', 'season-effects']), indent=2) + '\n')
    print(f'Wrote B2 figures and score tables to {args.output}')


if __name__ == '__main__':
    main()
