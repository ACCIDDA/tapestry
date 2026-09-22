"""Summarize a completed B2-kinsa experiment from manager ranking tables; no refitting."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tapestry.models.b2_scenarios import B2Scenario

ARMS = [('target', 'finalized'), ('target', 'wednesday'), ('pathogen', 'finalized'), ('pathogen', 'wednesday')]
SCORES = {'all': 'combined_mean', 'states/DC': 'states_dc_combined_mean', 'US': 'US_combined_mean'}
TARGETS = ['wk inc flu hosp', 'wk inc covid hosp', 'wk inc rsv hosp',
           'wk inc flu prop ed visits', 'wk inc covid prop ed visits', 'wk inc rsv prop ed visits']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ranking', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ranked = pd.read_csv(args.ranking / 'configuration_ranking.csv')
    meta = pd.DataFrame([dict(config_id=c, recipe=s.fit_partition, input_mode=s.input_mode, covariate_set=s.covariate_set)
                         for c in ranked.config_id for s in [B2Scenario.from_string(c)]])
    ranked = ranked.merge(meta, on='config_id', validate='one_to_one')
    if len(ranked) != 8 or not ranked.seeds.eq(1).all() or set(ranked.covariate_set) != {'none', 'kinsa'}:
        raise ValueError('This report expects the complete 8-configuration, single-seed B2-kinsa experiment')

    # One row per arm: matched no-covariate control, Kinsa run, and the change.
    rows = []
    for recipe, mode in ARMS:
        arm = ranked[(ranked.recipe == recipe) & (ranked.input_mode == mode)].set_index('covariate_set')
        row = dict(recipe=recipe, input_mode=mode)
        for name, column in {**SCORES, **{t: f'{t}_mean' for t in TARGETS}}.items():
            row[f'{name} · none'] = arm.loc['none', column]
            row[f'{name} · kinsa'] = arm.loc['kinsa', column]
            row[f'{name} · change %'] = 100 * (arm.loc['kinsa', column] / arm.loc['none', column] - 1)
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(args.output / 'matched_controls.csv', index=False)

    seasons = pd.read_csv(args.ranking / 'season_composite_scores.csv').merge(meta, on='config_id', validate='many_to_one')
    control = seasons[seasons.covariate_set.eq('none')][['recipe', 'input_mode', 'geography', 'season', 'combined']]
    seasons = seasons.merge(control.rename(columns={'combined': 'control'}),
                            on=['recipe', 'input_mode', 'geography', 'season'], validate='many_to_one')
    seasons['change_percent'] = 100 * (seasons.combined / seasons.control - 1)
    seasons = seasons[seasons.covariate_set.eq('kinsa')]
    seasons.to_csv(args.output / 'season_matched_controls.csv', index=False)

    labels = [f'{r.title()}\n{m}' for r, m in ARMS]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2), layout='constrained',
                                      gridspec_kw=dict(width_ratios=[1.25, 1]))
    width = .26
    for k, name in enumerate(SCORES):
        values = table[f'{name} · change %'].to_numpy()
        bars = left.bar(np.arange(len(ARMS)) + (k - 1) * width, values, width, label=name)
        left.bar_label(bars, fmt='%+.1f', fontsize=8, padding=2)
    left.axhline(0, color='black', lw=.8)
    left.set_xticks(range(len(ARMS)), labels)
    left.set_ylabel('Change in relative WIS vs matched control (%)')
    left.set_title('Kinsa vs no covariate · negative is better', loc='left')
    left.legend(title='Locations', fontsize=9)

    grid = seasons[seasons.geography.eq('all')].pivot_table(index=['recipe', 'input_mode'], columns='season',
                                                             values='change_percent').loc[ARMS]
    limit = max(float(np.nanmax(np.abs(grid.to_numpy()))), .01)
    view = right.imshow(grid.to_numpy(), cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
    right.set_xticks(range(grid.shape[1]), list(grid.columns))
    right.set_yticks(range(len(ARMS)), labels)
    for (i, j), value in np.ndenumerate(grid.to_numpy()):
        right.text(j, i, f'{value:+.1f}%', ha='center', va='center', fontsize=9)
    right.set_title('Per-season change, all locations', loc='left')
    fig.colorbar(view, ax=right, shrink=.8)
    fig.suptitle('B2-kinsa · seed 42 · Kinsa (US only) added to the matched no-covariate run\n'
                 'Wednesday arms have no shared Kinsa support: their change is run-to-run variability', fontsize=13)
    fig.savefig(args.output / 'kinsa-effect.png', dpi=180)
    fig.savefig(args.output / 'kinsa-effect.pdf')
    plt.close(fig)
    (args.output / 'manifest.json').write_text(json.dumps(dict(
        ranking=str(args.ranking), configurations=8, seeds=[42], plots=['kinsa-effect']), indent=2) + '\n')
    print(table[['recipe', 'input_mode', 'all · none', 'all · kinsa', 'all · change %',
                 'states/DC · change %', 'US · change %']].round(4).to_string(index=False))


if __name__ == '__main__':
    main()
