"""Figures for the b0-us-cross-4 crosses calibration.

Reads the `manager rank` artifacts plus each run's saved `forecasts.npz` and the
frozen hub comparison, and writes the PNGs the results page embeds. Ranking
figures come from `season_scores.csv`/`configuration_ranking.csv`; fan plots are
rendered from `forecasts.npz` via `tapestry.evaluation.hubs.export_b0` against
the frozen truth, so no refit and no EpiBench run is needed.

    .venv/bin/python scripts/plot_b0_crosses.py \
        -e b0-us-cross-4 -r ranking-2739af8db682
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tapestry.evaluation.hubs import export_b0
from tapestry.evaluation.totals import frozen_cases

NAMES = {'wk inc flu hosp': 'Influenza admissions', 'wk inc flu prop ed visits': 'Influenza ED visits',
         'wk inc covid hosp': 'COVID-19 admissions', 'wk inc covid prop ed visits': 'COVID-19 ED visits',
         'wk inc rsv hosp': 'RSV admissions', 'wk inc rsv prop ed visits': 'RSV ED visits'}
SHORT = {'wk inc flu hosp': 'flu_hosp', 'wk inc flu prop ed visits': 'flu_prop_ed_visits',
         'wk inc covid hosp': 'covid_hosp', 'wk inc covid prop ed visits': 'covid_prop_ed_visits',
         'wk inc rsv hosp': 'rsv_hosp', 'wk inc rsv prop ed visits': 'rsv_prop_ed_visits'}
# One colour per reference family, so a dot's recipe is readable at a glance.
FAMILY = {'raw': '#c0504d', 'anchor': '#4f81bd', 'conv': '#9bbb59'}
ENSEMBLE = '#7f7f7f'


def family(name):
    return name.split('__')[0]


def load(experiment, ranking):
    jobs = pd.read_csv(experiment / 'jobs.csv')
    names = dict(zip(jobs.scenario, jobs.name))
    configs = pd.read_csv(ranking / 'configuration_ranking.csv')
    configs['name'] = configs.config_id.map(names)
    runs = pd.read_csv(ranking / 'run_scores.csv')
    runs['name'] = runs.config_id.map(names)
    seasons = pd.read_csv(ranking / 'season_scores.csv')
    seasons['name'] = seasons.config_id.map(names)
    return names, configs, runs, seasons


def ranking_figure(configs, runs, output):
    """Every configuration's combined score with its three seeds, best on top."""
    order = configs.sort_values('combined_mean')
    seeds = runs[runs.geography.eq('all')]
    fig, ax = plt.subplots(figsize=(9.5, 0.23 * len(order) + 1.8))
    y = np.arange(len(order))[::-1]
    for offset, row in zip(y, order.itertuples()):
        points = seeds[seeds.name.eq(row.name)].combined
        ax.plot(points, [offset] * len(points), 'o', ms=4, mfc='none',
                mec=FAMILY[family(row.name)], alpha=.75, zorder=2)
        ax.plot(row.combined_mean, offset, 'o', ms=7, color=FAMILY[family(row.name)], zorder=3)
    ax.axvline(1, color=ENSEMBLE, lw=1.4, ls='--', zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(order.name, fontsize=7)
    ax.set_xlabel('Combined score (total model WIS / total ensemble WIS; lower is better, 1 = hub ensemble)')
    ax.set_title('Crosses calibration: combined score, all 60 configurations\n'
                 'Filled dot = mean over three seeds, open dots = individual seeds', fontsize=10)
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=f'`{f}` family') for f, c in FAMILY.items()]
    handles.append(plt.Line2D([], [], color=ENSEMBLE, ls='--', label='Hub ensemble parity'))
    ax.legend(handles=handles, fontsize=8, loc='lower right')
    ax.margins(y=.004)
    fig.tight_layout()
    fig.savefig(output / 'ranking-combined.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def states_vs_us(configs, output):
    """The headline of the scaling fix: states and US scores now agree far better."""
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    for f, color in FAMILY.items():
        part = configs[configs.name.map(family).eq(f)]
        ax.scatter(part.states_dc_combined_mean, part.US_combined_mean, s=38, c=color,
                   edgecolor='white', lw=.6, label=f'`{f}` family', zorder=3)
    lo = min(configs.states_dc_combined_mean.min(), configs.US_combined_mean.min()) - .02
    hi = max(configs.states_dc_combined_mean.max(), configs.US_combined_mean.max()) + .02
    ax.plot([lo, hi], [lo, hi], color='black', lw=.8, ls=':', zorder=1, label='States = US')
    ax.axhline(1, color=ENSEMBLE, lw=1, ls='--', zorder=1)
    ax.axvline(1, color=ENSEMBLE, lw=1, ls='--', zorder=1)
    for row in configs.nsmallest(4, 'combined_mean').itertuples():
        ax.annotate(row.name, (row.states_dc_combined_mean, row.US_combined_mean),
                    textcoords='offset points', xytext=(7, 4), fontsize=7.5)
    ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
    ax.set_xlabel('Combined score, states/DC tasks')
    ax.set_ylabel('Combined score, US tasks')
    ax.set_title('States/DC versus US after the per-location scaling fix\n'
                 'Dashed grey lines are hub-ensemble parity; dotted line is equal skill', fontsize=10)
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(output / 'states-vs-us.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def epoch_ladder(configs, output):
    """Training length is the dominant factor, and it reverses between families."""
    ladder = [('ep50\n(reference)', ''), ('ep100', '__stopping_100_0'),
              ('ep300', '__stopping_300_0'), ('ep300\npatience 20', '__stopping_300_20')]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    x = np.arange(len(ladder))
    for f, color in FAMILY.items():
        means, sds = [], []
        for _, suffix in ladder:
            row = configs[configs.name.eq(f + suffix)]
            means.append(row.combined_mean.iloc[0] if len(row) else np.nan)
            sds.append(row.combined_sd.iloc[0] if len(row) else np.nan)
        ax.errorbar(x, means, yerr=sds, marker='o', color=color, capsize=3, lw=1.6, label=f'`{f}` family')
    ax.axhline(1, color=ENSEMBLE, lw=1.2, ls='--', label='Hub ensemble parity')
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in ladder], fontsize=9)
    ax.set_ylabel('Combined score (lower is better)')
    ax.set_title('Training-length ladder by reference family\n'
                 'Error bars are the SD across three seeds', fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'epoch-ladder.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def seed_figure(configs, runs, output):
    """Seed noise against the spread the ranking is trying to resolve."""
    seeds = runs[runs.geography.eq('all')]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    ax.scatter(configs.combined_mean, configs.combined_sd, s=32,
               c=[FAMILY[family(n)] for n in configs.name], edgecolor='white', lw=.5)
    median_sd = configs.combined_sd.median()
    ax.axhline(median_sd, color='black', lw=1, ls=':', label=f'Median seed SD = {median_sd:.4f}')
    top10 = configs.nsmallest(10, 'combined_mean')
    ax.axvspan(top10.combined_mean.min(), top10.combined_mean.max(), color='gold', alpha=.25,
               label=f'Top-10 range = {top10.combined_mean.max() - top10.combined_mean.min():.4f}')
    ax.set_xlabel('Combined score (mean over seeds)')
    ax.set_ylabel('SD across the three seeds')
    ax.set_title('Seed noise is the same size as the leaderboard spread', fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    wide = seeds.pivot_table(index='name', columns='seed', values='combined')
    ranks = wide.rank()
    for name in wide.index:
        ax.plot(range(len(wide.columns)), ranks.loc[name], color='#b0b0b0', lw=.5, alpha=.5, zorder=1)
    # Two leaders share each family colour, so linestyle separates the siblings.
    for order, row in enumerate(configs.nsmallest(4, 'combined_mean').itertuples()):
        ax.plot(range(len(wide.columns)), ranks.loc[row.name], marker='o', lw=1.8,
                color=FAMILY[family(row.name)], ls='-' if order < 2 else '--',
                label=row.name, zorder=3)
    ax.set_xticks(range(len(wide.columns)))
    ax.set_xticklabels([f'seed {c}' for c in wide.columns])
    ax.invert_yaxis()
    ax.set_ylabel('Rank within that single seed (1 = best)')
    ax.set_title('Where each configuration would rank on one seed alone', fontsize=10)
    ax.legend(fontsize=7.5, loc='lower right')
    fig.tight_layout()
    fig.savefig(output / 'seed-instability.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def coverage_figure(seasons, output):
    """Calibration: nominal versus achieved, model against ensemble."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    for ax, geo in zip(axes, ['states_dc', 'US']):
        part = seasons[seasons.geography.eq(geo)]
        table = part.groupby('target')[['model_coverage_50', 'ensemble_coverage_50',
                                        'model_coverage_95', 'ensemble_coverage_95']].mean() * 100
        table = table.reindex(NAMES)
        y = np.arange(len(table))
        ax.barh(y + .18, table.model_coverage_50, height=.34, color='#4f81bd', label='B0 50%')
        ax.barh(y - .18, table.model_coverage_95, height=.34, color='#9bbb59', label='B0 95%')
        ax.plot(table.ensemble_coverage_50, y + .18, 'k|', ms=12, label='Ensemble 50%')
        ax.plot(table.ensemble_coverage_95, y - .18, 'k+', ms=9, label='Ensemble 95%')
        ax.axvline(50, color='red', lw=1, ls='--')
        ax.axvline(95, color='red', lw=1, ls='--')
        ax.set_yticks(y)
        ax.set_yticklabels([NAMES[t] for t in table.index], fontsize=8.5)
        ax.set_xlabel('Interval coverage (%)')
        ax.set_title('States/DC' if geo == 'states_dc' else 'US', fontsize=10)
    axes[0].legend(fontsize=7.5, loc='lower right')
    fig.suptitle('Coverage averaged over all 180 runs; red lines are the nominal 50% and 95% levels', fontsize=10)
    fig.tight_layout()
    fig.savefig(output / 'coverage.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def horizon_figure(totals, output):
    """Where the skill sits across the four forecast horizons."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    leaders = ['anchor__stopping_300_0', 'anchor__stopping_100_0', 'conv__decoder_leg', 'conv', 'anchor', 'raw']
    styles = {'anchor__stopping_300_0': ('#4f81bd', '-'), 'anchor__stopping_100_0': ('#4f81bd', '--'),
              'conv__decoder_leg': ('#9bbb59', '-'), 'conv': ('#9bbb59', '--'),
              'anchor': ('#4f81bd', ':'), 'raw': ('#c0504d', ':')}
    for ax, geo in zip(axes, ['states_dc', 'US']):
        for name in leaders:
            part = totals[totals.name.eq(name) & totals.geography.eq(geo)]
            ratio = part.groupby('horizon').apply(
                lambda g: g.model_wis.sum() / g.ensemble_wis.sum(), include_groups=False)
            color, ls = styles[name]
            ax.plot(ratio.index, ratio.values, marker='o', color=color, ls=ls, label=name)
        ax.axhline(1, color=ENSEMBLE, lw=1.2, ls='--')
        ax.set_xticks(range(4))
        ax.set_xlabel('Horizon (weeks ahead, 0 = first future week)')
        ax.set_title('States/DC' if geo == 'states_dc' else 'US', fontsize=10)
    axes[0].set_ylabel('Total WIS ratio to ensemble')
    axes[0].legend(fontsize=7.5)
    fig.suptitle('Skill by horizon: B0 is now strongest at the nowcast and decays outward', fontsize=10)
    fig.tight_layout()
    fig.savefig(output / 'horizon.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def fan_figures(experiment, ranking, names, configs, runs, frozen, output):
    """US and North Carolina fans for the leaders against the hub ensemble."""
    manifest = json.loads((ranking / 'manifest.json').read_text())
    paths = {(r['name'], r['seed']): r['path'] for r in manifest['runs']}
    seeds = runs[runs.geography.eq('all')]
    leaders = list(configs.nsmallest(3, 'combined_mean').name)
    chosen = []
    for name in leaders:
        part = seeds[seeds.name.eq(name)].sort_values(['combined', 'seed'])
        chosen.append((name, int(part.iloc[len(part) // 2].seed)))
    exports = {label: export_b0(paths[label]) for label in chosen}
    for case in frozen_cases(frozen):
        target, season = case['target'], case['season']
        folder = frozen / case['directory']
        units = pd.read_parquet(folder / 'units.parquet')
        quantiles = pd.read_parquet(folder / 'quantiles.parquet')
        ensemble = quantiles[quantiles.model.eq(case['ensemble'])]
        rows = [(f'{name} · seed {seed}', exports[(name, seed)][(season, target)], FAMILY[family(name)])
                for name, seed in chosen]
        rows.append((f"{case['ensemble']} (hub)", ensemble, ENSEMBLE))
        fig, axes = plt.subplots(len(rows), 2, figsize=(14, 2.3 * len(rows)), sharex='col', squeeze=False)
        for column, location in enumerate(['US', '37']):
            truth = (units[units.location.eq(location)][['target_end_date', 'observed']]
                     .drop_duplicates().sort_values('target_end_date'))
            for row, (label, frame, color) in enumerate(rows):
                ax = axes[row, column]
                data = frame[frame.location.eq(location)]
                ax.plot(pd.to_datetime(truth.target_end_date), truth.observed, color='black', lw=1,
                        label='Frozen truth' if row == 0 else None)
                # Every third origin keeps overlapping fans legible.
                for i, reference in enumerate(sorted(data.reference_date.unique())[::3]):
                    fan = data[data.reference_date.eq(reference)].sort_values('horizon').set_index('horizon').reindex(range(4))
                    x = pd.date_range(reference, periods=4, freq='7D')
                    ax.fill_between(x, fan['q0.025'], fan['q0.975'], color=color, alpha=.18,
                                    label='95%' if i == 0 and row == 0 else None)
                    ax.fill_between(x, fan['q0.25'], fan['q0.75'], color=color, alpha=.42,
                                    label='50%' if i == 0 and row == 0 else None)
                    ax.plot(x, fan['q0.5'], color=color, lw=1)
                ax.set_ylim(bottom=0)
                if column == 0:
                    ax.set_ylabel(label, fontsize=8)
                if row == 0:
                    ax.set_title('United States' if location == 'US' else 'North Carolina', fontsize=10)
        axes[0, 0].legend(fontsize=7.5, ncol=3, loc='upper right')
        fig.suptitle(f'{NAMES[target]} · {season} · four-week fans at every third origin', fontsize=11)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output / f'fans-{SHORT[target]}-{season}.png', dpi=130, bbox_inches='tight')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-e', '--experiment', default='b0-us-cross-4')
    parser.add_argument('-r', '--ranking', required=True, help='ranking-<hash> folder name')
    parser.add_argument('--root', type=Path, default=Path('data/experiments'))
    parser.add_argument('--frozen', type=Path, default=Path('data/evaluation/b0_hub_comparison_q23'))
    parser.add_argument('--output', type=Path, default=Path('docs/results/b0-crosses/figures'))
    parser.add_argument('--skip-fans', action='store_true')
    args = parser.parse_args()
    experiment = args.root / args.experiment
    ranking = experiment / args.ranking
    args.output.mkdir(parents=True, exist_ok=True)
    names, configs, runs, seasons = load(experiment, ranking)
    manifest = json.loads((ranking / 'manifest.json').read_text())
    totals = pd.concat([pd.read_csv(Path(r['path']) / 'totals.csv').assign(name=r['name'], seed=r['seed'])
                        for r in manifest['runs']], ignore_index=True)
    ranking_figure(configs, runs, args.output)
    states_vs_us(configs, args.output)
    epoch_ladder(configs, args.output)
    seed_figure(configs, runs, args.output)
    coverage_figure(seasons, args.output)
    horizon_figure(totals, args.output)
    if not args.skip_fans:
        fan_figures(experiment, ranking, names, configs, runs, args.frozen, args.output)
    print(f'Wrote figures to {args.output}')


if __name__ == '__main__':
    main()
