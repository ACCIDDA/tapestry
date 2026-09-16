"""Figures for the B0.1 architecture crosses.

Reads the `manager rank` artifacts plus each run's saved `forecasts.npz` and the
frozen hub comparison, and writes the PNGs the results page embeds. Ranking and
quality figures come from `configuration_ranking.csv`/`season_scores.csv`; fan
plots are rendered from `forecasts.npz` via `tapestry.evaluation.hubs.export_b0`
against the frozen truth, so no refit and no EpiBench run is needed.

    .venv/bin/python scripts/plot_b01_crosses.py \
        -e B0.1 -r ranking-509b07b0d243
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
FAMILY = {'local_mlp': '#4f81bd', 'spatial_conv': '#9bbb59', 'pathogen_multiscale': '#8064a2',
          'target_multiscale': '#f79646', 'joint_multiscale': '#4bacc6', 'joint_trend': '#c0504d',
          'independent': '#1f6f4a', 'conv_control': '#808080'}
ENSEMBLE = '#7f7f7f'


def family(name):
    """Reference family of a configuration; the four conv controls share one colour."""
    head = name.split('__')[0]
    return 'conv_control' if head.startswith('conv_control') else head


def wrap(name):
    """One cross factor per line, so long configuration names fit a row label."""
    return '\n'.join(name.split('__'))


def load(experiment, ranking):
    jobs = pd.read_csv(experiment / 'jobs.csv')
    names = {scenario: name.replace('B0.1/', '') for scenario, name in zip(jobs.scenario, jobs.name)}
    configs = pd.read_csv(ranking / 'configuration_ranking.csv')
    configs['name'] = configs.config_id.map(names)
    runs = pd.read_csv(ranking / 'run_scores.csv')
    runs['name'] = runs.config_id.map(names)
    seasons = pd.read_csv(ranking / 'season_scores.csv')
    seasons['name'] = seasons.config_id.map(names)
    return names, configs, runs, seasons


def quality_table(seasons, configs, geography='states_dc'):
    """Per-configuration decomposition of WIS into the quantities the pairplot compares."""
    part = seasons[seasons.geography.eq(geography)]
    table = part.groupby('name').agg(
        coverage_50=('model_coverage_50', 'mean'), coverage_95=('model_coverage_95', 'mean'),
        dispersion=('model_dispersion', 'sum'), wis=('model_wis', 'sum'),
        under=('model_underprediction', 'sum'), over=('model_overprediction', 'sum'))
    # WIS splits exactly into dispersion + under + over, so these are shares of one total.
    table['sharpness'] = table.dispersion / table.wis
    table['bias'] = (table.over - table.under) / table.wis
    table[['coverage_50', 'coverage_95']] *= 100
    return table.join(configs.set_index('name')[['combined_mean', 'combined_sd', 'rank']])


CATASTROPHIC = 1.2


def ranking_figure(configs, runs, output):
    """Every configuration's combined score with its three seeds, best on top.

    The axis stops at `CATASTROPHIC` so the interesting range around parity is
    readable; worse configurations are clipped to the edge and labelled with
    their true score rather than compressing everything else.
    """
    order = configs.sort_values('combined_mean')
    seeds = runs[runs.geography.eq('all')]
    fig, ax = plt.subplots(figsize=(10.5, 0.19 * len(order) + 1.8))
    y = np.arange(len(order))[::-1]
    for offset, row in zip(y, order.itertuples()):
        color = FAMILY[family(row.name)]
        if row.combined_mean > CATASTROPHIC:
            ax.plot(CATASTROPHIC, offset, '>', ms=6, color=color, clip_on=False, zorder=3)
            # Beyond ~2 the model is not merely worse, it has collapsed; say so.
            label = (f'catastrophic · {row.combined_mean:.2f}' if row.combined_mean > 2
                     else f'{row.combined_mean:.2f}')
            ax.annotate(label, (CATASTROPHIC, offset),
                        textcoords='offset points', xytext=(9, 0), va='center', fontsize=5,
                        color=color, annotation_clip=False)
            continue
        points = seeds[seeds.name.eq(row.name)].combined
        ax.plot(np.clip(points, None, CATASTROPHIC), [offset] * len(points), 'o', ms=3, mfc='none',
                mec=color, alpha=.7, zorder=2)
        ax.plot(row.combined_mean, offset, 'o', ms=5.5, color=color, zorder=3)
    ax.axvline(1, color=ENSEMBLE, lw=1.4, ls='--', zorder=1)
    ax.set_xlim(order.combined_mean.min() - .03, CATASTROPHIC)
    ax.set_yticks(y)
    ax.set_yticklabels(order.name, fontsize=5.2)
    ax.set_xlabel('Combined score (total model WIS / total ensemble WIS; lower is better, 1 = hub ensemble)')
    beyond = int((configs.combined_mean > CATASTROPHIC).sum())
    collapsed = int((configs.combined_mean > 2).sum())
    ax.set_title(f'B0.1 crosses: combined score, all {len(order)} configurations\n'
                 f'Filled dot = mean over three seeds, open dots = individual seeds · '
                 f'{beyond} configurations worse than {CATASTROPHIC:g} are clipped to the right edge '
                 f'and labelled ({collapsed} of them collapsed beyond 2)',
                 fontsize=10)
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=f) for f, c in FAMILY.items()]
    handles.append(plt.Line2D([], [], color=ENSEMBLE, ls='--', label='Hub ensemble parity'))
    ax.legend(handles=handles, fontsize=7.5, loc='lower right')
    ax.margins(y=.003)
    fig.tight_layout()
    fig.savefig(output / 'ranking-combined.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def family_figure(configs, output):
    """Where each reference family lands, and how wide its one-factor crosses spread."""
    order = configs.groupby(configs.name.map(family)).combined_mean.median().sort_values()
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for offset, name in enumerate(order.index):
        values = configs[configs.name.map(family).eq(name)].combined_mean
        jitter = np.random.default_rng(0).normal(0, .06, len(values))
        ax.plot(values, offset + jitter, 'o', ms=4.5, color=FAMILY[name], alpha=.75, mec='white', lw=0)
        ax.plot(values.median(), offset, '|', ms=20, color='black', zorder=4)
    ax.axvline(1, color=ENSEMBLE, lw=1.4, ls='--', label='Hub ensemble parity')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f'{n}  (n={(configs.name.map(family) == n).sum()})' for n in order.index], fontsize=8.5)
    ax.set_xscale('log')
    ax.set_xlabel('Combined score, log scale (lower is better)')
    ax.set_title('Reference family decides the outcome; the black bar is the family median\n'
                 'Each dot is one configuration, averaged over three seeds', fontsize=10)
    ax.legend(fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(output / 'family-spread.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def quality_pairplot(quality, output):
    """Pairwise scatter of the score against its calibration and decomposition drivers."""
    columns = [('combined_mean', 'Combined score'), ('coverage_50', '50% coverage (%)'),
               ('coverage_95', '95% coverage (%)'), ('sharpness', 'Dispersion share of WIS'),
               ('bias', 'Bias (over − under) / WIS')]
    nominal = {'coverage_50': 50, 'coverage_95': 95, 'bias': 0}
    colors = [FAMILY[family(n)] for n in quality.index]
    n = len(columns)
    fig, axes = plt.subplots(n, n, figsize=(13.5, 13.5))
    for row, (yk, ylabel) in enumerate(columns):
        for column, (xk, xlabel) in enumerate(columns):
            ax = axes[row, column]
            if row == column:
                values = quality[xk]
                # Log-spaced bins so the histogram lines up with the log axis below it.
                bins = (np.geomspace(values.min(), values.max(), 29) if xk == 'combined_mean'
                        else np.linspace(values.min(), values.max(), 29))
                ax.hist(values, bins=bins, color='#9099a2', edgecolor='white', lw=.4)
                if xk in nominal:
                    ax.axvline(nominal[xk], color='red', lw=1.1, ls='--')
                if xk == 'combined_mean':
                    ax.axvline(1, color=ENSEMBLE, lw=1.1, ls='--')
            else:
                ax.scatter(quality[xk], quality[yk], s=14, c=colors, alpha=.8, lw=.3, edgecolor='white')
                if xk in nominal:
                    ax.axvline(nominal[xk], color='red', lw=.9, ls='--')
                if yk in nominal:
                    ax.axhline(nominal[yk], color='red', lw=.9, ls='--')
                if xk == 'combined_mean':
                    ax.axvline(1, color=ENSEMBLE, lw=.9, ls='--')
                if yk == 'combined_mean':
                    ax.axhline(1, color=ENSEMBLE, lw=.9, ls='--')
                # Spearman: the relationships are monotone but far from linear.
                rho = quality[[xk, yk]].corr(method='spearman').iloc[0, 1]
                ax.annotate(f'ρ={rho:+.2f}', (.04, .93), xycoords='axes fraction', fontsize=7.5,
                            bbox=dict(boxstyle='round,pad=.2', fc='white', ec='none', alpha=.75))
            # The score spans 0.88 to 4.9, so a log axis keeps the leaders separable.
            if xk == 'combined_mean':
                ax.set_xscale('log')
                ax.set_xticks([1, 2, 3, 5])
                ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
                ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            if yk == 'combined_mean' and row != column:
                ax.set_yscale('log')
                ax.set_yticks([1, 2, 3, 5])
                ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
                ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            if row == n - 1:
                ax.set_xlabel(xlabel, fontsize=8.5)
            else:
                ax.set_xticklabels([])
            # A diagonal histogram shares the column's x axis but has its own count axis.
            if column == 0 and row != column:
                ax.set_ylabel(ylabel, fontsize=8.5)
            elif row == column:
                ax.set_ylabel('count' if column == 0 else '', fontsize=7.5)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)
    handles = [plt.Line2D([], [], marker='o', ls='', color=c, label=f) for f, c in FAMILY.items()]
    handles.append(plt.Line2D([], [], color='red', ls='--', label='Nominal / unbiased'))
    handles.append(plt.Line2D([], [], color=ENSEMBLE, ls='--', label='Hub ensemble parity'))
    fig.legend(handles=handles, fontsize=8.5, loc='upper center', ncol=5, frameon=False,
               bbox_to_anchor=(.5, 1.005))
    fig.suptitle('B0.1 forecast quality: score against calibration, sharpness and bias (states/DC tasks)\n'
                 'Diagonal panels are marginal distributions over the 172 configurations; ρ is Spearman',
                 fontsize=11, y=1.045)
    fig.tight_layout()
    fig.savefig(output / 'quality-pairplot.png', dpi=140, bbox_inches='tight')
    plt.close(fig)


def coverage_figure(seasons, output):
    """Calibration: nominal versus achieved, leaders against the full field and the ensemble."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)
    leaders = seasons[seasons.name.isin(LEADERS)]
    for ax, geo in zip(axes, ['states_dc', 'US']):
        part = seasons[seasons.geography.eq(geo)]
        lead = leaders[leaders.geography.eq(geo)]
        table = part.groupby('target')[['model_coverage_50', 'ensemble_coverage_50',
                                        'model_coverage_95', 'ensemble_coverage_95']].mean() * 100
        table = table.reindex(NAMES)
        best = lead.groupby('target')[['model_coverage_50', 'model_coverage_95']].mean().reindex(NAMES) * 100
        y = np.arange(len(table))
        ax.barh(y + .18, table.model_coverage_50, height=.34, color='#4f81bd', label='All configs 50%')
        ax.barh(y - .18, table.model_coverage_95, height=.34, color='#9bbb59', label='All configs 95%')
        ax.plot(best.model_coverage_50, y + .18, 'o', ms=5, color='#1f3864', label='Top-5 50%')
        ax.plot(best.model_coverage_95, y - .18, 'o', ms=5, color='#375623', label='Top-5 95%')
        ax.plot(table.ensemble_coverage_50, y + .18, 'k|', ms=12, label='Ensemble 50%')
        ax.plot(table.ensemble_coverage_95, y - .18, 'k+', ms=9, label='Ensemble 95%')
        ax.axvline(50, color='red', lw=1, ls='--')
        ax.axvline(95, color='red', lw=1, ls='--')
        ax.set_yticks(y)
        ax.set_yticklabels([NAMES[t] for t in table.index], fontsize=8.5)
        ax.set_xlabel('Interval coverage (%)')
        ax.set_title('States/DC' if geo == 'states_dc' else 'US', fontsize=10)
    # Below the panels: the bars run to the right edge, so an inset legend would cover data.
    axes[0].legend(fontsize=7.5, loc='upper center', bbox_to_anchor=(1.03, -.16), ncol=6, frameon=False)
    fig.suptitle('Coverage averaged over all 516 runs; red lines are the nominal 50% and 95% levels', fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(output / 'coverage.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def states_vs_us(configs, output):
    """States and US agreement, and which families generalise across geography."""
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    for f, color in FAMILY.items():
        part = configs[configs.name.map(family).eq(f)]
        ax.scatter(part.states_dc_combined_mean, part.US_combined_mean, s=34, c=color,
                   edgecolor='white', lw=.5, label=f, zorder=3)
    lo, hi = .8, 5.2
    ax.plot([lo, hi], [lo, hi], color='black', lw=.8, ls=':', zorder=1, label='States = US')
    ax.axhline(1, color=ENSEMBLE, lw=1, ls='--', zorder=1)
    ax.axvline(1, color=ENSEMBLE, lw=1, ls='--', zorder=1)
    for row in configs.nsmallest(4, 'combined_mean').itertuples():
        ax.annotate(row.name, (row.states_dc_combined_mean, row.US_combined_mean),
                    textcoords='offset points', xytext=(7, 4), fontsize=6.5)
    ax.set_xscale('log'), ax.set_yscale('log')
    ax.set_xlim(lo, hi), ax.set_ylim(lo, hi)
    ax.set_xlabel('Combined score, states/DC tasks')
    ax.set_ylabel('Combined score, US tasks')
    ax.set_title('States/DC versus US, log scale\n'
                 'Dashed grey lines are hub-ensemble parity; dotted line is equal skill', fontsize=10)
    ax.legend(fontsize=7.5, loc='upper left')
    fig.tight_layout()
    fig.savefig(output / 'states-vs-us.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def seed_figure(configs, runs, output):
    """Seed noise against the spread the ranking is trying to resolve."""
    seeds = runs[runs.geography.eq('all')]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    ax.scatter(configs.combined_mean, configs.combined_sd, s=28,
               c=[FAMILY[family(n)] for n in configs.name], edgecolor='white', lw=.4)
    median_sd = configs.combined_sd.median()
    ax.axhline(median_sd, color='black', lw=1, ls=':', label=f'Median seed SD = {median_sd:.4f}')
    top10 = configs.nsmallest(10, 'combined_mean')
    ax.axvspan(top10.combined_mean.min(), top10.combined_mean.max(), color='gold', alpha=.25,
               label=f'Top-10 range = {top10.combined_mean.max() - top10.combined_mean.min():.4f}')
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 3, 5])
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%g'))
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('Combined score (mean over seeds)')
    ax.set_ylabel('SD across the three seeds')
    ax.set_title('Seed noise grows with the score, and swamps the top of the table', fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    wide = seeds[seeds.name.isin(configs.nsmallest(30, 'combined_mean').name)].pivot_table(
        index='name', columns='seed', values='combined')
    ranks = wide.rank()
    for name in wide.index:
        ax.plot(range(len(wide.columns)), ranks.loc[name], color='#b0b0b0', lw=.5, alpha=.6, zorder=1)
    for row in configs.nsmallest(5, 'combined_mean').itertuples():
        ax.plot(range(len(wide.columns)), ranks.loc[row.name], marker='o', lw=1.7,
                color=FAMILY[family(row.name)], label=row.name, zorder=3)
    ax.set_xticks(range(len(wide.columns)))
    ax.set_xticklabels([f'seed {c}' for c in wide.columns])
    ax.invert_yaxis()
    ax.set_ylabel('Rank within the top 30, on one seed alone (1 = best)')
    ax.set_title('Where each of the top 30 would rank on a single seed', fontsize=10)
    ax.legend(fontsize=6.5, loc='lower right')
    fig.tight_layout()
    fig.savefig(output / 'seed-instability.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def horizon_figure(totals, output):
    """Where the skill sits across the four forecast horizons."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, geo in zip(axes, ['states_dc', 'US']):
        for name in LEADERS:
            part = totals[totals.name.eq(name) & totals.geography.eq(geo)]
            if part.empty:
                continue
            ratio = part.groupby('horizon').apply(
                lambda g: g.model_wis.sum() / g.ensemble_wis.sum(), include_groups=False)
            ax.plot(ratio.index, ratio.values, marker='o', color=FAMILY[family(name)], label=name)
        ax.axhline(1, color=ENSEMBLE, lw=1.2, ls='--')
        ax.set_xticks(range(4))
        ax.set_xlabel('Horizon (weeks ahead, 0 = first future week)')
        ax.set_title('States/DC' if geo == 'states_dc' else 'US', fontsize=10)
    axes[0].set_ylabel('Total WIS ratio to ensemble')
    axes[0].legend(fontsize=6.5)
    fig.suptitle('Skill by horizon for the five leading configurations', fontsize=10)
    fig.tight_layout()
    fig.savefig(output / 'horizon.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def fan_figures(ranking, configs, runs, frozen, output):
    """US and North Carolina fans for the leaders against the hub ensemble."""
    manifest = json.loads((ranking / 'manifest.json').read_text())
    jobs = {(r['config_id'], r['seed']): r['path'] for r in manifest['runs']}
    ids = dict(zip(configs.name, configs.config_id))
    seeds = runs[runs.geography.eq('all')]
    chosen = []
    for name in LEADERS[:3]:
        part = seeds[seeds.name.eq(name)].sort_values(['combined', 'seed'])
        # The median seed is the honest representative of a configuration.
        chosen.append((name, int(part.iloc[len(part) // 2].seed)))
    exports = {(name, seed): export_b0(jobs[(ids[name], seed)]) for name, seed in chosen}
    for case in frozen_cases(frozen):
        target, season = case['target'], case['season']
        folder = frozen / case['directory']
        units = pd.read_parquet(folder / 'units.parquet')
        quantiles = pd.read_parquet(folder / 'quantiles.parquet')
        ensemble = quantiles[quantiles.model.eq(case['ensemble'])]
        rows = [(f'rank {rank}\n{wrap(name)}\nseed {seed}', exports[(name, seed)][(season, target)],
                 FAMILY[family(name)]) for rank, (name, seed) in enumerate(chosen, start=1)]
        rows.append((f"{case['ensemble']}\n(hub)", ensemble, ENSEMBLE))
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
                    fan = (data[data.reference_date.eq(reference)].sort_values('horizon')
                           .set_index('horizon').reindex(range(4)))
                    x = pd.date_range(reference, periods=4, freq='7D')
                    ax.fill_between(x, fan['q0.025'], fan['q0.975'], color=color, alpha=.18,
                                    label='95%' if i == 0 and row == 0 else None)
                    ax.fill_between(x, fan['q0.25'], fan['q0.75'], color=color, alpha=.42,
                                    label='50%' if i == 0 and row == 0 else None)
                    ax.plot(x, fan['q0.5'], color=color, lw=1)
                ax.set_ylim(bottom=0)
                if column == 0:
                    # Horizontal and outside the axes: these names are too long to rotate.
                    ax.set_ylabel(label, fontsize=6.5, rotation=0, ha='right', va='center', labelpad=8)
                if row == 0:
                    ax.set_title('United States' if location == 'US' else 'North Carolina', fontsize=10)
        axes[0, 0].legend(fontsize=7.5, ncol=3, loc='upper right')
        fig.suptitle(f'{NAMES[target]} · {season} · four-week fans at every third origin\n'
                     'Rows are the three leading configurations at their median seed, then the hub ensemble',
                     fontsize=11)
        fig.autofmt_xdate()
        fig.tight_layout(rect=(.1, 0, 1, 1))
        fig.savefig(output / f'fans-{SHORT[target]}-{season}.png', dpi=130, bbox_inches='tight')
        plt.close(fig)


LEADERS = []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-e', '--experiment', default='B0.1')
    parser.add_argument('-r', '--ranking', required=True, help='ranking-<hash> folder name')
    parser.add_argument('--root', type=Path, default=Path('data/experiments'))
    parser.add_argument('--frozen', type=Path, default=Path('data/evaluation/b0_hub_comparison_q23'))
    parser.add_argument('--output', type=Path, default=Path('docs/results/b0-1-crosses/figures'))
    parser.add_argument('--skip-fans', action='store_true')
    args = parser.parse_args()
    experiment = args.root / args.experiment
    ranking = experiment / args.ranking
    args.output.mkdir(parents=True, exist_ok=True)
    names, configs, runs, seasons = load(experiment, ranking)
    global LEADERS
    LEADERS = list(configs.nsmallest(5, 'combined_mean').name)
    manifest = json.loads((ranking / 'manifest.json').read_text())
    totals = pd.concat([pd.read_csv(Path(r['path']) / 'totals.csv').assign(
        name=names[r['config_id']], seed=r['seed']) for r in manifest['runs']], ignore_index=True)
    quality = quality_table(seasons, configs)
    quality.to_csv(args.output.parent / 'quality-metrics.csv')
    ranking_figure(configs, runs, args.output)
    family_figure(configs, args.output)
    quality_pairplot(quality, args.output)
    coverage_figure(seasons, args.output)
    states_vs_us(configs, args.output)
    seed_figure(configs, runs, args.output)
    horizon_figure(totals, args.output)
    if not args.skip_fans:
        fan_figures(ranking, configs, runs, args.frozen, args.output)
    print(f'Wrote figures to {args.output}')


if __name__ == '__main__':
    main()
