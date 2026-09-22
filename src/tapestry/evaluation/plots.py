"""The four experiment figures (user request 2026-09-22); the only plotting code in the project.

Written by `planner rank` into `<ranking>/plots/` and by `planner plots -e NAME
[--configs A B] [--dates D ...]`. Choices (also in docs/design/restructure-2026-unified.md §5):

1. `cv-layout-*.png`: per fold (row = held-out season) and target (column), the
   finalized US series from panel.npz, the background showing each week's role from
   `dataset.cv.week_roles` (the function `cv.fold` uses): fit, early-stopping
   validation (only with patience > 0; the refit then trains on those weeks), scored
   held-out season, unused. One figure per distinct CV setting among the experiment's
   scenarios. Score episodes may take context from earlier weeks; that is not drawn.
2. `fans-US.png`, `fans-NC.png`: rows = targets, columns = held-out seasons; finalized
   truth (panel.npz) and 50%/90% intervals + median for up to two configurations (the
   two best ranked by default, `--configs` overrides; each at its lowest seed; seeds
   are not pooled) and the hub ensemble (frozen support, where it exists). Default
   reference dates: the 25th, 50th and 75th percentile positions of each season's
   score reference dates; `--dates` overrides.
3. `dotplot.png` (seaborn PairGrid dot plot, user request 2026-09-22): one row per
   configuration ranked by mean combined WIS ratio, one column per metric (combined WIS
   ratio for all / states-DC / US, 50/80/90/95% coverage, dispersion, under- and
   over-prediction over ensemble WIS, weighted exactly like the score so the three sum
   to the WIS ratio), one dot per seed and a bar at the seed mean. Grey dashed lines:
   the ensemble's value; red: nominal coverage.
4. `heatmap-*.png`: per selected configuration, WIS ratio (model / ensemble total WIS,
   all horizons) by location x season, targets combined with the ranking's target
   weights over the targets available there, mean over seeds; last column = mean of
   seasons. Colour on a log scale centred at 1.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tapestry.data.geography import STATE_FIPS
from tapestry.dataset.build import load as load_panel
from tapestry.dataset.cv import SEASONS, season, week_roles
from tapestry.model.scenario import Scenario
from .hubs import CHANNEL, export
from .totals import COMPONENTS, COVERAGE, frozen_cases, run_scores

ROLE_COLORS = {'fit': '#cfe0f3', 'validation': '#f6c77b', 'score': '#f3b6b6', 'unused': '#eeeeee'}
ROLE_LABELS = {'fit': 'training (fit)', 'validation': 'early-stopping validation (inner fit only; refit trains on it)',
               'score': 'held-out season (scored)', 'unused': 'unused (outside CV seasons)'}
MODEL_COLORS = ('#0072b2', '#d55e00')
ENSEMBLE_COLOR = '#555555'
FAN_LOCATIONS = ('US', 'NC')


def label(config_id):
    return config_id or 'default'


def _save(fig, path):
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return path


def cv_layout(panel, scenarios, output):
    """Figure 1: one file per distinct CV setting (patience > 0 and validation_* fields)."""
    dates = np.array([str(d) for d in panel['dates']])
    x = pd.to_datetime(dates)
    us = list(panel['locations']).index('US')
    groups = {}
    for scenario in scenarios:
        key = (bool(scenario.patience), scenario.validation_weeks, scenario.validation_spacing, scenario.validation_offset)
        groups.setdefault(key, []).append(scenario)
    paths = []
    for (early, weeks, spacing, offset), members in groups.items():
        fig, axes = plt.subplots(len(SEASONS), len(CHANNEL), figsize=(3.2 * len(CHANNEL), 2.2 * len(SEASONS)),
                                 sharex=True, squeeze=False)
        for row, held_out in enumerate(SEASONS):
            roles = week_roles(dates, members[0], held_out)
            starts = np.flatnonzero(np.r_[True, roles[1:] != roles[:-1]])
            ends = np.r_[starts[1:], len(roles)] - 1
            for column, target in enumerate(CHANNEL):
                ax = axes[row, column]
                for a, b in zip(starts, ends):
                    ax.axvspan(x[a] - pd.Timedelta(days=3.5), x[b] + pd.Timedelta(days=3.5),
                               color=ROLE_COLORS[roles[a]], lw=0)
                ax.plot(x, panel['targets'][:, us, CHANNEL[target]], color='black', lw=1)
                if row == 0:
                    ax.set_title(target, fontsize=9)
                if column == 0:
                    ax.set_ylabel(f'held out\n{held_out}', fontsize=9)
                ax.tick_params(labelsize=7)
        handles = [plt.Rectangle((0, 0), 1, 1, color=ROLE_COLORS[r], label=ROLE_LABELS[r]) for r in ROLE_COLORS
                   if early or r != 'validation']
        fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=8, frameon=False)
        rule = (f'early stopping: {weeks} of every {spacing} weeks from week {offset} of each training season'
                if early else 'no early stopping (patience=0): fixed epochs, no validation weeks')
        names = ', '.join(label(s.scenario_string) for s in members)
        fig.suptitle(f'Cross-validation layout, finalized US truth -- {rule}\nconfigurations: {names}', fontsize=10)
        fig.tight_layout(rect=(0, .04, 1, 1))
        name = f'cv-layout-es{weeks}-{spacing}-{offset}.png' if early else 'cv-layout-fixed-epochs.png'
        paths.append(_save(fig, output / name))
    return paths


def default_dates(frames):
    """25th/50th/75th percentile positions of each season's score reference dates."""
    chosen = []
    for held in SEASONS:
        references = sorted(frames[(held, next(iter(CHANNEL)))].reference_date.unique())
        chosen += [references[int(q * (len(references) - 1))] for q in (.25, .5, .75)]
    return chosen


def fans(panel, runs, configs, frozen, dates, output):
    """Figure 2: fans at `dates` for up to two configurations and the hub ensemble."""
    frames = {c: export(min((r for r in runs if r['config_id'] == c), key=lambda r: r['seed'])['path'])
              for c in configs}
    dates = sorted(dates or default_dates(next(iter(frames.values()))))
    ensemble = {}
    for case in frozen_cases(frozen):
        table = pd.read_parquet(Path(frozen) / case['directory'] / 'quantiles.parquet')
        ensemble[(case['season'], case['target'])] = table[table.model == case['ensemble']]
    truth_dates = pd.to_datetime([str(d) for d in panel['dates']])
    fips = {v: k for k, v in STATE_FIPS.items()} | {'US': 'US'}
    paths = []
    for location in FAN_LOCATIONS:
        l = list(panel['locations']).index(location)
        fig, axes = plt.subplots(len(CHANNEL), len(SEASONS), figsize=(5 * len(SEASONS), 2.3 * len(CHANNEL)), squeeze=False)
        for column, held in enumerate(SEASONS):
            in_season = np.array([season(d) == held for d in truth_dates.strftime('%Y-%m-%d')])
            for row, target in enumerate(CHANNEL):
                ax = axes[row, column]
                ax.plot(truth_dates[in_season], panel['targets'][in_season, l, CHANNEL[target]], color='black', lw=1.2,
                        label='finalized truth')
                sources = [(label(c), MODEL_COLORS[i], frames[c].get((held, target))) for i, c in enumerate(configs)]
                sources.append(('hub ensemble', ENSEMBLE_COLOR, ensemble.get((held, target))))
                for name, color, table in sources:
                    if table is None:
                        continue
                    part = table[table.location.eq(fips[location]) & table.reference_date.isin(dates)]
                    for i, (_, fan) in enumerate(part.sort_values('horizon').groupby('reference_date')):
                        x = pd.to_datetime(fan.target_end_date)
                        ax.fill_between(x, fan['q0.05'], fan['q0.95'], color=color, alpha=.15, lw=0)
                        ax.fill_between(x, fan['q0.25'], fan['q0.75'], color=color, alpha=.3, lw=0)
                        ax.plot(x, fan['q0.5'], color=color, lw=1.2, label=name if i == 0 else None)
                if row == 0:
                    ax.set_title(f'held out {held}', fontsize=10)
                if column == 0:
                    ax.set_ylabel(target, fontsize=8)
                ax.tick_params(labelsize=7)
        handles, labels = [], []
        for ax in axes.flat:
            for h, t in zip(*ax.get_legend_handles_labels()):
                if t not in labels:
                    handles.append(h)
                    labels.append(t)
        fig.legend(handles, labels, loc='lower center', ncol=len(labels), fontsize=8, frameon=False)
        fig.suptitle(f'{location}: 50% and 90% intervals and median at reference dates {", ".join(dates)}', fontsize=10)
        fig.tight_layout(rect=(0, .03, 1, .98))
        paths.append(_save(fig, output / f'fans-{location}.png'))
    return paths


def dotplot(seasons, weights, output):
    """Figure 3 (seaborn PairGrid dot plot): one row per configuration, ranked by the
    main score (mean combined WIS ratio over seeds, best on top); one column per metric;
    one dot per seed, black bar = seed mean. Metrics are weighted like the score:
    combined WIS ratio (all locations, states/DC, US), 50/80/90/95% coverage, and the
    WIS decomposition (dispersion, under-, over-prediction over ensemble WIS; they sum
    to the WIS ratio). Grey dashed = ensemble's value, red dotted = nominal coverage."""
    scored = lambda value, geography='all': run_scores(seasons, weights, value).query(
        'geography == @geography').set_index(['config_id', 'seed']).combined
    columns = {'WIS ratio': ('wis_ratio', 'all'), 'WIS ratio, states/DC': ('wis_ratio', 'states_dc'),
               'WIS ratio, US': ('wis_ratio', 'US'),
               **{f'{c}% coverage': (f'model_coverage_{c}', 'all') for c in COVERAGE},
               **{f'{c} / ens. WIS': (f'model_{c}_ratio', 'all') for c in COMPONENTS}}
    reference = {'WIS ratio': 1., 'WIS ratio, states/DC': 1., 'WIS ratio, US': 1.,
                 **{f'{c}% coverage': float(scored(f'ensemble_coverage_{c}').iloc[0]) for c in COVERAGE},
                 **{f'{c} / ens. WIS': float(scored(f'ensemble_{c}_ratio').iloc[0]) for c in COMPONENTS}}
    nominal = {f'{c}% coverage': c / 100 for c in COVERAGE}
    table = pd.DataFrame({name: scored(*value) for name, value in columns.items()}).reset_index()
    table['configuration'] = table.config_id.fillna('').map(label)
    order = list(table.groupby('configuration')['WIS ratio'].mean().sort_values().index)
    grid = sns.PairGrid(table, x_vars=list(columns), y_vars=['configuration'],
                        height=max(1.8, .45 * len(order) + 1), aspect=.55)
    grid.map(sns.stripplot, order=order, size=6, orient='h', jitter=False, linewidth=.5,
             edgecolor='w', color=MODEL_COLORS[0], alpha=.8)
    for ax, name in zip(grid.axes.flat, columns):
        means = table.groupby('configuration')[name].mean().reindex(order)
        ax.scatter(means.values, range(len(order)), marker='|', s=250, color='k', zorder=3)
        ax.axvline(reference[name], color=ENSEMBLE_COLOR, ls='--', lw=.8)
        if name in nominal:
            ax.axvline(nominal[name], color='red', ls=':', lw=.8)
        ax.set(xlabel='', ylabel='', title=name)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
    sns.despine(left=True, bottom=True)
    grid.figure.suptitle('Configurations ranked by combined WIS ratio (best on top); dot = seed, bar = mean; '
                         'grey dashed = ensemble, red dotted = nominal coverage', y=1.02)
    return [_save(grid.figure, output / 'dotplot.png')]


def heatmaps(runs, configs, weights, output):
    """Figure 4: WIS ratio by location x season for each selected configuration."""
    paths = []
    for config in configs:
        totals = pd.concat([pd.read_csv(Path(r['path']) / 'totals.csv', dtype={'location': str}).assign(seed=r['seed'])
                            for r in runs if r['config_id'] == config])
        cells = totals.groupby(['seed', 'season', 'location', 'target'])[['model_wis', 'ensemble_wis']].sum().reset_index()
        cells['ratio'] = cells.model_wis / cells.ensemble_wis
        cells['weight'] = cells.target.map(weights)
        combined = cells.groupby(['seed', 'season', 'location']).apply(
            lambda g: np.dot(g.weight, g.ratio) / g.weight.sum(), include_groups=False).rename('ratio').reset_index()
        grid = combined.groupby(['location', 'season']).ratio.mean().unstack('season')
        grid['mean of seasons'] = grid.mean(axis=1)
        grid = grid.loc[['US'] + sorted(i for i in grid.index if i != 'US')]
        names = {k: v for k, v in STATE_FIPS.items()} | {'US': 'US'}
        limit = max(float(np.nanmax(np.abs(np.log2(grid.to_numpy())))), .1)
        fig, ax = plt.subplots(figsize=(1.3 * grid.shape[1] + 2, .2 * len(grid) + 1.5))
        view = ax.imshow(np.log2(grid.to_numpy()), cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
        for (i, j), value in np.ndenumerate(grid.to_numpy()):
            if np.isfinite(value):
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=5.5)
        ax.set_xticks(range(grid.shape[1]), grid.columns, fontsize=8)
        ax.set_yticks(range(len(grid)), [names.get(i, i) for i in grid.index], fontsize=6)
        bar = fig.colorbar(view, ax=ax, shrink=.5)
        ticks = np.arange(-np.floor(limit), np.floor(limit) + 1)
        bar.set_ticks(ticks, labels=[f'{2 ** t:g}' for t in ticks])
        bar.set_label('WIS ratio to ensemble (log scale)')
        ax.set_title(f'{label(config)}\nWIS ratio by location and season, targets weighted {weights}', fontsize=7)
        fig.tight_layout()
        slug = Scenario.from_string(config).run_id
        paths.append(_save(fig, output / f'heatmap-{slug}.png'))
    return paths


def plot_experiment(folder, ranking, configs=None, dates=None):
    """All four figures for one experiment and one ranking folder; returns the file paths."""
    folder, ranking = Path(folder), Path(ranking)
    output = ranking / 'plots'
    output.mkdir(exist_ok=True)
    for old in output.glob('*.png'):  # a redraw replaces the folder's figures, never mixes selections
        old.unlink()
    settings = json.loads((folder / 'experiment.json').read_text())
    manifest = json.loads((ranking / 'manifest.json').read_text())
    runs, weights = manifest['runs'], manifest['target_weights']
    ranked = pd.read_csv(ranking / 'configuration_ranking.csv', keep_default_na=False).sort_values('combined_mean')
    if configs:
        configs = [Scenario.from_string('' if c == 'default' else c).scenario_string for c in configs]
        missing = set(configs) - set(ranked.config_id)
        if missing:
            raise ValueError(f'Configurations not in {ranking}: {sorted(missing)}')
    else:
        configs = list(ranked.config_id[:2])
    if len(configs) > 2:
        raise ValueError('Fans show at most two configurations')
    panel = load_panel(settings['dataset'])
    scenarios = [Scenario.from_string(c) for c in ranked.config_id]
    seasons = pd.read_csv(ranking / 'season_scores.csv', keep_default_na=False, na_values=[''])
    seasons['config_id'] = seasons.config_id.fillna('')
    return [*cv_layout(panel, scenarios, output), *fans(panel, runs, configs, settings['frozen'], dates, output),
            *dotplot(seasons, weights, output), *heatmaps(runs, configs, weights, output)]


WRITEUP_START, WRITEUP_END = '<!-- write-up: kept across regenerations -->', '<!-- end write-up -->'
FIGURE_ORDER = (('cv-layout-*.png', 'Cross-validation layout'), ('fans-US.png', 'Forecast fans, US'),
                ('fans-NC.png', 'Forecast fans, NC'), ('dotplot.png', 'Configurations ranked, per-seed metrics against the ensemble'),
                ('heatmap-*.png', 'WIS ratio by location and season'))


def write_report(folder, ranking, root='docs/results'):
    """`docs/results/<experiment>/index.md`: the fixed figures, then the hand-written
    write-up (the text between the write-up markers survives every regeneration), then
    the ranking table. Figures are embedded in the page as base64 PNG data (user request
    2026-09-22: visible in the page itself, in any viewer, not one click away)."""
    import base64
    folder, ranking = Path(folder), Path(ranking)
    page_dir = Path(root) / folder.name
    page_dir.mkdir(parents=True, exist_ok=True)
    page = page_dir / 'index.md'
    writeup = '_Not written yet._'
    if page.exists():
        text = page.read_text()
        if WRITEUP_START in text and WRITEUP_END in text:
            writeup = text.split(WRITEUP_START, 1)[1].split(WRITEUP_END, 1)[0].strip()
    manifest = json.loads((ranking / 'manifest.json').read_text())
    ranked = pd.read_csv(ranking / 'configuration_ranking.csv', keep_default_na=False).sort_values('combined_mean')
    seeds = sorted({r['seed'] for r in manifest['runs']})
    lines = [f'# {folder.name}', '',
             f'{len(manifest["runs"])} runs, {len(ranked)} configurations, seeds {", ".join(map(str, seeds))}. '
             f'Ranking `{ranking.name}`: US weight {manifest["us_weight"]}, admissions {manifest["admissions_weight"]}, '
             f'ED {manifest["ed_weight"]}. '
             'Lower WIS ratio is better; 1 = hub ensemble. Figures and table are regenerated by `planner rank`.', '']
    for pattern, title in FIGURE_ORDER:
        for path in sorted((ranking / 'plots').glob(pattern)):
            data = base64.b64encode(path.read_bytes()).decode()
            lines += [f'## {title}' + (f' ({path.stem})' if '*' in pattern else ''), '',
                      f'![{path.stem}](data:image/png;base64,{data})', '']
    lines += ['## Write-up', '', WRITEUP_START, writeup, WRITEUP_END, '', '## Ranking', '']
    targets = [c.removesuffix('_mean') for c in ranked.columns
               if c.endswith('_mean') and c.startswith('wk inc')]
    header = ['Configuration', 'Seeds', 'Combined', 'States/DC', 'US', *targets]
    lines += ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    fmt = lambda row, c: '' if pd.isna(pd.to_numeric(row.get(c), errors='coerce')) else f'{float(row[c]):.3f}'
    for _, row in ranked.iterrows():
        sd = fmt(row, 'combined_sd')
        combined = fmt(row, 'combined_mean') + (f' ± {sd}' if sd else '')
        lines.append('| ' + ' | '.join([f'`{label(row.config_id)}`', str(row.seeds), combined,
                                        fmt(row, 'states_dc_combined_mean'), fmt(row, 'US_combined_mean'),
                                        *[fmt(row, f'{t}_mean') for t in targets]]) + ' |')
    page.write_text('\n'.join(lines) + '\n')
    return page
