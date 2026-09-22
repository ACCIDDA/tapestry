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
3. `pairplot.png`: one point per run (seed), coloured by configuration: combined WIS
   ratio, 50/80/90/95% coverage and the WIS ratio's decomposition into dispersion,
   under- and over-prediction (each component / ensemble WIS per location, weighted
   exactly like the score, so the three sum to the WIS ratio). Grey dashed lines: the
   ensemble's value; red: nominal coverage.
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


def pairplot(seasons, weights, output):
    """Figure 3: per-run metrics, weighted like the score (geography 'all')."""
    columns = {'WIS ratio': 'wis_ratio', **{f'{c}% coverage': f'model_coverage_{c}' for c in COVERAGE},
               **{f'{c} / ens. WIS': f'model_{c}_ratio' for c in COMPONENTS}}
    reference = {'WIS ratio': 1., **{f'{c}% coverage': f'ensemble_coverage_{c}' for c in COVERAGE},
                 **{f'{c} / ens. WIS': f'ensemble_{c}_ratio' for c in COMPONENTS}}
    combined = lambda value: run_scores(seasons, weights, value).query("geography == 'all'").set_index(['config_id', 'seed']).combined
    table = pd.DataFrame({name: combined(value) for name, value in columns.items()}).reset_index()
    table['configuration'] = table.config_id.fillna('').map(label)
    for name, value in reference.items():
        if isinstance(value, str):
            reference[name] = float(combined(value).iloc[0])  # identical for every run (same frozen tasks)
    nominal = {f'{c}% coverage': c / 100 for c in COVERAGE}
    grid = sns.pairplot(table, vars=list(columns), hue='configuration', diag_kind='hist', corner=True,
                        plot_kws=dict(s=30, alpha=.8), height=1.7)
    for i, y in enumerate(columns):
        for j, x in enumerate(columns):
            ax = grid.axes[i][j]
            if ax is None:
                continue
            ax.axvline(reference[x], color=ENSEMBLE_COLOR, ls='--', lw=.8)
            if x in nominal:
                ax.axvline(nominal[x], color='red', ls=':', lw=.8)
            if i != j:
                ax.axhline(reference[y], color=ENSEMBLE_COLOR, ls='--', lw=.8)
                if y in nominal:
                    ax.axhline(nominal[y], color='red', ls=':', lw=.8)
    grid.figure.suptitle('Per-run performance relative to the hub ensemble (one point per seed); '
                         'grey dashed = ensemble, red dotted = nominal coverage', y=1.01)
    grid.figure.savefig(output / 'pairplot.png', dpi=110, bbox_inches='tight')
    plt.close(grid.figure)
    return [output / 'pairplot.png']


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
            *pairplot(seasons, weights, output), *heatmaps(runs, configs, weights, output)]
