"""The experiment figures and report page (user requests 2026-09-22); the only plotting code in the project.

Written by `planner rank` into `<ranking>/plots/` and by `planner plots -e NAME
[--configs A B ...] [--dates D ...]`. Choices (also in docs/design/restructure-2026-unified.md §5):

Labels. Every configuration gets a short label `C<k>`, k = its position in the ranking
(`configuration_ranking.csv` sorted by the main score, 1 = best). Figures show the short
label followed by the full scenario string wrapped at commas (about 40 characters a
line); the report appendix maps each label to its full string, run id, seeds and
non-default fields. `default` stands for the empty scenario string.

Selected configurations (fans and heatmaps): the best-ranked configuration only by
default; `--configs` replaces the selection (any number, in the order given). Fans use
each configuration's lowest seed (seeds are not pooled); heatmaps average over seeds.

1. `cv-layout-*.png`: per fold (row = held-out season) and target (column), the
   finalized US series from panel.npz, the background showing each week's role from
   `dataset.cv.week_roles` (the function `cv.fold` uses): fit, early-stopping
   validation (only with patience > 0; the refit then trains on those weeks), scored
   held-out season, unused. One figure per distinct CV setting among the experiment's
   scenarios. Score episodes may take context from earlier weeks; that is not drawn.
2. `fans-<location>-<hosp|ed>.png` for US and NC (admissions and ED proportions have
   different units, so one file per location x target kind): rows = disease (flu,
   COVID, RSV), columns = hub ensemble, then one column per selected configuration;
   x = time across the three held-out seasons, each panel with the finalized truth
   (panel.npz) and, per reference date, the 50% and 90% intervals and the median over
   horizons 0-3. Panels in a row share y limits, so ensemble and models compare
   directly. Default reference dates: every 4th score reference date of each held-out
   season, starting at its first (read from the first selected configuration's
   export); `--dates` overrides. The hub ensemble is drawn only where frozen support
   exists (e.g. ED in 2025-26 only); its panels are otherwise empty.
3. `dotplot.png` (seaborn PairGrid dot plot, after the seaborn `pairgrid_dotplot`
   example): one row per configuration of the ranking plus one row for the hub
   ensemble, rows ordered by the main score (mean combined WIS ratio over seeds; the
   ensemble sits at 1), best on top; one column per metric: combined WIS ratio for all
   locations, states/DC and US; 50/80/90/95% coverage; dispersion, under- and
   over-prediction over the ensemble's WIS (weighted exactly like the score, so a
   model's three sum to its WIS ratio). Opaque dot = median over seeds, light dots =
   individual seeds. The ensemble row is its own value (WIS ratio 1, its coverage, its
   decomposition over itself, identical for every run since tasks are identical).
   Red dotted = nominal coverage. Figure height grows with the number of rows and the
   wrapped label lines, width with the number of metric columns.
4. `heatmap-C<k>.png`: per selected configuration (file named by its short label;
   the full scenario string is in the figure title and the report heading, since
   scenario strings can exceed file-name limits), one panel per target (rows =
   disease, columns = admissions / ED): WIS ratio (model / ensemble total WIS over
   horizons 0-3) by location x held-out season, mean over seeds, plus the mean of the
   seasons with support. One log colour scale centred at 1, shared by every panel of
   every heatmap file of the call; grey = no frozen support.

`write_report` writes `docs/results/<experiment>/index.md`: summary line, figures (CV
layout, fans, dot plot, heatmaps; base64-embedded; configurations named `C<k>` +
scenario string, as in the appendix), the preserved write-up, the ranking table (with
short labels), and "Appendix: scenarios run". An existing page lacking either
write-up marker raises rather than overwrite hand-written text. `planner rank` calls
it only for the complete ranking (every planned run, default score weights). It also regenerates
the scenario field key `docs/reference/scenario.md` (`write_scenario_key`, from
`Scenario` and `scenario.CODES`/`MEANING`) at `<root>/../reference/scenario.md`, which
the appendix links as `../../reference/scenario.md`.
"""
import base64
import json
from dataclasses import fields
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
from tapestry.model.scenario import CODES, MEANING, Scenario
from .hubs import CHANNEL, export
from .totals import COMPONENTS, COVERAGE, frozen_cases, run_scores

ROLE_COLORS = {'fit': '#cfe0f3', 'validation': '#f6c77b', 'score': '#f3b6b6', 'unused': '#eeeeee'}
ROLE_LABELS = {'fit': 'training (fit)', 'validation': 'early-stopping validation (inner fit only; refit trains on it)',
               'score': 'held-out season (scored)', 'unused': 'unused (outside CV seasons)'}
MODEL_COLOR, ENSEMBLE_COLOR = '#0072b2', '#555555'
FAN_LOCATIONS = ('US', 'NC')
KINDS = {'hosp': ('admissions', lambda t: 'prop ed' not in t), 'ed': ('ED visit proportion', lambda t: 'prop ed' in t)}
FAN_EVERY = 4  # weeks between default fan reference dates


def label(config_id):
    return config_id or 'default'


def short_labels(ranking):
    """{config_id: 'C<k>'}, k = position in the ranking by the main score (1 = best)."""
    ranked = pd.read_csv(Path(ranking) / 'configuration_ranking.csv', keep_default_na=False).sort_values('combined_mean')
    return {config: f'C{k}' for k, config in enumerate(ranked.config_id, 1)}


def wrap(text, width=40):
    """Break a scenario string after commas into lines of about `width` characters."""
    lines = ['']
    for token in text.split(','):
        if lines[-1] and len(lines[-1]) + len(token) + 1 > width:
            lines[-1] += ','
            lines.append(token)
        else:
            lines[-1] += (',' if lines[-1] else '') + token
    return '\n'.join(lines)


def title(config, labels):
    return f'{labels[config]}  {wrap(label(config))}'


def _save(fig, path):
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return path


def cv_layout(panel, scenarios, labels, output):
    """Figure 1: one file per distinct CV setting (patience > 0 and validation_* fields)."""
    dates = np.array([str(d) for d in panel['dates']])
    x = pd.to_datetime(dates)
    us = list(panel['locations']).index('US')
    groups = {}
    for scenario in scenarios:
        effective = scenario.stage('forecast') if scenario.task == 'pipeline' else scenario
        key = (bool(effective.patience), effective.validation_weeks, effective.validation_spacing, effective.validation_offset)
        groups.setdefault(key, []).append(scenario)
    paths = []
    for (early, weeks, spacing, offset), members in groups.items():
        fig, axes = plt.subplots(len(members[0].scored_seasons), len(CHANNEL), figsize=(3.2 * len(CHANNEL), 2.2 * len(members[0].scored_seasons)),
                                 sharex=True, squeeze=False)
        for row, held_out in enumerate(members[0].scored_seasons):
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
        names = ', '.join(labels[s.scenario_string] for s in members)
        fig.suptitle(f'Cross-validation layout, finalized US truth -- {rule}\nconfigurations: {names}', fontsize=10)
        fig.tight_layout(rect=(0, .04, 1, 1))
        name = f'cv-layout-es{weeks}-{spacing}-{offset}.png' if early else 'cv-layout-fixed-epochs.png'
        paths.append(_save(fig, output / name))
    return paths


def default_dates(frames):
    """Every FAN_EVERY-th score reference date of each held-out season, from its first."""
    chosen = []
    for held in dict.fromkeys(key[0] for key in frames):
        chosen += sorted(frames[(held, next(iter(CHANNEL)))].reference_date.unique())[::FAN_EVERY]
    return chosen


def fans(panel, runs, configs, labels, frozen, dates, output):
    """Figure 2: one file per location x target kind; rows = disease, columns = ensemble + configurations."""
    frames = {c: export(min((r for r in runs if r['config_id'] == c), key=lambda r: r['seed'])['path'])
              for c in configs}
    which = (f'reference dates {", ".join(dates)}' if dates
             else f'reference dates every {FAN_EVERY} weeks of each held-out season')
    dates = sorted(dates or default_dates(frames[configs[0]]))
    ensemble = {}
    for case in frozen_cases(frozen):
        table = pd.read_parquet(Path(frozen) / case['directory'] / 'quantiles.parquet')
        ensemble[(case['season'], case['target'])] = table[table.model == case['ensemble']]
    tables = {'hub ensemble': ensemble, **{c: frames[c] for c in configs}}
    truth_dates = pd.to_datetime([str(d) for d in panel['dates']])
    scored_seasons = sorted({held for config in configs for held, _ in frames[config]})
    held_out = np.isin([season(d) for d in truth_dates.strftime('%Y-%m-%d')], scored_seasons)
    fips = {v: k for k, v in STATE_FIPS.items()} | {'US': 'US'}
    band = lambda alpha: plt.Rectangle((0, 0), 1, 1, color=MODEL_COLOR, alpha=alpha, lw=0)
    legend = ([plt.Line2D([], [], color='black', lw=1), plt.Line2D([], [], color=MODEL_COLOR, lw=1.2), band(.3), band(.15)],
              ['finalized truth', 'median', '50% interval', '90% interval'])
    paths = []
    for location in FAN_LOCATIONS:
        l = list(panel['locations']).index(location)
        for kind, (unit, keep) in KINDS.items():
            targets = [t for t in CHANNEL if keep(t)]
            fig, axes = plt.subplots(len(targets), len(tables), figsize=(6 * len(tables), 2.4 * len(targets) + .8),
                                     sharex=True, sharey='row', squeeze=False)
            for column, (source, frame) in enumerate(tables.items()):
                color = ENSEMBLE_COLOR if source == 'hub ensemble' else MODEL_COLOR
                for row, target in enumerate(targets):
                    ax = axes[row, column]
                    ax.plot(truth_dates[held_out], panel['targets'][held_out, l, CHANNEL[target]], color='black', lw=1)
                    parts = [frame[key] for key in ((held, target) for held in scored_seasons) if key in frame]
                    for table in parts:
                        part = table[table.location.eq(fips[location]) & table.reference_date.isin(dates)]
                        for _, fan in part.sort_values('horizon').groupby('reference_date'):
                            x = pd.to_datetime(fan.target_end_date)
                            ax.fill_between(x, fan['q0.05'], fan['q0.95'], color=color, alpha=.15, lw=0)
                            ax.fill_between(x, fan['q0.25'], fan['q0.75'], color=color, alpha=.3, lw=0)
                            ax.plot(x, fan['q0.5'], color=color, lw=1)
                    if row == 0:
                        ax.set_title(source if source == 'hub ensemble' else title(source, labels), fontsize=9)
                    if column == 0:
                        ax.set_ylabel(target.removeprefix('wk inc '), fontsize=9)
                    ax.tick_params(labelsize=7)
            fig.legend(*legend, loc='lower center', ncol=4, fontsize=8, frameon=False)
            fig.suptitle(f'{location}, {unit}: 50%/90% intervals and median (horizons 0-3), {which}; '
                         'hub ensemble only where frozen support exists', fontsize=10)
            fig.tight_layout(rect=(0, .04, 1, 1))
            paths.append(_save(fig, output / f'fans-{location}-{kind}.png'))
    return paths


def dotplot(seasons, weights, labels, output):
    """Figure 3 (seaborn PairGrid dot plot): configurations and the hub ensemble, ranked by
    the main score; opaque dot = median over seeds, light dots = seeds; see module docstring."""
    scored = lambda value, geography='all': run_scores(seasons, weights, value).query(
        'geography == @geography').set_index(['config_id', 'seed']).combined
    columns = {'WIS ratio\nall': ('wis_ratio', 'all'), 'WIS ratio\nstates/DC': ('wis_ratio', 'states_dc'),
               'WIS ratio\nUS': ('wis_ratio', 'US'),
               **{f'{c}%\ncoverage': (f'model_coverage_{c}', 'all') for c in COVERAGE},
               **{f'{c}\n/ ens. WIS': (f'model_{c}_ratio', 'all') for c in COMPONENTS}}
    ensemble = {name: 1. if value == 'wis_ratio' else float(scored(value.replace('model_', 'ensemble_'), geography).iloc[0])
                for name, (value, geography) in columns.items()}
    nominal = {f'{c}%\ncoverage': c / 100 for c in COVERAGE}
    seeds = pd.DataFrame({name: scored(*value) for name, value in columns.items()}).reset_index()
    seeds['row'] = seeds.config_id.map(lambda c: title(c, labels))
    medians = seeds.groupby('row')[list(columns)].median()
    medians.loc['hub ensemble'] = ensemble
    order = list(pd.concat([seeds.groupby('row')['WIS ratio\nall'].mean(), pd.Series({'hub ensemble': 1.})])
                 .sort_values().index)
    lines = max(r.count('\n') + 1 for r in order)
    height = len(order) * (.16 * lines + .22) + .8
    grid = sns.PairGrid(medians.reset_index(), x_vars=list(columns), y_vars=['row'], hue='row', hue_order=order,
                        palette={r: ENSEMBLE_COLOR if r == 'hub ensemble' else MODEL_COLOR for r in order},
                        height=height, aspect=1.35 / height)
    grid.map(sns.stripplot, order=order, size=7, orient='h', jitter=False, linewidth=.6, edgecolor='w')
    for ax, name in zip(grid.axes.flat, columns):
        ax.scatter(seeds[name], [order.index(r) for r in seeds.row], s=26, color=MODEL_COLOR, alpha=.4, lw=0, zorder=1)
        if name in nominal:
            ax.axvline(nominal[name], color='red', ls=':', lw=.8)
        ax.set(xlabel='', ylabel='')
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=7, left=False)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.xaxis.grid(False)
        ax.yaxis.grid(True)
    sns.despine(left=True, bottom=True)
    grid.figure.suptitle('Configurations and hub ensemble ranked by combined WIS ratio (best on top)\n'
                         'opaque dot = median over seeds, light dots = seeds; red dotted = nominal coverage',
                         y=1 + .45 / height, va='bottom', fontsize=9)
    return [_save(grid.figure, output / 'dotplot.png')]


def heatmaps(runs, configs, labels, output):
    """Figure 4: per selected configuration, WIS ratio by location x season, one panel per target."""
    grids = {}
    for config in configs:
        totals = pd.concat([pd.read_csv(Path(r['path']) / 'totals.csv', dtype={'location': str}).assign(seed=r['seed'])
                            for r in runs if r['config_id'] == config])
        cells = totals.groupby(['seed', 'target', 'season', 'location'])[['model_wis', 'ensemble_wis']].sum()
        ratio = (cells.model_wis / cells.ensemble_wis).groupby(['target', 'location', 'season']).mean()
        scored_seasons = sorted(totals.season.unique())
        for target in CHANNEL:
            grid = (ratio.xs(target).unstack('season').reindex(columns=scored_seasons) if target in ratio.index.levels[0]
                    else pd.DataFrame(columns=scored_seasons, dtype=float))
            grid['mean of seasons'] = grid.mean(axis=1)
            grids[(config, target)] = grid
    locations = sorted({i for g in grids.values() for i in g.index} - {'US'})
    locations = ['US'] + locations
    values = np.concatenate([g.to_numpy().ravel() for g in grids.values()])
    limit = max(float(np.nanmax(np.abs(np.log2(values)))) if np.isfinite(values).any() else 0, .1)
    names = dict(STATE_FIPS) | {'US': 'US'}
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#e0e0e0')
    paths = []
    for config in configs:
        kinds = {k: [t for t in CHANNEL if keep(t)] for k, (_, keep) in KINDS.items()}
        fig, axes = plt.subplots(3, 2, figsize=(9, .13 * len(locations) * 3 + 3), layout='constrained', squeeze=False)
        for column, (kind, targets) in enumerate(kinds.items()):
            for row, target in enumerate(targets):
                ax = axes[row, column]
                grid = grids[(config, target)].reindex(locations)
                view = ax.imshow(np.ma.masked_invalid(np.log2(grid.to_numpy(dtype=float))), cmap=cmap,
                                 vmin=-limit, vmax=limit, aspect='auto', interpolation='none')
                for (i, j), value in np.ndenumerate(grid.to_numpy(dtype=float)):
                    if np.isfinite(value):
                        ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=4.5)
                ax.set_xticks(range(grid.shape[1]), [c.replace('mean of seasons', 'mean') for c in grid.columns],
                              fontsize=7)
                ax.set_yticks(range(len(grid)), [names.get(i, i) for i in grid.index], fontsize=5)
                ax.set_title(target.removeprefix('wk inc '), fontsize=9)
        bar = fig.colorbar(view, ax=axes.ravel().tolist(), shrink=.25, aspect=30)
        ticks = [r for r in (1 / 8, 1 / 4, 1 / 2, 2 / 3, .8, 1, 1.25, 1.5, 2, 4, 8) if abs(np.log2(r)) <= limit]
        bar.set_ticks(np.log2(ticks), labels=[f'{r:.2g}' for r in ticks])
        bar.set_label('WIS ratio to hub ensemble (log scale; grey = no frozen support)', fontsize=8)
        fig.suptitle(f'{title(config, labels)}\nWIS ratio by location and held-out season, mean over seeds', fontsize=9)
        paths.append(_save(fig, output / f'heatmap-{labels[config]}.png'))
    return paths


def plot_experiment(folder, ranking, configs=None, dates=None):
    """All figures for one experiment and one ranking folder; returns the file paths."""
    folder, ranking = Path(folder), Path(ranking)
    output = ranking / 'plots'
    output.mkdir(exist_ok=True)
    for old in output.glob('*.png'):  # a redraw replaces the folder's figures, never mixes selections
        old.unlink()
    settings = json.loads((folder / 'experiment.json').read_text())
    manifest = json.loads((ranking / 'manifest.json').read_text())
    runs, weights = manifest['runs'], manifest['target_weights']
    labels = short_labels(ranking)
    if configs:
        configs = [Scenario.from_string('' if c == 'default' else c).scenario_string for c in configs]
        missing = set(configs) - set(labels)
        if missing:
            raise ValueError(f'Configurations not in {ranking}: {sorted(missing)}')
    else:
        configs = list(labels)[:1]
    panel = load_panel(settings['dataset'])
    scenarios = [Scenario.from_string(c) for c in labels]
    seasons = pd.read_csv(ranking / 'season_scores.csv', keep_default_na=False, na_values=[''])
    seasons['config_id'] = seasons.config_id.fillna('')
    return [*cv_layout(panel, scenarios, labels, output),
            *fans(panel, runs, configs, labels, settings['frozen'], dates, output),
            *dotplot(seasons, weights, labels, output), *heatmaps(runs, configs, labels, output)]


def write_scenario_key(path):
    """`docs/reference/scenario.md`: every `Scenario` field with type, default, allowed values, meaning."""
    from tapestry.dataset.build import SOURCE_GROUPS
    lines = ['# Scenario fields', '',
             'Generated from `tapestry.model.scenario.Scenario` (`CODES`, `MEANING`) by '
             '`tapestry.evaluation.plots.write_scenario_key`, rewritten by every report; do not edit by hand. '
             'A scenario string names only the fields that differ from these defaults, as `key=value` tokens '
             'joined by `,`; booleans are written `0`/`1`. "Design" = '
             '[the unified design](../design/restructure-2026-unified.md); other documents are under `docs/`.', '',
             '| Field | Type | Default | Allowed values | Meaning |', '|---|---|---|---|---|']
    for f in fields(Scenario):
        if f.name in ('nowcast', 'forecast'):
            lines.append(f'| `{f.name}.<field>` | stage override | inherit | Model/training fields | {MEANING[f.name]} |')
            continue
        allowed = (', '.join(f'`{v}`' for v in sorted(CODES[f.name])) if f.name in CODES
                   else '`0`, `1`' if f.type is bool
                   else '`+`-joined subset of ' + ', '.join(f'`{g}`' for g in SOURCE_GROUPS) if f.name == 'covariate_set'
                   else f'any {f.type.__name__} (checked in `Scenario.__post_init__`)')
        default = f"`{f.default!r}`" if isinstance(f.default, str) else f'`{int(f.default) if f.type is bool else f.default}`'
        lines.append(f'| `{f.name}` | {f.type.__name__} | {default} | {allowed} | {MEANING.get(f.name, "")} |')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n')
    return path


WRITEUP_START, WRITEUP_END = '<!-- write-up: kept across regenerations -->', '<!-- end write-up -->'
FIGURE_ORDER = (('cv-layout-*.png', 'Cross-validation layout'),
                *((f'fans-{l}-{k}.png', f'Forecast fans, {l}, {u}') for l in FAN_LOCATIONS for k, (u, _) in KINDS.items()),
                ('dotplot.png', 'Configurations and hub ensemble ranked, per-seed metrics'),
                ('heatmap-*.png', 'WIS ratio by location, season and target'))


def write_report(folder, ranking, root='docs/results'):
    """`<root>/<experiment>/index.md`: figures (base64-embedded, user request 2026-09-22:
    visible in the page itself, not one click away), the hand-written write-up (text
    between the markers survives every regeneration), the ranking table, and the
    appendix of scenarios run. Also rewrites the field key `<root>/../reference/scenario.md`."""
    folder, ranking = Path(folder), Path(ranking)
    page_dir = Path(root) / folder.name
    page_dir.mkdir(parents=True, exist_ok=True)
    page = page_dir / 'index.md'
    write_scenario_key(Path(root).parent / 'reference' / 'scenario.md')
    writeup = '_Not written yet._'
    if page.exists():
        text = page.read_text()
        if WRITEUP_START not in text or WRITEUP_END not in text:
            raise ValueError(f'{page} exists without both write-up markers ({WRITEUP_START!r}, {WRITEUP_END!r}); '
                             'add them around the hand-written text (or move the page away) so it is not overwritten')
        writeup = text.split(WRITEUP_START, 1)[1].split(WRITEUP_END, 1)[0].strip()
    manifest = json.loads((ranking / 'manifest.json').read_text())
    ranked = pd.read_csv(ranking / 'configuration_ranking.csv', keep_default_na=False).sort_values('combined_mean')
    labels = short_labels(ranking)
    configs = {short: config for config, short in labels.items()}
    seeds = sorted({r['seed'] for r in manifest['runs']})
    lines = [f'# {folder.name}', '',
             f'{len(manifest["runs"])} runs, {len(ranked)} configurations, seeds {", ".join(map(str, seeds))}. '
             f'Ranking `{ranking.name}`: US weight {manifest["us_weight"]}, admissions {manifest["admissions_weight"]}, '
             f'ED {manifest["ed_weight"]}. '
             'Lower WIS ratio is better; 1 = hub ensemble. Figures, table and appendix are regenerated by '
             '`planner rank`; `C<k>` labels are ranking positions, see the appendix.', '']
    for pattern, heading in FIGURE_ORDER:
        for path in sorted((ranking / 'plots').glob(pattern)):
            data = base64.b64encode(path.read_bytes()).decode()
            suffix = path.stem.removeprefix('heatmap-')
            named = f'{suffix}: `{label(configs[suffix])}`' if pattern.startswith('heatmap') else path.stem
            lines += [f'## {heading}' + (f' ({named})' if '*' in pattern else ''), '',
                      f'![{path.stem}](data:image/png;base64,{data})', '']
    lines += ['## Write-up', '', WRITEUP_START, writeup, WRITEUP_END, '', '## Ranking', '']
    targets = [c.removesuffix('_mean') for c in ranked.columns
               if c.endswith('_mean') and c.startswith('wk inc')]
    header = ['Label', 'Configuration', 'Seeds', 'Combined', 'States/DC', 'US', *targets]
    lines += ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    fmt = lambda row, c: '' if pd.isna(pd.to_numeric(row.get(c), errors='coerce')) else f'{float(row[c]):.3f}'
    for _, row in ranked.iterrows():
        sd = fmt(row, 'combined_sd')
        combined = fmt(row, 'combined_mean') + (f' ± {sd}' if sd else '')
        lines.append('| ' + ' | '.join([labels[row.config_id], f'`{label(row.config_id)}`', str(row.seeds), combined,
                                        fmt(row, 'states_dc_combined_mean'), fmt(row, 'US_combined_mean'),
                                        *[fmt(row, f'{t}_mean') for t in targets]]) + ' |')
    defaults = Scenario()
    lines += ['', '## Appendix: scenarios run', '',
              'Every configuration in this ranking. Field meanings, defaults and allowed values: '
              '[scenario field key](../../reference/scenario.md). "hub ensemble" in the figures is the frozen '
              'hub ensemble on the same tasks (WIS ratio 1).', '',
              '| Label | Scenario string | Run id | Seeds | Non-default fields |', '|---|---|---|---|---|']
    for config, short in labels.items():
        scenario = Scenario.from_string(config)
        changed = '; '.join(f'`{f.name}` = {getattr(scenario, f.name)!r} (default {f.default!r})'
                            for f in fields(Scenario) if getattr(scenario, f.name) != getattr(defaults, f.name)) or 'none'
        run_seeds = sorted(r['seed'] for r in manifest['runs'] if r['config_id'] == config)
        lines.append(f'| {short} | `{label(config)}` | `{scenario.run_id}` | {", ".join(map(str, run_seeds))} | {changed} |')
    page.write_text('\n'.join(lines) + '\n')
    return page
