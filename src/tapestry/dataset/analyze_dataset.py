"""Describe `panel.npz`: size of the vintaged truth fallback, revisions, covariate availability.

Run after every panel rebuild:

    python -m tapestry.dataset.analyze_dataset [--dataset data/processed/panel.npz] [--output docs/data/panel.md]

Writes one Markdown page with base64-embedded PNG figures (as the experiment reports do)
and tables whose numbers are all computed here; no hand-written claims.

Definitions are the episode builder's own (`episodes.episodes`, user decision 2026-09-22
to keep the truth fallback): vintaged episodes are cut with `lookback = asof_weeks =
WEEKS`, so every one of the last WEEKS context weeks goes through the as-of rule. For a
context cell of issuance w, *weeks before issuance* j = 0 is the context end (the
Saturday four days before the Wednesday), j = 1 the week before, and so on.

- **Fallback** = `known_final` inside the as-of window: nothing was visible at the
  cutoff and the final truth fills the cell. Share = fallback cells / available cells
  (`available` = visible as of the cutoff or filled from truth). Cells with neither are
  excluded (they are unavailable in the episode either way). A missing archive and a
  week not yet reported both count as fallback; the panel cannot tell them apart.
- **Revision** = (as-of - final) / final on cells where an as-of value was visible
  (`available & ~known_final`) and the final truth is > 0.
- **Covariates** have no truth fallback in episodes; "not visible" = final truth
  available but no as-of value at the cutoff, share of cells with final truth.
  Kinsa's as-of archive starts 2026-04-06, so its share is also given from then on.
- Season = `cv.season` of the context end (CDC epiweek 31-30).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .build import PANEL_DATASET, context_end, load
from .cv import season
from .episodes import episodes, select_covariates

WEEKS = 9  # j = 0..8 weeks before issuance
KINSA_ASOF_START = '2026-04-06'
OUTPUT = 'docs/data/panel.md'
PLACES = ('US', 'NC')


def _png(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    return f'![](data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()})'


def _cells(panel, eps):
    """Long table of every as-of-window target cell: issuance, season, j, target, location, fallback, revision."""
    names, locations = list(panel['target_names']), [str(l) for l in panel['locations']]
    index = {str(d): i for i, d in enumerate(panel['dates'])}
    rows = []
    for ep in eps:
        for i, day in enumerate(ep['context_dates']):
            if day not in index:
                continue  # padding before the calendar: never available
            j = WEEKS - 1 - i
            final = panel['targets'][index[day]].T  # [C, L]
            available, fallback = ep['available'][i], ep['known_final'][i]
            asof = np.where(available & ~fallback, ep['values'][i], np.nan)
            with np.errstate(divide='ignore', invalid='ignore'):
                revision = np.where(final > 0, (asof - final) / final, np.nan)
            c, l = np.nonzero(available)
            rows.append(pd.DataFrame(dict(issuance=ep['issuance'], j=j, target=np.array(names)[c],
                                          location=np.array(locations)[l], fallback=fallback[c, l],
                                          revision=revision[c, l])))
    cells = pd.concat(rows, ignore_index=True)
    cells['season'] = cells.issuance.map(lambda d: season(context_end(d)))
    count = cells.groupby('season').issuance.nunique()  # label each season with its issuance count
    cells['season'] = cells.season.map(lambda s: f'{s} ({count[s]} iss.)')
    return cells


def _whole_issuance(cells):
    """Issuances in which every available j <= 1 cell of a target fell back (no as-of at all), per season."""
    recent = cells[cells.j <= 1].groupby(['target', 'season', 'issuance']).fallback.all()
    return recent.groupby(['target', 'season']).agg(['sum', 'size']).rename(
        columns={'sum': 'all fallback', 'size': 'issuances'}).astype(int)


def _share_table(cells, scope):
    frame = cells if scope == 'states+US' else cells[cells.location == 'US']
    return frame.pivot_table(index=['target', 'season'], columns='j', values='fallback', aggfunc='mean') * 100


def _heatmaps(table, title):
    targets = list(dict.fromkeys(table.index.get_level_values(0)))
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.5), sharex=True, sharey=True, constrained_layout=True)
    for ax, target in zip(axes.flat, targets):
        sns.heatmap(table.loc[target], ax=ax, vmin=0, vmax=100, cmap='rocket_r', annot=True, fmt='.0f',
                    annot_kws=dict(size=8), cbar=ax is axes.flat[-1], cbar_kws=dict(label='% fallback'))
        ax.set_title(target, fontsize=10)
        ax.set_xlabel('weeks before issuance (0 = context end)')
        ax.set_ylabel('')
        ax.grid(False)
        ax.tick_params(axis='y', labelrotation=0)
    fig.suptitle(title)
    return _png(fig)


def _per_issuance(cells):
    frame = cells[cells.j <= 3].groupby(['target', 'j', 'issuance']).fallback.mean().mul(100).reset_index()
    frame['issuance'] = pd.to_datetime(frame.issuance)
    grid = sns.relplot(frame, x='issuance', y='fallback', hue='j', col='target', col_wrap=3, kind='line',
                       height=2.6, aspect=1.8, palette='viridis', facet_kws=dict(sharey=True))
    grid.set_titles('{col_name}')
    grid.set_axis_labels('Wednesday issuance', '% fallback (states+US)')
    grid.legend.set_title('weeks before\nissuance')
    sns.move_legend(grid, 'upper left', bbox_to_anchor=(1.0, 0.95))
    for ax in grid.axes.flat:
        ax.tick_params(axis='x', labelrotation=30)
    return _png(grid.figure)


def _revisions(cells):
    frame = cells.dropna(subset=['revision'])
    table = frame.groupby(['target', 'j']).revision.describe(percentiles=[.1, .5, .9])
    table = table[['count', '10%', '50%', '90%']].rename(columns={'10%': 'p10', '50%': 'median', '90%': 'p90'})
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True, constrained_layout=True)
    for ax, target in zip(axes.flat, dict.fromkeys(table.index.get_level_values(0))):
        part = table.loc[target] * [1, 100, 100, 100]
        ax.fill_between(part.index, part.p10, part.p90, alpha=.3, label='10-90%')
        ax.plot(part.index, part['median'], marker='o', label='median')
        ax.axhline(0, color='k', lw=.6)
        ax.set_title(target, fontsize=10)
        ax.set_xlabel('weeks before issuance')
        ax.set_ylabel('(as-of - final) / final, %')
    axes.flat[0].legend()
    fig.suptitle('Revision of visible as-of target values (US and states pooled, all issuances)')
    return table, _png(fig)


def _covariates(panel, eps):
    """% of cells with final truth but no as-of value, per covariate and j."""
    names = [*map(str, panel['covariate_names']), *map(str, panel['covariate_national_names'])]
    truth = select_covariates(panel['covariates'], panel['covariates_national'], panel['covariate_names'],
                              panel['covariate_national_names'], panel['locations'], names)[1]  # [T, K, L]
    index = {str(d): i for i, d in enumerate(panel['dates'])}
    counts, recent = {}, []
    for ep in eps:
        late = ep['issuance'] >= KINSA_ASOF_START
        for i, day in enumerate(ep['context_dates']):
            if day not in index:
                continue
            final, seen = truth[index[day]], ep['covariates'][i, :, 1].astype(bool)
            for k, name in enumerate(names):
                keys = [(name, WEEKS - 1 - i)] + ([(f'{name} (from {KINSA_ASOF_START})', WEEKS - 1 - i)]
                                                   if name == 'kinsa_ili' and late else [])
                for key in keys:
                    has = final[k]
                    total, missing = counts.get(key, (0, 0))
                    counts[key] = (total + has.sum(), missing + (has & ~seen[k]).sum())
                if WEEKS - 1 - i == 1 and final[k].any():
                    recent.append((ep['issuance'], name, 100 * (final[k] & ~seen[k]).sum() / final[k].sum()))
    table = pd.Series({k: 100 * m / t for k, (t, m) in counts.items() if t}).unstack()
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    sns.heatmap(table, ax=ax, vmin=0, vmax=100, cmap='rocket_r', annot=True, fmt='.0f', annot_kws=dict(size=8),
                cbar_kws=dict(label='% not visible at cutoff'))
    ax.set_xlabel('weeks before issuance (0 = context end)')
    ax.set_ylabel('')
    ax.set_title('Covariates: final value exists but nothing visible at the Wednesday cutoff (all locations)')
    frame = pd.DataFrame(recent, columns=['issuance', 'covariate', 'share'])
    frame['issuance'] = pd.to_datetime(frame.issuance)
    frame['source'] = frame.covariate.str.replace(r'_(flu|covid|rsv)', '', regex=True)
    frame['pathogen'] = frame.covariate.str.extract(r'(flu|covid|rsv)', expand=False).fillna('ili')
    grid = sns.relplot(frame, x='issuance', y='share', hue='pathogen', col='source', col_wrap=3, kind='line',
                       height=2.4, aspect=1.9, lw=1, palette={'flu': 'C0', 'covid': 'C1', 'rsv': 'C2', 'ili': 'C4'})
    grid.set_titles('{col_name}')
    grid.set_axis_labels('Wednesday issuance', '% not visible, j = 1')
    grid.figure.suptitle('Covariates per issuance, week before the context end (j = 1), all locations', y=1.03)
    for ax in grid.axes.flat:
        ax.tick_params(axis='x', labelrotation=30)
    line = grid.figure
    return table, _png(fig), _png(line)


def _covariate_lines(panel):
    """One axes per location: every covariate's final series / its max, faint as-of latest-week value."""
    names = [*map(str, panel['covariate_names']), *map(str, panel['covariate_national_names'])]
    args = (panel['covariate_names'], panel['covariate_national_names'], panel['locations'], names)
    values, available = select_covariates(panel['covariates'], panel['covariates_national'], *args)  # [T, K, L]
    asof_values, asof_available = select_covariates(panel['asof_covariates'], panel['asof_covariates_national'], *args)
    dates = pd.to_datetime(panel['dates'])
    ends = pd.to_datetime(panel['issuance_dates']) - pd.Timedelta(days=4)
    week = {d: t for t, d in enumerate(dates)}
    colour = {'flu': 'C0', 'covid': 'C1', 'rsv': 'C2', 'kinsa': 'C4'}
    style = {'inpatient': '-', 'outpatient': '--', 'wval_like': ':', 'pct_rank': '-.', 'kinsa': '-'}
    pick = lambda table, name: next(v for k, v in table.items() if k in name)
    locations = [str(l) for l in panel['locations']]
    fig, axes = plt.subplots(len(PLACES), 1, figsize=(14, 4.2 * len(PLACES)), sharex=True, constrained_layout=True)
    for ax, place in zip(axes, PLACES):
        l = locations.index(place)
        for k, name in enumerate(names):
            if not available[:, k, l].any():
                ax.plot([], [], color='0.6', ls=':', label=f'{name} (not available at {place})')
                continue
            series = np.where(available[:, k, l], values[:, k, l], np.nan)
            scale = np.nanmax(series)
            c, s = pick(colour, name), pick(style, name)
            ax.plot(dates, series / scale, color=c, ls=s, lw=1.4, label=name)
            # faint: the context-end week as visible at each Wednesday cutoff, same scale
            latest = [asof_values[w, week[e], k, l] if e in week and asof_available[w, week[e], k, l] else np.nan
                      for w, e in enumerate(ends)]
            ax.plot(ends, np.array(latest) / scale, color=c, ls='none', marker='.', ms=3, alpha=.35)
        ax.set_title(f'{place}: covariates, final value / own max (dots: context-end week as of each Wednesday)')
        ax.set_ylabel('value / max')
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    return _png(fig)


def _markdown(frame, digits=1):
    frame = frame.round(digits)
    head = '| ' + ' | '.join(map(str, frame.columns)) + ' |'
    lines = [head, '|' + '---|' * len(frame.columns)]
    lines += ['| ' + ' | '.join('' if pd.isna(v) else str(v) for v in row) + ' |' for row in frame.itertuples(index=False)]
    return '\n'.join(lines)


def analyze(dataset=PANEL_DATASET, output=OUTPUT):
    sns.set_theme(style='whitegrid', font_scale=.9)
    panel = load(dataset)
    metadata = json.loads(str(panel['metadata']))
    names = [*map(str, panel['covariate_names']), *map(str, panel['covariate_national_names'])]
    eps = episodes(panel, WEEKS, 'vintaged', covariate_names=names, asof_weeks=WEEKS)
    cells = _cells(panel, eps)
    tables = {scope: _share_table(cells, scope) for scope in ('states+US', 'US')}
    overall = (cells[cells.j <= 2].pivot_table(index='target', columns='j', values='fallback', aggfunc='mean') * 100)
    overall_us = (cells[(cells.j <= 2) & (cells.location == 'US')]
                  .pivot_table(index='target', columns='j', values='fallback', aggfunc='mean') * 100)
    revision_table, revision_figure = _revisions(cells)
    covariate_table, covariate_figure, covariate_line = _covariates(panel, eps)
    sha = hashlib.sha256(Path(dataset).read_bytes()).hexdigest()
    headline = pd.concat({'states+US': overall, 'US': overall_us}, axis=1)
    headline.columns = [f'{scope} j={j}' for scope, j in headline.columns]

    lines = [
        '# Dataset panel: truth fallback, revisions, covariates', '',
        f'Generated by `python -m tapestry.dataset.analyze_dataset --dataset {dataset}` '
        '(rerun after every panel rebuild). All numbers and figures are computed from the panel; '
        'definitions are in the module docstring and below. Design: '
        '[unified restructuring §3](../design/restructure-2026-unified.md).', '',
        '## Panel', '',
        f'- file: `{dataset}`, sha256 `{sha}`, {Path(dataset).stat().st_size / 1e6:.1f} MB',
        f'- build: version {metadata.get("version")}, truth day {metadata.get("truth_day")}, '
        f'calendar {metadata.get("start")} to {metadata.get("end")}',
        f'- dimensions: {len(panel["dates"])} Saturdays x {len(panel["locations"])} locations x '
        f'{len(panel["target_names"])} targets; {len(panel["covariate_names"])} state covariates, '
        f'{len(panel["covariate_national_names"])} national covariate(s)',
        f'- issuances: {len(panel["issuance_dates"])} Wednesdays, {panel["issuance_dates"][0]} to '
        f'{panel["issuance_dates"][-1]}; {len(eps)} vintaged episodes kept (lookback = asof_weeks = {WEEKS})',
        '- raw snapshots: ' + ', '.join(f'`{k}` {v}' for k, v in metadata.get('snapshots', {}).items()), '',
        '## 1. Truth fallback in vintaged episodes', '',
        'User decision 2026-09-22: in vintaged mode, a target cell in the as-of window with nothing '
        'visible at the Wednesday cutoff is filled with the final truth and flagged `known_final=True` '
        '(reproduces old B1). This section measures how many cells that is. j = weeks before issuance: '
        'j = 0 is the context end (Saturday four days before the Wednesday). Share = fallback cells / '
        'available cells; a missing archive and a not-yet-reported week both count as fallback. The '
        'default `asof_weeks=2` uses j = 0 and 1 only; larger `asof_weeks` use more columns.', '',
        '### Headline: all seasons pooled, % fallback', '',
        'The last season listed is the one in progress at the truth day (few issuances).', '',
        _markdown(headline.reset_index(), 1), '',
        _heatmaps(tables['states+US'], '% of available target cells filled from final truth (states + DC + US)'), '',
        _heatmaps(tables['US'], '% of available target cells filled from final truth (US only)'), '',
        '### Whole-issuance fallback', '',
        'Issuances (per target, season of the context end) in which no available j <= 1 cell had an '
        'as-of value, i.e. every cell of the default as-of window came from final truth, over all '
        'issuances with an available cell.', '',
        _markdown(_whole_issuance(cells).reset_index(), 0), '',
        '### Per issuance (states+US, j = 0..3)', '',
        _per_issuance(cells), '',
        '### Table, states+US, % fallback by j', '',
        _markdown(tables['states+US'].reset_index()), '',
        '### Table, US only, % fallback by j', '',
        _markdown(tables['US'].reset_index()), '',
        '## 2. Revision where an as-of value was visible', '',
        '(as-of - final) / final in %, on cells with a visible as-of value and final > 0, US and states '
        'pooled, all issuances.', '',
        revision_figure, '',
        _markdown(pd.concat([revision_table[['count']], revision_table[['p10', 'median', 'p90']] * 100], axis=1)
                  .reset_index(), 1), '',
        '## 3. Covariates: final value present but not visible at the cutoff', '',
        'Episodes have no truth fallback for covariates: such cells are unavailable. % of cells with a '
        f'final value, all locations and issuances. Kinsa (US only) has as-of data only from '
        f'{KINSA_ASOF_START}; its row restricted to issuances from then on is shown separately.', '',
        covariate_figure, '',
        covariate_line, '',
        _markdown(covariate_table.reset_index().rename(columns={'index': 'covariate'}), 1), '',
        '## 4. Covariates in ' + ' and '.join(PLACES), '',
        'Final (truth-day) values, each divided by its own maximum at that location over the calendar. '
        'Colour = pathogen, line style = source (solid inpatient claims, dashed outpatient claims, dotted '
        'NWSS wval_like, dash-dot NWSS pct_rank; Kinsa purple). Faint dots: the context-end week of each '
        'Wednesday issuance as visible at its cutoff, same scale (the latest as-of value an episode sees). '
        'A covariate with no value at a location is listed in the legend as not available.', '',
        _covariate_lines(panel), '',
    ]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text('\n'.join(lines))
    return dict(output=output, sha256=sha, episodes=len(eps), fallback_pooled=headline.round(1).to_dict())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default=PANEL_DATASET)
    parser.add_argument('--output', default=OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.dataset, args.output), indent=1, default=str))


if __name__ == '__main__':
    main()
