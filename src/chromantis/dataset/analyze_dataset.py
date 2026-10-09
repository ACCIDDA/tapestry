"""Describe `panel.npz`: how much of a vintaged as-of window was unpublished, revisions, covariates.

Run after every panel rebuild:

    python -m chromantis.dataset.analyze_dataset [--dataset data/processed/panel.npz] [--output docs/data/panel.md]

Writes one Markdown page with base64-embedded PNG figures (as the experiment reports do)
and tables whose numbers are all computed here; no hand-written claims.

Definitions are the episode builder's own (`episodes.episodes`, user decision 2026-09-22
that an unpublished cell is unavailable): vintaged episodes are cut with `lookback = asof_weeks =
WEEKS`, so every one of the last WEEKS context weeks goes through the as-of rule. For a
context cell of issuance w, *weeks before issuance* j = 0 is the context end (the
Saturday four days before the Wednesday), j = 1 the week before, and so on.

- **Unpublished** = nothing was visible at the cutoff, so the episode cell is
  unavailable (user decision 2026-09-22: no truth fallback). Share = unpublished cells
  / cells that exist at all (visible at the cutoff or present in the final truth); the
  denominator is the one the truth fallback used before, so the shares are comparable
  with the pre-2026-09-22 "fallback" ones. A missing archive and a week not yet
  reported both count as unpublished; the panel cannot tell them apart.
- **Revision** = (as-of - final) / final on cells with a visible as-of value and a
  final truth > 0.
- **Covariates** are unavailable when not visible, as targets now are; "not visible" = final truth
  available but no as-of value at the cutoff, share of cells with final truth.
  Kinsa's as-of archive starts 2026-04-06, so its share is also given from then on.
- Season = `cv.season` of the context end (CDC epiweek 31-30).
- **Wastewater** (section 5): the two NWSS indices per pathogen from the panel's truth
  and as-of arrays; coverage = number of states + DC with a value per week (US excluded).
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
WASTEWATER_PLACES = ('US', 'NC', 'CA', 'NY', 'TX')  # section 5: US, NC and a few large states


def _png(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    return f'![](data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()})'


def _cells(panel, eps):
    """Long table of every as-of-window target cell: issuance, season, j, target, location, unpublished, revision."""
    names, locations = list(panel['target_names']), [str(l) for l in panel['locations']]
    index = {str(d): i for i, d in enumerate(panel['dates'])}
    rows = []
    for ep in eps:
        for i, day in enumerate(ep['context_dates']):
            if day not in index:
                continue  # padding before the calendar: never available
            j = WEEKS - 1 - i
            final = panel['targets'][index[day]].T  # [C, L]
            visible = ep['available'][i]  # in the as-of window this is exactly "published at the cutoff"
            known = visible | np.isfinite(final)  # the cell exists at all (as-of or final truth)
            asof = np.where(visible, ep['values'][i], np.nan)
            with np.errstate(divide='ignore', invalid='ignore'):
                revision = np.where(final > 0, (asof - final) / final, np.nan)
            c, l = np.nonzero(known)
            rows.append(pd.DataFrame(dict(issuance=ep['issuance'], j=j, target=np.array(names)[c],
                                          location=np.array(locations)[l], unpublished=~visible[c, l],
                                          revision=revision[c, l])))
    cells = pd.concat(rows, ignore_index=True)
    cells['season'] = cells.issuance.map(lambda d: season(context_end(d)))
    count = cells.groupby('season').issuance.nunique()  # label each season with its issuance count
    cells['season'] = cells.season.map(lambda s: f'{s} ({count[s]} iss.)')
    return cells


def _whole_issuance(cells):
    """Issuances in which every available j <= 1 cell of a target fell back (no as-of at all), per season."""
    recent = cells[cells.j <= 1].groupby(['target', 'season', 'issuance']).unpublished.all()
    return recent.groupby(['target', 'season']).agg(['sum', 'size']).rename(
        columns={'sum': 'all unpublished', 'size': 'issuances'}).astype(int)


def _share_table(cells, scope):
    frame = cells if scope == 'states+US' else cells[cells.location == 'US']
    return frame.pivot_table(index=['target', 'season'], columns='j', values='unpublished', aggfunc='mean') * 100


def _heatmaps(table, title):
    targets = list(dict.fromkeys(table.index.get_level_values(0)))
    fig, axes = plt.subplots(2, 3, figsize=(15, 6.5), sharex=True, sharey=True, constrained_layout=True)
    for ax, target in zip(axes.flat, targets):
        sns.heatmap(table.loc[target], ax=ax, vmin=0, vmax=100, cmap='rocket_r', annot=True, fmt='.0f',
                    annot_kws=dict(size=8), cbar=ax is axes.flat[-1], cbar_kws=dict(label='% unpublished'))
        ax.set_title(target, fontsize=10)
        ax.set_xlabel('weeks before issuance (0 = context end)')
        ax.set_ylabel('')
        ax.grid(False)
        ax.tick_params(axis='y', labelrotation=0)
    fig.suptitle(title)
    return _png(fig)


def _per_issuance(cells):
    frame = cells[cells.j <= 3].groupby(['target', 'j', 'issuance']).unpublished.mean().mul(100).reset_index()
    frame['issuance'] = pd.to_datetime(frame.issuance)
    grid = sns.relplot(frame, x='issuance', y='unpublished', hue='j', col='target', col_wrap=3, kind='line',
                       height=2.6, aspect=1.8, palette='viridis', facet_kws=dict(sharey=True))
    grid.set_titles('{col_name}')
    grid.set_axis_labels('Wednesday issuance', '% unpublished (states+US)')
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
    frame.loc[frame.covariate.eq('ilinet_ili'), 'pathogen'] = 'ili'  # ILI, not an influenza-specific signal
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
    colour = {'flu': 'C0', 'covid': 'C1', 'rsv': 'C2', 'kinsa': 'C4', 'ilinet': 'C5'}
    style = {'inpatient': '-', 'outpatient': '--', 'wval_like': ':', 'pct_rank': '-.', 'kinsa': '-',
             'ilinet': '-', 'clinical_lab': (0, (5, 1)), 'flusurv': (0, (3, 1, 1, 1, 1, 1))}
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


def _wastewater(panel):
    """NWSS indices at WASTEWATER_PLACES (final line, as-of dots) and state coverage per week."""
    names = [str(n) for n in panel['covariate_names']]
    locations = [str(l) for l in panel['locations']]
    dates = pd.to_datetime(panel['dates'])
    ends = pd.to_datetime(panel['issuance_dates']) - pd.Timedelta(days=4)
    week = {d: t for t, d in enumerate(dates)}
    pathogens, indices = ('flu', 'covid', 'rsv'), ('wval_like', 'pct_rank')
    colour = {'flu': 'C0', 'covid': 'C1', 'rsv': 'C2'}
    k_of = {(p, i): names.index(f'nwss_{p}_{i}') for p in pathogens for i in indices}
    has = ~np.isnan(panel['covariates'][:, :, list(k_of.values())]).all(axis=(1, 2))
    first, last = dates[has].min(), dates[has].max()
    figures = []
    for index in indices:
        places = [p for p in WASTEWATER_PLACES if p in locations]
        fig, axes = plt.subplots(len(places), len(pathogens), figsize=(15, 2.3 * len(places)), sharex=True,
                                 sharey=index == 'pct_rank', squeeze=False, constrained_layout=True)
        for row, place in zip(axes, places):
            l = locations.index(place)
            for ax, pathogen in zip(row, pathogens):
                k = k_of[pathogen, index]
                ax.plot(dates, panel['covariates'][:, l, k], color=colour[pathogen], lw=1.3, label='final')
                latest = [panel['asof_covariates'][w, week[e], l, k] if e in week else np.nan
                          for w, e in enumerate(ends)]
                ax.plot(ends, latest, ls='none', marker='.', ms=4, color='k', alpha=.5,
                        label='context-end week as of each Wednesday')
                ax.set_title(f'{place} {pathogen}', fontsize=9)
                ax.set_xlim(first - pd.Timedelta(weeks=2), last + pd.Timedelta(weeks=2))
                ax.tick_params(axis='x', labelrotation=30)
            row[0].set_ylabel(index)
        axes[0, 0].legend(fontsize=7, loc='upper left')
        fig.suptitle(f'NWSS {index}: final (truth-day) value, and the latest week as visible at each Wednesday cutoff')
        figures.append(_png(fig))
    states = [i for i, l in enumerate(locations) if l != 'US']
    counts, rows = {}, []
    fig, axes = plt.subplots(1, len(indices), figsize=(15, 3.6), sharey=True, constrained_layout=True)
    for ax, index in zip(axes, indices):
        for pathogen in pathogens:
            k = k_of[pathogen, index]
            final = (~np.isnan(panel['covariates'][:, states, k])).sum(axis=1)
            asof = [(~np.isnan(panel['asof_covariates'][w, week[e], states, k])).sum() if e in week else np.nan
                    for w, e in enumerate(ends)]
            ax.plot(dates, final, color=colour[pathogen], lw=1.3, label=f'{pathogen} final')
            ax.plot(ends, asof, color=colour[pathogen], ls='none', marker='.', ms=4, alpha=.6,
                    label=f'{pathogen} as of Wednesday (context-end week)')
            seen = final[has]
            rows.append(dict(index=index, pathogen=pathogen, weeks_with_any_state=int((final > 0).sum()),
                             states_median=float(np.median(seen)), states_max=int(seen.max()),
                             issuances_with_asof=int(np.nansum(np.array(asof) > 0))))
        ax.set_title(index)
        ax.set_ylabel('states + DC with a value')
        ax.set_xlim(first - pd.Timedelta(weeks=2), last + pd.Timedelta(weeks=2))
        ax.tick_params(axis='x', labelrotation=30)
    axes[-1].legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
    fig.suptitle('NWSS coverage: number of states + DC with a value per week (US excluded)')
    table = pd.DataFrame(rows)
    return figures, _png(fig), table, (first.date(), last.date())


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
    overall = (cells[cells.j <= 2].pivot_table(index='target', columns='j', values='unpublished', aggfunc='mean') * 100)
    overall_us = (cells[(cells.j <= 2) & (cells.location == 'US')]
                  .pivot_table(index='target', columns='j', values='unpublished', aggfunc='mean') * 100)
    revision_table, revision_figure = _revisions(cells)
    covariate_table, covariate_figure, covariate_line = _covariates(panel, eps)
    ww_figures, ww_coverage, ww_table, (ww_first, ww_last) = _wastewater(panel)
    sha = hashlib.sha256(Path(dataset).read_bytes()).hexdigest()
    headline = pd.concat({'states+US': overall, 'US': overall_us}, axis=1)
    headline.columns = [f'{scope} j={j}' for scope, j in headline.columns]

    lines = [
        '# Dataset panel: unpublished cells, revisions, covariates', '',
        f'Generated by `python -m chromantis.dataset.analyze_dataset --dataset {dataset}` '
        '(rerun after every panel rebuild). All numbers and figures are computed from the panel; '
        'definitions are in the module docstring and below. Design: '
        '[data contract](index.md#current-dataset).', '',
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
        '## 1. Unpublished cells in vintaged episodes', '',
        'In vintaged mode, a target cell in the as-of window with '
        'nothing visible at the Wednesday cutoff is **unavailable** (`available=False`), not filled with '
        'final truth. This section measures how many cells that is. j = weeks before issuance: j = 0 is '
        'the context end (Saturday four days before the Wednesday). Share = unpublished cells / cells '
        'that exist at all (visible at the cutoff or present in the final truth). A '
        'missing archive and a not-yet-reported week both count. The default `asof_weeks=2` uses j = 0 '
        'and 1 only; larger `asof_weeks` use more columns.', '',
        '### Headline: all seasons pooled, % unpublished', '',
        'The last season listed is the one in progress at the truth day (few issuances).', '',
        _markdown(headline.reset_index(), 1), '',
        _heatmaps(tables['states+US'], '% of target cells unpublished at the cutoff (states + DC + US)'), '',
        _heatmaps(tables['US'], '% of target cells unpublished at the cutoff (US only)'), '',
        '### Whole-issuance gaps', '',
        'Issuances (per target, season of the context end) in which no j <= 1 cell had an as-of value, '
        'i.e. the whole default as-of window was unavailable, over all issuances with such a cell.', '',
        _markdown(_whole_issuance(cells).reset_index(), 0), '',
        '### Per issuance (states+US, j = 0..3)', '',
        _per_issuance(cells), '',
        '### Table, states+US, % unpublished by j', '',
        _markdown(tables['states+US'].reset_index()), '',
        '### Table, US only, % unpublished by j', '',
        _markdown(tables['US'].reset_index()), '',
        '## 2. Revision where an as-of value was visible', '',
        '(as-of - final) / final in %, on cells with a visible as-of value and final > 0, US and states '
        'pooled, all issuances.', '',
        revision_figure, '',
        _markdown(pd.concat([revision_table[['count']], revision_table[['p10', 'median', 'p90']] * 100], axis=1)
                  .reset_index(), 1), '',
        '## 3. Covariates: final value present but not visible at the cutoff', '',
        'Covariate cells not visible at the cutoff are unavailable (as target cells now are). % of cells with a '
        f'final value, all locations and issuances. Kinsa (US only) has as-of data only from '
        f'{KINSA_ASOF_START}; its row restricted to issuances from then on is shown separately.', '',
        covariate_figure, '',
        covariate_line, '',
        _markdown(covariate_table.reset_index().rename(columns={'index': 'covariate'}), 1), '',
        '## 4. Covariates in ' + ' and '.join(PLACES), '',
        'Final (truth-day) values, each divided by its own maximum at that location over the calendar. '
        'Colour = pathogen, line style = source (solid inpatient claims, dashed outpatient claims, dotted '
        'NWSS wval_like, dash-dot NWSS pct_rank; Kinsa purple; ILINet ILI brown; long dashes clinical-lab '
        'influenza percent positive; dash-dot-dot FluSurv-NET rate). Faint dots: the context-end week of each '
        'Wednesday issuance as visible at its cutoff, same scale (the latest as-of value an episode sees). '
        'A covariate with no value at a location is listed in the legend as not available.', '',
        _covariate_lines(panel), '',
        '## 5. Wastewater indices (NWSS)', '',
        'The two indices per pathogen, `wval_like` and `pct_rank`, from the '
        '`derived_nwss_state_indices` snapshot (rebuilt by `python -m chromantis.dataset.build nwss-indices`; '
        'definitions in [wastewater](sources.md#wastewater)). Lines: final (truth-day) value per week; dots: the '
        'context-end week as visible at each Wednesday cutoff (NaN before the first Delphi archive vintage, '
        f'2026-02-25). Weeks with any value: {ww_first} to {ww_last}; plotted range restricted to it. States '
        'need at least 3 eligible sites.', '',
        *[x for figure in ww_figures for x in (figure, '')],
        '### Coverage', '',
        'Number of states + DC with a value per week (US excluded): final truth (line) and the '
        'context-end week as of each Wednesday (dots). `issuances_with_asof` = Wednesdays whose '
        'context-end week had a value for at least one state.', '',
        ww_coverage, '',
        _markdown(ww_table, 1), '',
    ]
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text('\n'.join(lines))
    return dict(output=output, sha256=sha, episodes=len(eps), unpublished_pooled=headline.round(1).to_dict())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dataset', default=PANEL_DATASET)
    parser.add_argument('--output', default=OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.dataset, args.output), indent=1, default=str))


if __name__ == '__main__':
    main()
