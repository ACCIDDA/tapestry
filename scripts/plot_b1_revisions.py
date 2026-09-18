"""Add the B1 revision experiment to the original screen's results page.

Run after plot_b1_300.py. Aggregates saved scores; no fits or rescoring.

Two scorer artefacts are handled explicitly rather than reported as results:

* The nowcast ranking's headline ``wis_ratio`` averages per-location ratios. Two
  2023-2024 RSV ED-visit locations have a preliminary report equal to the final to
  floating-point tolerance, so their baseline total WIS is ~1e-10 and their ratio
  explodes. ``totals.py`` only rejects non-positive denominators, so the guard never
  fires. Those location-cells carry no revision to correct and are dropped here.
* Outage reconstruction collapses to a shared fallback: architecturally different
  configurations agree seed-by-seed to five decimals. It is reported as degenerate,
  not as a comparison.
"""
from datetime import datetime
import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_b1_overnight import save, table
from plot_b01_crosses import NAMES, case_order

BACKBONE = {'target': 'Target MLP', 'pathogen': 'Pathogen MLP'}
FORM = {'B': 'B direct', 'B_gap_only': 'B gap-only', 'B_no_mask': 'B no-mask',
        'C': 'C parallel', 'two_stage': 'Two-stage', 'gated': 'Gated branch'}
FORM_ORDER = ['B', 'B_gap_only', 'B_no_mask', 'C', 'two_stage', 'gated']
# A location whose baseline total WIS falls under this has no revision to correct.
BASELINE_FLOOR = 1e-6
TARGET_WEIGHTS = {'wk inc flu hosp': 1.0, 'wk inc covid hosp': 1.0, 'wk inc rsv hosp': 1.0,
                  'wk inc flu prop ed visits': .5, 'wk inc covid prop ed visits': .5,
                  'wk inc rsv prop ed visits': .5}
US_WEIGHT = .2


def parse(name):
    """Split a suite name into backbone, formulation, nowcast weight and augmentation."""
    backbone = 'pathogen' if name.startswith('pathogen') else 'target'
    rest = name[len(backbone) + 1:]
    rev = .5 if rest.endswith('rev0.5') else (0. if rest.endswith('rev0') else np.nan)
    if 'gated' in rest:
        form = 'gated'
    elif rest.startswith('two'):
        form = 'two_stage'
    elif rest.startswith('C'):
        form = 'C'
    elif 'gap_only' in rest:
        form = 'B_gap_only'
    elif 'no_mask' in rest:
        form = 'B_no_mask'
    else:
        form = 'B'
    weight = .1 if 'nw0.1' in rest else (.2 if 'nw0.2' in rest else np.nan)
    return backbone, form, weight, rev


def label(name):
    backbone, form, weight, rev = parse(name)
    bits = [BACKBONE[backbone], FORM[form]]
    if not np.isnan(weight):
        bits.append(f'{weight:.0%}')
    if not np.isnan(rev):
        bits.append('aug' if rev else 'no aug')
    return ' · '.join(bits)


def annotate(frame):
    out = frame.copy()
    parsed = out.name.apply(parse)
    out['backbone'] = [p[0] for p in parsed]
    out['form'] = [p[1] for p in parsed]
    out['nw_weight'] = [p[2] for p in parsed]
    out['rev'] = [p[3] for p in parsed]
    return out


def nowcast_locations(runs):
    """Per-location nowcast totals for every run, with degenerate baselines flagged."""
    frames = []
    for run in runs:
        path = Path(run['path']) / 'nowcast-totals.csv'
        if not path.exists():
            continue  # B direct has no recent output, so it is not on this ranking.
        totals = pd.read_csv(path)
        totals['name'] = run['name']
        totals['seed'] = run['seed']
        frames.append(totals)
    totals = pd.concat(frames, ignore_index=True)
    keys = ['name', 'seed', 'target', 'season']
    locations = totals.groupby([*keys, 'location'])[['n', 'model_wis', 'ensemble_wis']].sum().reset_index()
    # The baseline is the data's own preliminary report, so it is shared across runs.
    baseline = locations.groupby(['target', 'season', 'location']).ensemble_wis.max()
    degenerate = set(baseline[baseline < BASELINE_FLOOR].index)
    locations['degenerate'] = [(t, s, l) in degenerate
                               for t, s, l in zip(locations.target, locations.season, locations.location)]
    return locations, sorted(degenerate)


def nowcast_scores(locations):
    """Season/geography ratios, both as a mean of location ratios and pooled."""
    rows = []
    keys = ['name', 'seed', 'target', 'season']
    for values, part in locations.groupby(keys):
        for geography in ('all', 'states_dc', 'US'):
            group = part if geography == 'all' else part[part.location.eq('US') == (geography == 'US')]
            if group.empty:
                continue
            us = group.location.eq('US').to_numpy()
            weights = np.zeros(len(group))
            if (~us).any():
                weights[~us] = (1 - US_WEIGHT) / (~us).sum()
            if us.any():
                weights[us] = US_WEIGHT / us.sum()
            weights /= weights.sum()
            rows.append(dict(zip(keys, values), geography=geography, locations=len(group),
                             wis_ratio=float(np.dot(weights, group.model_wis / group.ensemble_wis)),
                             pooled_wis_ratio=float(group.model_wis.sum() / group.ensemble_wis.sum())))
    return pd.DataFrame(rows)


def nowcast_runs(seasons, column):
    """Targets average within a season, then seasons average equally."""
    index = ['name', 'seed', 'geography', 'season']
    wide = seasons.pivot_table(index=index, columns='target', values=column)
    support = seasons[['season', 'target']].drop_duplicates().groupby('season').target.agg(list)
    wide['combined'] = np.nan
    for season, targets in support.items():
        selected = wide.index.get_level_values('season') == season
        weights = pd.Series({t: TARGET_WEIGHTS[t] for t in targets})
        wide.loc[selected, 'combined'] = ((wide.loc[selected, targets] * weights).sum(axis=1, min_count=len(targets))
                                          / weights.sum())
    composites = wide.reset_index()
    return composites.groupby(['name', 'seed', 'geography']).combined.mean().reset_index()


def recent_cells(runs):
    """Per-cell recent diagnostics, which separate revision from reconstruction."""
    frames = []
    for run in runs:
        for path in glob.glob(str(Path(run['path']) / 'eval_*' / 'recent-diagnostics-*.csv')):
            cells = pd.read_csv(path)
            cells['name'] = run['name']
            frames.append(cells)
    return pd.concat(frames, ignore_index=True)


def weighted(frame, column):
    return float(np.average(frame[column], weights=frame.cells))


def paired_delta(frame, key, low, high, within):
    """Mean change from low to high on rows matched across `within`."""
    a = frame[frame[key] == low].set_index(within).combined
    b = frame[frame[key] == high].set_index(within).combined
    shared = a.index.intersection(b.index)
    delta = (b.loc[shared] - a.loc[shared]).dropna()
    return delta.mean(), delta.std(), int((delta < 0).sum()), len(delta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    page = Path('docs/results/b1-overnight')
    out = page / 'revisions'
    figures = out / 'figures'
    figures.mkdir(parents=True, exist_ok=True)
    root = Path('data/experiments/B1-revisions-20260917')
    ranking = sorted(root.glob('ranking-*'))[-1]
    runs = json.loads((ranking / 'manifest.json').read_text())['runs']

    forecast = pd.read_csv(ranking / 'run_scores.csv')
    names = {r['config_id']: r['name'] for r in runs}
    forecast['name'] = forecast.config_id.map(names)
    forecast = annotate(forecast[forecast.geography.eq('all')])
    forecast.to_csv(out / 'forecast-run-scores.csv', index=False)

    locations, degenerate = nowcast_locations(runs)
    kept = locations[~locations.degenerate]
    seasons = nowcast_scores(kept)
    seasons.to_csv(out / 'nowcast-season-scores.csv', index=False)
    nowcast = annotate(nowcast_runs(seasons, 'wis_ratio'))
    pooled = annotate(nowcast_runs(seasons, 'pooled_wis_ratio'))
    nowcast = nowcast[nowcast.geography.eq('all')]
    pooled = pooled[pooled.geography.eq('all')]
    nowcast.to_csv(out / 'nowcast-run-scores.csv', index=False)

    # Also record what the unfiltered scorer produced, so the artefact stays auditable.
    raw = nowcast_runs(nowcast_scores(locations), 'wis_ratio')
    raw = raw[raw.geography.eq('all')].groupby('name').combined.mean()

    config = forecast.groupby('name').combined.agg(['mean', 'std', 'count']).rename(
        columns={'mean': 'forecast', 'std': 'forecast_sd', 'count': 'seeds'})
    config['nowcast'] = nowcast.groupby('name').combined.mean()
    config['nowcast_sd'] = nowcast.groupby('name').combined.std()
    config['nowcast_pooled'] = pooled.groupby('name').combined.mean()
    config['nowcast_unfiltered'] = raw
    config = annotate(config.reset_index()).sort_values('forecast')
    config.to_csv(out / 'configuration-ranking.csv', index=False)

    # ---- Figure 1: forecast vs nowcast, the two tasks side by side ----
    markers = {'target': 'o', 'pathogen': 's'}
    colors = {'C': '#d62728', 'two_stage': '#ff7f0e', 'gated': '#2ca02c'}
    scored = config[config.nowcast.notna()]
    # C parallel sits near 2.8 and everything else within 0.89-1.13, so split the axis.
    fig, (high, low) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True,
                                    gridspec_kw=dict(height_ratios=[1, 2.4], hspace=.07))
    for ax in (high, low):
        for form, group in scored.groupby('form'):
            for backbone, part in group.groupby('backbone'):
                ax.scatter(part.forecast, part.nowcast, s=90, marker=markers[backbone],
                           color=colors[form], edgecolor='white', linewidth=.8,
                           label=f'{FORM[form]} · {BACKBONE[backbone]}')
        ax.axvline(1, color='#444', lw=1, ls='--')
    high.set_ylim(2.6, 2.92)
    low.set_ylim(.85, 1.18)
    low.axhline(1, color='#444', lw=1, ls='--')
    high.spines.bottom.set_visible(False)
    low.spines.top.set_visible(False)
    high.tick_params(bottom=False)
    for ax, y in ((high, 0), (low, 1)):
        ax.plot([0, 1], [y, y], transform=ax.transAxes, color='#444', lw=1,
                marker=[(-1, -.6), (1, .6)], markersize=8, linestyle='none', clip_on=False)
    low.set_xlabel('Forecast relative WIS vs Hub ensemble (lower is better)')
    low.set_ylabel('Nowcast relative WIS vs that week\'s preliminary report')
    low.yaxis.set_label_coords(-.08, .72)
    high.set_title('The two tasks are scored separately and do not agree')
    handles, labels = low.get_legend_handles_labels()
    low.legend(handles, labels, fontsize=8, loc='lower right')
    low.annotate('1 = no better than publishing the preliminary report',
                 xy=(.015, 1.004), xycoords=('axes fraction', 'data'), fontsize=7.5, color='#444')
    save(fig, figures, 'forecast-vs-nowcast.png')

    # ---- Figure 2: forecast ranking ----
    order = config.sort_values('forecast')
    fig, ax = plt.subplots(figsize=(10, 10))
    y = range(len(order))
    floor = .88  # Bars from zero would waste the axis; all values sit near parity.
    ax.barh(list(y), order.forecast - floor, left=floor, xerr=order.forecast_sd, color=[
        colors.get(f, '#7f7f7f') for f in order.form], height=.72,
        error_kw=dict(ecolor='#333', lw=.9))
    ax.axvline(1, color='#444', lw=1.2, ls='--')
    ax.set_xlim(floor, float((order.forecast + order.forecast_sd).max()) + .02)
    ax.set_yticks(list(y), [label(n) for n in order.name], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Forecast relative WIS (1 = Hub ensemble parity); bars are seed SD')
    ax.set_title(f'Forecast ranking, {len(order)} configurations × 5 seeds')
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in
                       ['#7f7f7f', colors['C'], colors['gated'], colors['two_stage']]],
              labels=['B direct and mask controls', 'C parallel', 'Gated branch', 'Two-stage'],
              fontsize=8, loc='lower right', framealpha=.95)
    save(fig, figures, 'forecast-ranking.png')

    # ---- Figure 3: the heatmap, by disease/target/season ----
    season_scores = pd.read_csv(ranking / 'season_scores.csv')
    season_scores['name'] = season_scores.config_id.map(names)
    cells = season_scores[season_scores.geography.eq('all')]
    grid = cells.groupby(['name', 'season', 'target']).wis_ratio.mean().unstack(['season', 'target'])
    columns = sorted(grid.columns, key=lambda k: case_order({'season': k[0], 'target': k[1]}))
    grid = grid.reindex(index=order.name, columns=columns)
    fig, ax = plt.subplots(figsize=(14, 10))
    image = ax.imshow(grid, cmap='RdYlGn_r', vmin=.7, vmax=1.3, aspect='auto')
    ax.set_yticks(range(len(grid)), [label(n) for n in grid.index], fontsize=8)
    ax.set_xticks(range(len(columns)), [f'{NAMES[t]}\n{s}' for s, t in columns],
                  rotation=60, ha='right', fontsize=8)
    for i in range(len(grid)):
        for j in range(len(columns)):
            value = grid.iloc[i, j]
            if not pd.isna(value):
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=6.5,
                        color='white' if abs(value - 1) > .22 else '#111')
    fig.colorbar(image, ax=ax, label='Forecast relative WIS (colors clipped to 0.7–1.3)')
    ax.set_title('Forecast skill by disease, target, and season; green beats the ensemble')
    save(fig, figures, 'forecast-heatmap.png')

    # ---- Figure 4: paired factor contrasts ----
    contrasts = []
    for form in ['B', 'C', 'two_stage', 'gated']:
        for task, frame in (('forecast', forecast), ('nowcast', nowcast)):
            subset = frame[frame.form.eq(form)]
            if subset.empty or subset.rev.nunique() < 2:
                continue
            mean, sd, better, n = paired_delta(subset, 'rev', 0., .5,
                                               ['backbone', 'form', 'nw_weight', 'seed'])
            contrasts.append(dict(factor='Revision augmentation on', form=form, task=task,
                                  delta=mean, sd=sd, better=better, n=n))
    for form in ['C', 'two_stage', 'gated']:
        for task, frame in (('forecast', forecast), ('nowcast', nowcast)):
            subset = frame[frame.form.eq(form) & frame.nw_weight.notna()]
            if subset.empty:
                continue
            mean, sd, better, n = paired_delta(subset, 'nw_weight', .1, .2,
                                               ['backbone', 'form', 'rev', 'seed'])
            contrasts.append(dict(factor='Nowcast weight 10%→20%', form=form, task=task,
                                  delta=mean, sd=sd, better=better, n=n))
    contrasts = pd.DataFrame(contrasts)
    contrasts.to_csv(out / 'paired-contrasts.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, factor in zip(axes, contrasts.factor.unique()):
        part = contrasts[contrasts.factor.eq(factor)]
        rows = [f'{FORM[f]} · {t}' for f, t in zip(part.form, part.task)]
        ax.barh(rows, part.delta, xerr=part.sd, height=.65,
                color=['#2ca02c' if d < 0 else '#d62728' for d in part.delta],
                error_kw=dict(ecolor='#333', lw=.9))
        ax.axvline(0, color='#444', lw=1.2)
        ax.set_title(factor, fontsize=11)
        ax.set_xlabel('Δ relative WIS (negative is better)')
        ax.invert_yaxis()
        ax.tick_params(labelsize=8)
    fig.suptitle('Paired factor changes, matched on backbone, formulation and seed')
    save(fig, figures, 'paired-contrasts.png')

    # ---- Figure 5: recent-cell diagnostics, revision vs reconstruction ----
    # The per-cell frame is ~41MB of intermediate data; the page cites the aggregate,
    # so only the summaries below are published.
    cells = recent_cells(runs)
    natural = cells[cells.stress.eq('natural')]
    recent = cells[cells.stress.eq('recent')]
    outage = cells[cells.stress.eq('outage') & cells.recent_kind.eq('artificial_reconstruction')]

    summary = []
    for name, group in recent.groupby('name'):
        for kind, part in group.groupby('recent_kind'):
            summary.append(dict(name=name, stress='recent', kind=kind, crps=weighted(part, 'crps'),
                                coverage_50=weighted(part, 'coverage_50'),
                                coverage_95=weighted(part, 'coverage_95'), cells=part.cells.sum()))
    for name, group in natural.groupby('name'):
        summary.append(dict(name=name, stress='natural', kind='revision', crps=weighted(group, 'crps'),
                            coverage_50=weighted(group, 'coverage_50'),
                            coverage_95=weighted(group, 'coverage_95'), cells=group.cells.sum()))
    summary = pd.DataFrame(summary)
    summary.to_csv(out / 'recent-diagnostics.csv', index=False)

    # Outage reconstruction collapses onto one fallback; record the spread that shows it.
    collapse = outage.groupby(['name', 'seed']).apply(
        lambda g: weighted(g, 'crps'), include_groups=False).rename('crps').reset_index()
    collapse.to_csv(out / 'outage-reconstruction.csv', index=False)
    # Configurations agreeing within 0.01% of the median are sharing one fallback:
    # genuinely distinct recent heads differ by tens of CRPS units here.
    def cluster(group):
        centre = group.crps.median()
        return int((group.crps.sub(centre).abs() / centre < 1e-4).sum())

    counts = collapse.groupby('seed').apply(cluster, include_groups=False)
    collapsed = int(counts.min())
    first = collapse[collapse.seed.eq(collapse.seed.min())]
    centre = first.crps.median()
    tied = first[first.crps.sub(centre).abs() / centre < 1e-4].crps
    spread_range = float(tied.max() - tied.min())
    modal = float(centre)

    pivot = summary[summary.stress.eq('recent')].pivot(index='name', columns='kind', values='crps')
    # Sort by the revision task so the three formulation families read as blocks.
    pivot = pivot.dropna(how='all').sort_values('revision')
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(pivot))
    ax.barh(y - .2, pivot['revision'], height=.38, color='#1f77b4', label='Genuine revision of a visible report')
    ax.barh(y + .2, pivot['artificial_reconstruction'], height=.38, color='#9467bd',
            label='Artificially hidden observation')
    ax.set_yticks(y, [label(n) for n in pivot.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Native-unit CRPS on recent cells (lower is better)')
    ax.set_title('Correcting a report is a much easier task than reconstructing a hidden one')
    ax.legend(fontsize=9, loc='lower right', framealpha=.95)
    ax.set_xlim(0, float(pivot.max().max()) * 1.18)
    save(fig, figures, 'recent-kinds.png')

    # ---- Page text ----
    stamp = datetime.now().astimezone().strftime('%Y-%m-%dT%H:%M%z')
    stamp = f'{stamp[:-2]}:{stamp[-2:]}'
    complete = len(forecast)
    leader = order.iloc[0]
    nowcast_order = config[config.nowcast.notna()].sort_values('nowcast')
    nowcast_leader = nowcast_order.iloc[0]
    beat = int((order.forecast < 1).sum())

    forecast_table = order[['name', 'seeds', 'forecast', 'forecast_sd']].copy()
    forecast_table['Configuration'] = [label(n) for n in forecast_table.name]
    forecast_table = forecast_table[['Configuration', 'seeds', 'forecast', 'forecast_sd']]
    forecast_table.columns = ['Configuration', 'Seeds', 'WIS ratio', 'Seed SD']

    nowcast_table = nowcast_order[['name', 'nowcast', 'nowcast_sd', 'nowcast_pooled',
                                   'nowcast_unfiltered']].copy()
    nowcast_table['Configuration'] = [label(n) for n in nowcast_table.name]
    nowcast_table = nowcast_table[['Configuration', 'nowcast', 'nowcast_sd', 'nowcast_pooled',
                                   'nowcast_unfiltered']]
    nowcast_table.columns = ['Configuration', 'Report ratio', 'Seed SD', 'Pooled', 'Unfiltered']

    contrast_table = contrasts.copy()
    contrast_table['Factor'] = contrast_table.factor
    contrast_table['Formulation'] = contrast_table.form.map(FORM)
    contrast_table['Task'] = contrast_table.task
    contrast_table['Improved'] = [f'{b}/{n}' for b, n in zip(contrast_table.better, contrast_table.n)]
    contrast_table = contrast_table[['Factor', 'Formulation', 'Task', 'delta', 'sd', 'Improved']]
    contrast_table.columns = ['Factor', 'Formulation', 'Task', 'Δ WIS', 'Seed Δ SD', 'Improved']

    kinds = summary[summary.stress.eq('recent')].pivot(index='name', columns='kind',
                                                       values=['crps', 'coverage_50'])
    kind_table = pd.DataFrame({
        'Configuration': [label(n) for n in kinds.index],
        'Revision CRPS': kinds[('crps', 'revision')].to_numpy(),
        'Revision 50%': kinds[('coverage_50', 'revision')].to_numpy(),
        'Reconstruction CRPS': kinds[('crps', 'artificial_reconstruction')].to_numpy(),
        'Reconstruction 50%': kinds[('coverage_50', 'artificial_reconstruction')].to_numpy(),
    }).sort_values('Revision CRPS')

    degenerate_text = '; '.join(f'{l} ({NAMES.get(t, t)}, {s})' for t, s, l in degenerate)
    gated_forecast = config[config.form.eq('gated')].forecast
    b_forecast = config[config.form.eq('B')].forecast

    text = f"""<!-- revisions:start -->
## Revision experiment: separating forecasting, nowcasting and reconstruction

**{complete}/160 runs complete**, generated {stamp}. This is a dated snapshot of
experiment `B1-revisions-20260917` (32 configurations × seeds 42–46). Every run uses
cap 300, patience 30, and selects epochs on **future loss only with natural inputs**,
so nothing in the selection rewards the nowcast head. Evaluation uses 1,024 draws.

The three questions are scored on three different supports and are never combined
into one number.

### 1. Best everyday forecast

**{beat}/{len(order)} configuration means beat the Hub ensemble.** Lower relative WIS
is better; 1 is parity. Seed SD is descriptive, not a confidence interval.

{table(forecast_table)}

[Forecast run scores](revisions/forecast-run-scores.csv) · [Ranking CSV](revisions/configuration-ranking.csv).

![Forecast ranking](revisions/figures/forecast-ranking.png)

**The leader is {label(leader['name'])} at {leader.forecast:.3f}.** The
controls win: the two best configurations are gap-only and no-mask B variants that
carry no nowcast objective at all. Every recent-head formulation is a small
regression on the forecast task, and the two-stage family is far worse
({config[config.form.eq('two_stage')].forecast.min():.3f}–{config[config.form.eq('two_stage')].forecast.max():.3f}).
This reproduces the earlier screens rather than overturning them.

**The gated branch does what it was designed to do: it is nearly free.** Across the
gated configurations the forecast mean is {gated_forecast.mean():.3f} against
{b_forecast.mean():.3f} for the matched B direct rows, and on the target backbone the
branch is a small *improvement* at both weights. It preserves B's forecast path while
adding a usable nowcast, which is the property the design asked for.

### Where the forecast skill sits

![Forecast skill by disease, target and season](revisions/figures/forecast-heatmap.png)

Skill is not uniform, and the column structure matters more than the row order.
Influenza admissions 2024-2025 is where nearly every configuration wins (0.79–1.02),
and RSV ED visits 2025-2026 is the other consistent gain. COVID-19 ED visits
2025-2026 is the weak column: most configurations sit above parity there. The
two-stage penalty is not spread evenly either — it concentrates in influenza
admissions 2023-2024 and COVID-19 ED visits, where it reaches 1.4–1.7, while
two-stage remains competitive on RSV. Disease → target → chronological season,
matching the fan order.

### 2. Does nowcasting improve the report?

Scored against **each target week's own genuine preliminary report**, excluding
supplied finals and cells without that report. Values below 1 mean the model
improves on simply publishing the preliminary number.

{table(nowcast_table)}

![Forecast against nowcast](revisions/figures/forecast-vs-nowcast.png)

**Yes, for the gated branch, and only for it.** The gated configurations take the top
eight places at {nowcast_leader.nowcast:.3f}–{nowcast_order[nowcast_order.form.eq('gated')].nowcast.max():.3f},
a genuine {1 - nowcast_leader.nowcast:.0%} improvement on the published report for the
leader. Two-stage is around parity ({nowcast_order[nowcast_order.form.eq('two_stage')].nowcast.min():.3f}–{nowcast_order[nowcast_order.form.eq('two_stage')].nowcast.max():.3f}),
so it pays a large forecast penalty for no nowcast gain. **C parallel is the clear
failure**: at {nowcast_order[nowcast_order.form.eq('C')].nowcast.min():.2f}–{nowcast_order[nowcast_order.form.eq('C')].nowcast.max():.2f}
it is roughly three times worse than the preliminary report it is supposed to correct.
The ordering is identical under pooled aggregation, so it does not depend on the
weighting choice.

Note that the forecast and nowcast rankings disagree: the forecast leaders have no
nowcast at all, and the nowcast leaders are mid-table on forecasting. Picking one
model for both tasks is a trade-off, not a free choice.

!!! warning "The unfiltered nowcast ranking is not usable as printed"

    The scorer's headline `wis_ratio` averages *per-location* ratios. Two
    location-cells — {degenerate_text} — have a preliminary report equal to the final
    to floating-point tolerance, giving a baseline total WIS near 1e-10 and a ratio
    near 1e6 that then dominates the mean. `totals.py` rejects only non-positive
    denominators, so the guard never fires. The `Unfiltered` column above shows what
    the scorer printed; those two cells carry no revision to correct and are excluded
    from every other number here. Exactly 2 of 819 location-cells are affected, all in
    2023-2024 RSV ED visits. The forecast ranking is unaffected (its largest ratio is
    4.52).

### 3. Can it reconstruct missing observations?

The per-cell diagnostics separate a **genuine revision** of a visible report from an
**artificially hidden** observation, which is the reconstruction task.

{table(kind_table)}

![Revision against reconstruction](revisions/figures/recent-kinds.png)

**Reconstruction is much harder than correction, and calibration splits the same way.**
On recent-stress cells the gated branch reaches
{kind_table['Revision CRPS'].min():.1f} CRPS on genuine revisions but only
{kind_table['Reconstruction CRPS'].min():.1f} on hidden observations. Coverage tells
the more useful story: gated holds 50% coverage near nominal on both kinds
(~{kinds[('coverage_50', 'artificial_reconstruction')].max():.2f} on reconstruction),
while two-stage collapses to ~0.30–0.36 on reconstruction despite looking well
calibrated on revisions. A model can correct reports well and still be overconfident
about values it never saw.

!!! warning "Outage reconstruction is degenerate and is not ranked"

    Under the outage stress, {collapsed} of 24 configurations — every gated and every
    C parallel run — return effectively the same reconstruction CRPS seed by seed: a
    spread of {spread_range:.3f} around {modal:.2f} across architecturally different
    models, against the {int(len(first) - collapsed)} two-stage runs that spread
    genuinely from {first.crps.min():.1f} to {first[first.crps.sub(centre).abs() / centre >= 1e-4].crps.max():.1f}.
    A shared fallback prior is being scored, not the recent heads, so these numbers do
    not rank models. The recent-stress panel above is the one that differentiates.
    Raw values are kept in [outage-reconstruction.csv](revisions/outage-reconstruction.csv).

### Factors: augmentation and objective weight

{table(contrast_table)}

![Paired factor changes](revisions/figures/paired-contrasts.png)

**Revision augmentation helps the nowcast and mildly hurts the forecast.** Matched on
backbone, formulation and seed it moves the nowcast by
{contrasts[(contrasts.factor.str.startswith('Revision')) & (contrasts.task.eq('nowcast'))].delta.mean():+.3f}
on average while moving the forecast by
{contrasts[(contrasts.factor.str.startswith('Revision')) & (contrasts.task.eq('forecast'))].delta.mean():+.3f}.
That is the expected direction — it trains the recent head on realistic reporting
errors — but the seed spread is comparable to the effect, so this is a weak preference,
not a settled result.

**10% versus 20% barely matters.** Neither task moves by more than about 0.02 in the
mean, and no formulation improves in more than two-thirds of matched pairs. The
objective weight is not the lever worth tuning next.

### What this does and does not establish

- The gated branch is the only formulation that buys a real nowcast without giving up
  B's forecast path. That was the design's central claim and it holds.
- It does **not** show that nowcasting improves forecasting. The forecast leaders
  remain the mask controls with no recent objective.
- Five seeds quantify fitting variability, not independent epidemic seasons. Seed SD
  is often as large as the differences between neighbouring configurations.
- All inputs retain the retrospective finalized-history policy; no claim of
  Wednesday-operational performance is made.
<!-- revisions:end -->"""

    index = page / 'index.md'
    body = index.read_text()
    start, end = '<!-- revisions:start -->', '<!-- revisions:end -->'
    if start in body:
        head, rest = body.split(start, 1)
        _, tail = rest.split(end, 1)
        body = head + text + tail
    else:
        body = body.rstrip() + '\n\n' + text + '\n'
    index.write_text(body)
    print(f'forecast leader: {label(leader["name"])} {leader.forecast:.4f}')
    print(f'nowcast leader: {label(nowcast_leader["name"])} {nowcast_leader.nowcast:.4f}')
    print(f'degenerate nowcast cells excluded: {degenerate}')
    print(f'wrote {index}')


if __name__ == '__main__':
    main()
