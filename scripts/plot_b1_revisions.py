"""Add the B1 revision experiment to the original screen's results page.

Run after plot_b1_300.py. Aggregates saved scores; no fits or rescoring.

The adjusted nowcast ratio excludes two near-zero baseline groups as an explicit
post-hoc sensitivity analysis, not a replacement for the registered scorer.
Reconstruction diagnostics use training-scale-normalized errors and matched cells.
Near-identical outage scores are model-behavior evidence, not a scorer artefact.
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
# Explicit post-hoc cutoff in native summed WIS units; inspect threshold sensitivity.
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
    candidates = [p for p in root.glob('ranking-*') if len(json.loads((p / 'manifest.json').read_text())['runs']) == 160]
    if len(candidates) != 1:
        raise ValueError('Expected one complete 160-run ranking; select explicitly if ambiguous')
    ranking = candidates[0]
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
    low.set_ylabel('Adjusted nowcast WIS ratio (two near-zero baseline groups excluded)')
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
    fig.suptitle('Paired factor changes; error bars are SD of differences, not confidence intervals')
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

    # Retain the old raw aggregate for audit; similarity does not identify a mechanism.
    collapse = outage.groupby(['name', 'seed']).apply(
        lambda g: weighted(g, 'crps'), include_groups=False).rename('crps').reset_index()
    collapse.to_csv(out / 'outage-reconstruction.csv', index=False)
    # Updated scientific diagnostics are aggregated from existing per-cell scores.
    # Run analysis/b1-revisions-review/audit.py before regenerating this page.
    scaled = pd.read_csv(out / 'recent-scaled-run-diagnostics.csv')
    matched = pd.read_csv(out / 'matched-reconstruction.csv')
    for frame in (scaled, matched):
        frame['form'] = frame.name.map(lambda n: parse(n)[1])
    metrics = ['scaled_crps', 'scaled_report_ae', 'coverage_50', 'coverage_95']
    natural_summary = scaled[(scaled.stress == 'natural') & (scaled.recent_kind == 'revision')].groupby(
        ['form', 'horizon'])[metrics].mean().reset_index()
    natural_summary['relative_crps'] = natural_summary.scaled_crps / natural_summary.scaled_report_ae
    natural_summary.to_csv(out / 'natural-revision-family-summary.csv', index=False)
    matched_summary = matched.groupby(['form', 'stress', 'horizon', 'condition'])[
        ['scaled_crps', 'coverage_50', 'coverage_95']].mean().reset_index()
    matched_summary.to_csv(out / 'matched-reconstruction-family-summary.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, age in zip(axes, [-2, -1]):
        part = matched_summary[(matched_summary.stress == 'recent') & (matched_summary.horizon == age)]
        pivot = part.pivot(index='form', columns='condition', values='scaled_crps').reindex(['C','two_stage','gated'])
        x = np.arange(len(pivot))
        ax.bar(x-.18, pivot.visible, width=.36, label='Report visible')
        ax.bar(x+.18, pivot.hidden, width=.36, label='Same report hidden')
        ax.set_xticks(x, [FORM[f] for f in pivot.index]); ax.set_title('11-day-old week' if age == -2 else '4-day-old week')
        ax.set_ylabel('CRPS / training-only native Q95 scale')
    axes[0].legend(fontsize=8)
    fig.suptitle('Effect of hiding the same genuine reports; scientific weights, family means')
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
    nowcast_table.columns = ['Configuration', 'Adjusted location ratio', 'Seed SD', 'Pooled sensitivity', 'Registered unfiltered']

    contrast_table = contrasts.copy()
    contrast_table['Factor'] = contrast_table.factor
    contrast_table['Formulation'] = contrast_table.form.map(FORM)
    contrast_table['Task'] = contrast_table.task
    contrast_table['Improved'] = [f'{b}/{n}' for b, n in zip(contrast_table.better, contrast_table.n)]
    contrast_table = contrast_table[['Factor', 'Formulation', 'Task', 'delta', 'sd', 'Improved']]
    contrast_table.columns = ['Factor', 'Formulation', 'Task', 'Δ WIS', 'Seed Δ SD', 'Improved']

    natural_table = natural_summary.copy()
    natural_table['Formulation'] = natural_table.form.map(FORM)
    natural_table['Age (days)'] = natural_table.horizon.map({-2:11,-1:4})
    natural_table = natural_table[['Formulation','Age (days)','scaled_crps','scaled_report_ae','relative_crps','coverage_50','coverage_95']]
    natural_table.columns = ['Formulation','Age (days)','Scaled CRPS','Report error','Relative CRPS','50% coverage','95% coverage']
    kind_table = matched_summary[matched_summary.stress == 'recent'].copy()
    kind_table['Formulation'] = kind_table.form.map(FORM)
    kind_table['Age (days)'] = kind_table.horizon.map({-2:11,-1:4})
    kind_table = kind_table[['Formulation','Age (days)','condition','scaled_crps','coverage_50','coverage_95']]
    kind_table.columns = ['Formulation','Age (days)','Report condition','Scaled CRPS','50% coverage','95% coverage']
    outage_table = scaled[(scaled.stress == 'outage') & (scaled.recent_kind == 'artificial_reconstruction')].groupby(
        ['form','horizon'])[['scaled_crps','scaled_zero_crps','coverage_50','coverage_95']].mean().reset_index()
    outage_table['Formulation'] = outage_table.form.map(FORM)
    outage_table['Age (days)'] = outage_table.horizon.map({-2:11,-1:4})
    outage_table = outage_table[['Formulation','Age (days)','scaled_crps','scaled_zero_crps','coverage_50','coverage_95']]
    outage_table.columns = ['Formulation','Age (days)','Scaled CRPS','Predict-zero error','50% coverage','95% coverage']
    paired_b = pd.read_csv(out / 'gated-vs-matched-B.csv')
    paired_table = paired_b[['backbone','augmentation','nowcast_weight','B','gated','delta','seed_delta_sd','improved','pairs']].copy()
    paired_table.columns = ['Backbone','Augmentation','Recent weight','B forecast','Gated forecast','Delta','Seed delta SD','Improved','Pairs']

    degenerate_text = '; '.join(f'{l} ({NAMES.get(t, t)}, {s})' for t, s, l in degenerate)
    gated_forecast = config[config.form.eq('gated')].forecast
    b_forecast = config[config.form.eq('B')].forecast

    text = f"""<!-- revisions:start -->
## Revision experiment: separating forecasting, nowcasting and reconstruction

**{complete}/160 runs complete**, generated {stamp}. This is a dated snapshot of
experiment `B1-revisions-20260917` (32 configurations × seeds 42–46). Every run uses
cap 300, patience 30, and selects epochs on **future loss only with natural inputs**,
so recent accuracy is not an explicit selection criterion. Evaluation uses 1,024 draws.

The three questions are scored on three different supports and are never combined
into one number.

### 1. Best everyday forecast

**{beat}/{len(order)} configuration means beat the Hub ensemble.** Lower relative WIS
is better; 1 is parity. Seed SD is descriptive, not a confidence interval.

{table(forecast_table)}

[Forecast run scores](revisions/forecast-run-scores.csv) · [Ranking CSV](revisions/configuration-ranking.csv).

![Forecast ranking](revisions/figures/forecast-ranking.png)

**The leader is {label(leader['name'])} at {leader.forecast:.3f}.** The two
best configuration means are B gap-only and B no-mask controls. That does not mean
all recent-head models worsen forecasting: compare matched backbone, augmentation
and seeds, rather than each model against the winner selected from a different recipe.

{table(paired_table)}

Without augmentation, target gated-20% improves on matched B from **0.997 to 0.948**
(delta -0.050; all five seeds improve). Target gated-10% improves in four of five.
Pathogen gated variants without augmentation worsen on average and improve in only
one of five seeds each. The branch is **promising on the target backbone, not a
universally free addition**. The target gap-only control still has the best mean
at 0.938; we did not cross gated nowcasting with gap-only masking.

Two-stage remains worse than Hub in every configuration mean
({config[config.form.eq('two_stage')].forecast.min():.3f}–{config[config.form.eq('two_stage')].forecast.max():.3f}).
C's weak nowcast outputs do not prevent competitive forecast performance: an
auxiliary objective can help the shared representation without yielding the best
recent-head checkpoint, because selection uses future loss only.

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

**Gated is the strongest recent-head family under both displayed aggregations.**
The eight gated means are {nowcast_leader.nowcast:.3f}–{nowcast_order[nowcast_order.form.eq('gated')].nowcast.max():.3f}
under the location-relative metric **after the two-group exclusion below**. The best
point estimate is about 11% lower WIS on that adjusted metric; it is not a universal
11% reduction in count error or a proven improvement in every target/season.

Two-stage is 1.004–1.129 under that metric, but **0.691–0.737 under pooled scoring**.
It therefore improves on reports under one aggregation while failing to improve
under the other. C is worse under both (2.69–2.86 adjusted location-relative;
1.48–1.57 pooled). WIS ratios measure distributional score, not simply point error.

The **family ordering** gated < two-stage < C survives pooling; the exact
configuration ordering does not. Pooling sums numerator and denominator within
target/season before taking their ratio, thereby changing location importance and
losing the prescribed states/DC 80% versus US 20% weighting. It is a sensitivity
analysis, not a confirmation of the same estimand.

#### Reporting ages, accuracy and uncertainty

{table(natural_table)}

This additional full-support diagnostic divides each cell's CRPS and unchanged-report
absolute error by its model's training-only target/location Q95 scale, then averages
locations with states/DC 80% and US 20%, targets with admissions 1 and ED .5, and
seasons equally, keeping ages separate. It retains the two near-zero-baseline groups:
there is no division by their individual report errors. Relative CRPS is the ratio
of the displayed aggregate scores. It is **a different, explicitly labelled metric**,
not a replacement WIS ranking. Values shown are descriptive means across all eight
configurations and five seeds in each family; they are not independent replicates
or the performance of an ensemble. Target/season support can differ by reporting age.

The age split matters: C has relative scaled CRPS about **2.53 for the preceding
11-day-old week**, but **0.85 for the newest four-day-old week**. Calling C useless
at every revision task is therefore incorrect. Gated improves both ages (about
0.78 and 0.73), and two-stage also improves on this alternative metric (0.83 and
0.77). C's anchoring both recent outputs to the latest observation is a concrete
hypothesis for its older-week weakness, not a proven cause. **Gated is not fully
calibrated**: natural 95% coverage is only about 86% and 84%, respectively, despite
its competitive error scores. Newest-week 50% coverage is about 44%.

!!! warning "Near-zero report error makes the registered ratio unstable"

    The unfiltered registered scorer averages per-location ratios. Two
    target/season/location groups — {degenerate_text} — have report-versus-final
    errors near floating-point precision. Positive denominators near 1e-10 yield
    extremely large ratios. This is a fragile metric in the presence of a nearly
    perfect baseline, not evidence of a thousandfold error in predictions.

    The adjusted columns exclude groups with total baseline WIS below **1e-6 in
    native units**, an explicitly **post-hoc** threshold. This excludes 2 of 819
    target/season/location groups, not two individual forecast observations.
    They still matter for absolute error: a model should not damage an accurate
    report. Only the adjusted nowcast ratio columns exclude them; the new scaled
    diagnostics retain them. Forecast rankings are unchanged. See
    [small denominators](revisions/small-nowcast-denominators.csv) and
    [threshold sensitivity](revisions/nowcast-threshold-sensitivity.csv).
    Each group contains ten observations per run. Cutoffs from 1e-9 through 1e-5
    remove the same two groups and give identical adjusted scores; this numerical
    sensitivity result does not turn the post-hoc choice into a predefined endpoint.

    The prelaunch audit checked for exactly zero aggregate errors and found none.
    That statement was literally true but insufficient to check numerical stability;
    it did not rule out near-zero positive denominators. Neither deleting these
    groups nor changing to pooling should silently replace the predefined endpoint.

### 3. Can it reconstruct missing observations?

The per-cell diagnostics separate a **genuine revision** of a visible report from an
**artificially hidden** observation, which is the reconstruction task.

{table(kind_table)}

![Revision against reconstruction](revisions/figures/recent-kinds.png)

**Hiding an observation worsens recent estimation, but the old 7-versus-18
CRPS comparison was not a valid scientific aggregate.** It averaged admissions
counts and ED proportions in native units with cell-count weights, and compared
different sets of observations. The replacement table/figure pairs each genuinely
preliminary report hidden by the recent stress with its own natural-input result,
normalizes CRPS by the training-only Q95 scale, and applies the scientific weights.
Supplied finals are not in this paired comparison. Both 50% and 95% coverage are
shown by age; approximate overall 50% coverage alone does not establish calibration.

The table averages the eight configurations and five seeds within each family.
It is descriptive of these fitted recipes; paired masking identifies the effect
of this whole stress intervention, which can hide multiple channels/locations at once.

For gated, hiding the same report increases scaled CRPS from about 0.014 to 0.045
at 11 days, and 0.018 to 0.055 at four days. Reconstruction 95% coverage is about
89% and 86%, below nominal; two-stage is substantially lower, about 71% and 64%.
The near-nominal gated 50% coverage is encouraging, but is not full calibration.

Full-channel outage is a separate, more severe condition (all reconstruction
cells here, including hidden supplied finals; not the matched report-only subset):

{table(outage_table)}

!!! warning "Outage scores reveal a model failure mode, not a demonstrated scorer bug"

    The old native-unit aggregate places 16 C/gated configurations very close to
    84.11 for seed 42. Similar aggregate scores do not prove identical predictions
    or a bypassed recent head. The frozen code **does execute** C's recent heads
    and the gated model's separate missing-value heads under outage. When focal
    history is absent, both use B's small 0.01 transformed anchor, and decoded
    admission nowcasts remain near zero. Across all audited C/gated runs, every
    exported outage admission median is zero after count rounding, while ED
    predictions differ. That is model behavior worth reporting,
    not a reason to discard the condition. Exact causal attribution to anchoring
    requires an ablation; the near-zero behavior is consistent with this design.

    Count-dominated raw pooling can conceal differences in ED outputs. See
    [outage by target](revisions/outage-by-target.csv) for predicted medians and
    comparison to predicting zero, and [scaled diagnostics](revisions/recent-scaled-run-diagnostics.csv)
    for normalization and coverage. The old raw summary is retained only as an
    audit artifact, not as a ranking or evidence that the scorer substitutes a prior.

### Factors: augmentation and objective weight

{table(contrast_table)}

![Paired factor changes](revisions/figures/paired-contrasts.png)

**Augmentation produces a trade-off, not a uniform gain.** On the adjusted
nowcast metric it improves the mean in all three recent-head families. Its forecast
effect depends on formulation: B +0.021, C +0.008, gated **+0.041** (worse), two-stage
-0.012 (better). For gated models only 5/20 matched forecast contrasts improve.
Do not label the pooled +0.014 across families as a universal mild effect.
The natural revision error is not interchangeable with reconstruction error.

**The 10%→20% effect is formulation-dependent.** C's adjusted nowcast mean improves
by 0.055, two-stage by 0.021, while gated changes by +0.004. Forecast changes are
small on average (gated -0.017, C +0.005, two-stage +0.003). The prior statement
that neither task moves by more than 0.02 was incorrect. There is no universal
winner between the weights, but the screen does not establish that weight is irrelevant.

Error bars show **SD of paired differences**, pooled over several related settings;
they are not confidence intervals. Crossing zero is not a significance test. Five
seeds measure fitting variability on the same seasons, not uncertainty over future
epidemics. These tables provide effect sizes and direction counts, not a formal
claim of superiority or equivalence.

### Conclusions and next decisions

- For forecasting alone, retain **target B gap-only** and **pathogen B no-mask** as
  strong controls. Target gated without augmentation is a competitive joint-output
  candidate and improves matched mixed-mask B; it does not yet beat the best control.
- Forecasts and nowcasts need not come from the same fitted model. B forecasts
  plus gated recent estimates are a valid separate-output option; this does not
  establish a coherent joint trajectory distribution or a new combined score.
- For revision correction, gated is the most promising tested family. Its advantage
  over C/two-stage survives the two displayed aggregation choices, while the claim
  that *only* gated beats the report does not. The 11% headline is conditional on
  the post-hoc location-relative calculation; inspect age-specific scaled errors
  and coverage too.
- For reconstruction, use matched, scaled, age-specific diagnostics. Treat near-zero
  outage admissions as a real failure to address, rather than discarding the test.
- The next useful architecture control is a parallel recent head anchored to **each
  week's own report**, with a learned missing-history anchor. C currently anchors
  both recent outputs to the latest visible focal value, unlike the gated branch.
  This could explain part of its older-week weakness, but is not isolated by these
  runs. The experiment changes several architectural details, so it does not prove
  that the gate or forecast feedback alone causes the improvement.
- Try the gated branch with the successful gap-only mask and no revision augmentation;
  retain B with the same recipe. Do not select augmentation merely because it helps
  the revision endpoint while worsening the primary forecast endpoint.
- Natural forecast-only validation is now consistent across candidates, but there
  is no contemporaneous masked-validation control in this experiment. Its independent
  causal benefit is not identified by comparing with older screens.
- All results remain retrospective development CV with finalized older history and
  supplied-final fallbacks. None resolves untouched future-season performance or
  establishes the effect of seasonal reporting shifts.

The audit/regeneration commands are `.venv/bin/python analysis/b1-revisions-review/audit.py`
and `.venv/bin/python scripts/plot_b1_revisions.py`. They aggregate saved scores;
no models are retrained or repredicted. Corrections dated 2026-09-18.
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
