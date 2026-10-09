"""The experiment report page, generated from a ranking (`planner rank`).

Writes `docs/experiments/<experiment>/`: `index.md`, PNG figures and the ranking CSVs.
Text between the write-up markers is kept across regenerations; an existing page
without both markers is never overwritten. Each configuration is shown in its recipe's
own forecast view (`forecast_view`, choice F); other views are diagnostics (2026-10-09:
the best view per configuration used to be picked after scoring). The model-choices
table describes every configuration (grouped when identical), with training seasons per
fold, evaluated seasons and labels, each heading linked to docs/reference/model-choices.md.

Figures (lower WIS is better throughout):
1. `model-comparison.png`: mean headline WIS per configuration, with the range over
   fitting seeds; log admissions, admissions and ED; states/DC and US.
2. `dotplot-<target>.png` (seaborn PairGrid dot plot): every configuration's WIS for states/DC and
   US, WIS components (dispersion, under- and overprediction) and interval coverage, seeds and their mean.
3. `monthly-log-wis.png`: the best three configurations' states/DC log-admission WIS by
   reference month, one panel per evaluated season.
4. `hub-ranking-<season>.png`: CDC pairwise relative WIS among Hub models on the frozen
   Hub tasks, the best three configurations (seed means) highlighted.
5. `fans-<location>-<hosp|ed>.png`: US and NC 50%/90% intervals and medians of the Hub
   ensemble and the best three configurations (lowest seed), every fourth reference date.
6. `cv-layout.png`: each week's role per held-out season (`dataset.cv.week_roles`).

Also regenerates the scenario field key `docs/reference/scenario.md`. The runs index
`docs/experiments/index.md` is curated by hand (one line per phase; 9 October 2026): add new
reports there. History: replaces `evaluation/plots.py`, `report_layout.py`,
`reports.py` and `effects.py` (B0 composite-score report) and scripts/report_b7.py (2026-10-08).
"""
from dataclasses import fields
from datetime import datetime
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from chromantis.data.geography import STATE_FIPS
from chromantis.dataset.cv import season, week_roles
from chromantis.model.scenario import CODES, MEANING, Scenario
from .hubs import CHANNEL, export
from .ranking import views
from .standard import frozen_cases, input_fills, star_note

MAX_TABLE_BYTES = 1_000_000
WRITEUP_START, WRITEUP_END = '<!-- write-up: kept across regenerations -->', '<!-- end write-up -->'
ROLE_COLORS = {'fit': '#cfe0f3', 'validation': '#f6c77b', 'score': '#f3b6b6', 'unused': '#eeeeee'}
ROLE_LABELS = {'fit': 'training (fit)', 'validation': 'early-stopping validation (inner fit only; refit trains on it)',
               'score': 'held-out season (scored)', 'unused': 'unused'}
MODEL_COLOR, ENSEMBLE_COLOR, GOOGLE_COLOR = '#247b9c', '#555555', '#e69f00'
FAN_LOCATIONS = ('US', 'NC')
FAN_EVERY = 4
HEADLINE = [('wk inc flu hosp', 'log', 'Log-admission WIS'), ('wk inc flu hosp', 'natural', 'Admission WIS'),
            ('wk inc flu prop ed visits', 'natural', 'ED-proportion WIS')]
VIEW_MEANING = {'raw': 'reports as they are', 'corrected': 'newest weeks corrected by the fitted trees',
                'half': '50/50 predictive mixture of raw and corrected inputs', 'calibrated': 'corrected, spread recalibrated',
                'sampled': 'corrected with sampled correction errors', 'delayed': 'newest admission week withheld',
                'nokinsa': 'covariates withheld'}


CHOICES = 'reference/model-choices.md'  # relative to docs/


COLUMNS = [('Training histories (A)', 'a-training-histories'), ('Error source (B)', 'b-error-source'),
           ('Error signals (C)', 'c-error-signals'), ('Correction model (D)', 'd-correction-model'),
           ('Evaluation inputs (E)', 'e-evaluation-inputs'), ('Forecast view (F)', 'f-input-view'),
           ('Prediction labels', 'labels-and-folds'), ('Evaluated season ← training seasons', 'labels-and-folds')]


def protocol(s, inputs=None, views=None, seasons=None):
    """The model choices of one configuration, in words (one value per COLUMNS entry)."""
    from chromantis.dataset.cv import training_seasons
    short = lambda season: f'{season[:4]}-{season[7:]}'
    source = {'finalized': 'final values', 'reported': 'actual archived reports',
              'artificial': 'final values with artificial reporting errors'}[s.history_source]
    histories = source + ('; corrected by the cross-fitted correction model' if s.history_correction else '') \
        + ('; plus reconstruction labels' if s.reconstruction_labels else '')
    errors = ('none' if s.history_source != 'artificial' and s.corrector_examples != 'synthetic' and s.evaluation_inputs == 'reported' else
              f'prescribed {short(s.error_reference)} process' if s.error_reference else
              'each fold\'s latest training season' if s.error_seasons == 'latest' else 'all training seasons, recency-weighted')
    signals = {'admissions': 'admissions', 'targets': 'admissions and ED', 'all': 'admissions, ED and covariates'}[s.error_signals]
    corrector = (f'{s.corrector_model} on {s.corrector_examples.replace("_", " ")} examples; newest {s.correction_weeks} '
                 f'week(s) of {"admissions and ED" if s.correction_ed else "admissions"}')
    evaluation = inputs or ('real archived Wednesday reports' if s.evaluation_inputs == 'reported' else
                            f'artificial: prescribed {short(s.error_reference)} errors, {s.evaluation_draws} draw(s)')
    diagnostics = [v for v in ['raw', 'corrected', 'half'] if v != s.forecast_view]
    diagnostics += (['calibrated'] if s.patience and s.evaluation_inputs == 'reported' else []) \
        + (['sampled'] if s.correction_noise else []) + (['delayed'] + (['nokinsa'] if s.covariate_set else []) if s.stress_views else [])
    view = views or f'**{s.forecast_view}**; diagnostics: {", ".join(diagnostics)}'
    labels = 'latest panel values, next 4 weeks' + (f' + last {s.reconstruction_weeks} context weeks' if s.reconstruction_labels else '')
    held = seasons or (('2026-2027',) if s.evaluation_seasons == 'production' else s.scored_seasons)
    folds = '; '.join((f'none held out ← ' if season == '2026-2027' else f'{short(season)} ← ')
                      + ', '.join(short(t) for t in training_seasons(s, season)) for season in sorted(held))
    return [histories, errors, signals, corrector, evaluation, view, labels, folds]


def choices_table(named, link='../../' + CHOICES):
    """Markdown protocol of every configuration [(name, Scenario, overrides)], grouped when identical.

    Each column header links to its explanation; `overrides` may set `inputs`/`views` for a
    report that scored something other than the fit's own evaluation (e.g. a replay)."""
    groups = {}
    for name, scenario, overrides in named:
        groups.setdefault(tuple(protocol(scenario, **overrides)), []).append(name)
    header = lambda: [f'[{title}]({link}#{anchor})' for title, anchor in COLUMNS]
    def who(names):
        return ', '.join(names) if len(names) <= 4 else f'{len(names)} configurations: {", ".join(names[:3])}, …'
    lines = [f'Each heading links to its explanation in [Model choices A–F]({link}). One '
             f'{"column" if len(groups) <= 4 else "row"} per group of configurations with identical choices.', '']
    if len(groups) <= 4:
        lines += ['| Choice | ' + ' | '.join(who(n) for n in groups.values()) + ' |', '|---|' + '---|' * len(groups)]
        for k, title in enumerate(header()):
            lines.append(f'| {title} | ' + ' | '.join(values[k] for values in groups) + ' |')
    else:
        lines += ['| Configurations | ' + ' | '.join(header()) + ' |', '|---|' + '---|' * len(COLUMNS)]
        lines += [f'| {who(names)} | ' + ' | '.join(values) + ' |' for values, names in groups.items()]
    return '\n'.join(lines)


def report_metadata(directory, name):
    path = Path(directory) / 'report.json'
    if path.exists():
        return json.loads(path.read_text())
    metadata = dict(date=datetime.now(ZoneInfo('America/New_York')).date().isoformat(), title=name, stage='forecast')
    path.write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


def write_scenario_key(path):
    """`docs/reference/scenario.md`: every `Scenario` field with type, default, allowed values, meaning."""
    from chromantis.dataset.build import SOURCE_GROUPS
    lines = ['# Scenario fields', '',
             'Generated from `chromantis.model.scenario.Scenario` (`CODES`, `MEANING`) by '
             '`chromantis.evaluation.report.write_scenario_key`, rewritten by every report; do not edit by hand. '
             'A scenario string names only the fields that differ from these defaults, as `key=value` tokens '
             'joined by `,`; booleans are written `0`/`1`. See [the architecture](../architecture.md) and '
             '[Workflow](../workflow.md).', '',
             '| Field | Type | Default | Allowed values | Meaning |', '|---|---|---|---|---|']
    for f in fields(Scenario):
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


def save(fig, path):
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)
    return path


def model_comparison(summary, seeds, selected, out):
    order = selected.name.tolist()
    y = np.arange(len(order))
    fig, axes = plt.subplots(3, 2, figsize=(16, 3 + .45 * len(order) * 3), constrained_layout=True, squeeze=False)
    for row, (target, scale, title) in enumerate(HEADLINE):
        for col, geography in enumerate(('states_dc', 'US')):
            ax = axes[row, col]
            pick = lambda t: t.merge(selected, on=['name', 'history'])
            means = pick(summary)
            means = means[(means.target == target) & (means.scale == scale) & (means.geography == geography)].set_index('name').reindex(order)
            per_seed = pick(seeds)
            per_seed = per_seed[(per_seed.target == target) & (per_seed.scale == scale) & (per_seed.geography == geography)]
            ranges = per_seed.groupby('name').score.agg(['min', 'max']).reindex(order)
            mean = means['mean'].to_numpy()
            ax.barh(y, mean, color=MODEL_COLOR, alpha=.85)
            ax.errorbar(mean, y, xerr=np.maximum(0, np.stack((mean - ranges['min'], ranges['max'] - mean))),
                        fmt='none', ecolor='#152934', capsize=3)
            ax.set_yticks(y, order if col == 0 else [''] * len(order), fontsize=8)
            ax.invert_yaxis()
            ax.set_title(f"{title}, {'states/DC' if geography == 'states_dc' else 'US'}", fontsize=10)
            ax.grid(axis='x', alpha=.2)
            ax.set_xlabel('Mean WIS per task; lower is better; line spans the fitting seeds', fontsize=8)
    fig.suptitle('Configurations in their own forecast view: October-May, held-out seasons averaged equally', fontsize=11)
    return save(fig, out / 'model-comparison.png')


DOT_METRICS = [('states_dc', 'mean_wis', 'WIS, states/DC'), ('US', 'mean_wis', 'WIS, US'),
               ('states_dc', 'mean_dispersion', 'Dispersion, states/DC'),
               ('states_dc', 'mean_underprediction', 'Underprediction, states/DC'),
               ('states_dc', 'mean_overprediction', 'Overprediction, states/DC'),
               *[('states_dc', f'covered_{c}', f'{c}% coverage, states/DC') for c in (50, 80, 90, 95)],
               *[('US', f'covered_{c}', f'{c}% coverage, US') for c in (50, 95)]]


def dotplot(metrics, selected, out):
    """Seaborn PairGrid dot plot (after the `pairgrid_dotplot` example), one file per target and scale.

    Rows: every configuration in its own forecast view, in ranking order. Columns: WIS for
    states/DC and US, the three WIS components for states/DC (they sum to its WIS), and
    50/80/90/95% interval coverage for states/DC and 50/95% for the US. Light dots are fitting
    seeds, dark dots their mean; every value averages the evaluated seasons equally. Red
    dotted lines mark nominal coverage. Lower WIS is better; coverage should match nominal."""
    import seaborn as sns
    order = selected.name.tolist()
    rows = metrics.merge(selected, on=['name', 'history'])
    paths = []
    for target, scale, title in HEADLINE:
        part = rows[(rows.target == target) & (rows.scale == scale)]
        if part.empty:
            continue
        wide = {label: part[part.geography == geography].set_index(['name', 'seed'])[metric]
                for geography, metric, label in DOT_METRICS}
        wide = pd.DataFrame(wide).reset_index()
        columns = [label for _, _, label in DOT_METRICS]
        means = wide.groupby('name')[columns].mean().reset_index()
        g = sns.PairGrid(wide, x_vars=columns, y_vars=['name'], height=max(3., .32 * len(order) + 1.2), aspect=.42)
        g.map(sns.stripplot, order=order, orient='h', jitter=False, size=6, color='#9ec5d6', linewidth=0)
        for ax, column in zip(g.axes.flat, columns):
            sns.stripplot(data=means, x=column, y='name', order=order, orient='h', jitter=False, size=9,
                          color=MODEL_COLOR, linewidth=1, edgecolor='w', ax=ax)
            ax.set_title(column, fontsize=8)
            ax.set_xlabel('')
            ax.tick_params(axis='x', labelsize=7)
            if column.startswith(tuple(str(c) for c in (50, 80, 90, 95))):
                ax.axvline(int(column.split('%')[0]) / 100, color='#c33f30', ls=':', lw=1)
                ax.set_xlim(0, 1.02)
            ax.xaxis.grid(False)
            ax.yaxis.grid(True)
        g.axes.flat[0].set_ylabel('')
        sns.despine(left=True, bottom=True)
        g.figure.suptitle(f'{title}: every configuration in its own forecast view; light = seeds, dark = seed mean; '
                          'seasons averaged equally; lower WIS is better, coverage should meet the red line', fontsize=9, y=1.02)
        name = f'dotplot-{"log-admissions" if scale == "log" else "admissions" if "hosp" in target else "ed"}.png'
        paths.append(save(g.figure, out / name))
    return paths


def monthly(details, selected, out):
    top = selected.head(3)
    rows = details.merge(top, on=['name', 'history'])
    rows = rows[(rows.dimension == 'month') & (rows.target == 'wk inc flu hosp') & (rows.scale == 'log')
                & (rows.geography == 'states_dc')]
    seasons = sorted(rows.season.unique())
    fig, axes = plt.subplots(1, len(seasons), figsize=(5.2 * len(seasons), 4.4), constrained_layout=True, squeeze=False)
    for ax, held in zip(axes[0], seasons):
        part = rows[rows.season == held]
        for name in top.name:
            line = part[part.name == name].groupby('value').wis.mean().sort_index()
            ax.plot(np.arange(len(line)), line.to_numpy(), marker='o', label=name)
            ax.set_xticks(np.arange(len(line)), [v[5:] for v in line.index])
        ax.set_title(f'Evaluated on {held}')
        ax.set_xlabel('Reference-date month')
        ax.set_ylabel('States/DC log-admission WIS; lower is better')
        ax.grid(alpha=.2)
    axes[0, -1].legend(fontsize=8)
    return save(fig, out / 'monthly-log-wis.png')


def hub_ranking(pairwise, selected, out):
    """CDC pairwise relative WIS among Hub models, ours (seed means in their own forecast view) highlighted."""
    paths = []
    ours = pairwise[pairwise.ours.astype(bool)].merge(selected.head(3), on=['name', 'history'])
    for held, part in pairwise.groupby('season'):
        targets = sorted(part.target.unique(), key=list(CHANNEL).index)
        fig, axes = plt.subplots(len(targets), 2, figsize=(11, 3.4 * len(targets)), squeeze=False)
        for i, target in enumerate(targets):
            for j, scale in enumerate(('natural', 'log')):
                ax = axes[i, j]
                hub = part[(part.target == target) & (part.scale == scale) & ~part.ours.astype(bool)]
                if hub.empty:
                    ax.axis('off')
                    continue
                values = hub.dropna(subset=['relative_wis']).set_index('model').relative_wis
                mine = ours[(ours.season == held) & (ours.target == target) & (ours.scale == scale)]
                for row in mine.itertuples():
                    values[row.name] = row.relative_wis
                values = values.sort_values()
                colors = [MODEL_COLOR if n in set(mine.name) else GOOGLE_COLOR if n.startswith('Google_') else '#bbbbbb'
                          for n in values.index]
                ax.barh(range(len(values)), values.to_numpy(), color=colors)
                ax.set_yticks(range(len(values)), values.index, fontsize=5)
                ax.invert_yaxis()
                ax.axvline(1, color='k', ls='--', lw=.7)
                ax.set_title(f'{target.removeprefix("wk inc ")}, {scale} scale', fontsize=8)
        fig.suptitle(f'{held}: relative WIS among Hub models (CDC pairwise method; states/DC; lower is better, '
                     '1 = Hub baseline)\nblue = our best three configurations (seed mean), orange = Google', fontsize=9)
        fig.tight_layout()
        paths.append(save(fig, out / f'hub-ranking-{held}.png'))
    return paths


def fans(panel, runs, selected, frozen, out):
    """US and NC fans of the Hub ensemble and the best three configurations (lowest seed, own forecast view)."""
    chosen = {}
    for row in selected.head(3).itertuples():
        run = min((r for r in runs if r['name'] == row.name), key=lambda r: r['seed'])
        chosen[f'{row.name} ({row.history})'] = export(views(run['path'])[row.history])
    first = next(iter(chosen.values()))
    dates = sorted(d for held in dict.fromkeys(k[0] for k in first)
                   for d in sorted(first[(held, 'wk inc flu hosp')].reference_date.unique())[::FAN_EVERY])
    ensemble = {}
    for case in frozen_cases(frozen):
        table = pd.read_parquet(Path(frozen) / case['directory'] / 'quantiles.parquet')
        ensemble[(case['season'], case['target'])] = table[table.model == case['ensemble']]
    tables = {'Hub ensemble': ensemble, **chosen}
    truth_dates = pd.to_datetime([str(d) for d in panel['dates']])
    held_seasons = sorted({held for held, _ in first})
    shown = np.isin([season(d) for d in truth_dates.strftime('%Y-%m-%d')], held_seasons)
    fips = {v: k for k, v in STATE_FIPS.items()} | {'US': 'US'}
    targets = sorted({t for held, t in first}, key=list(CHANNEL).index)
    paths = []
    for location in FAN_LOCATIONS:
        index = list(panel['locations']).index(location)
        for kind, keep in (('hosp', lambda t: 'prop ed' not in t), ('ed', lambda t: 'prop ed' in t)):
            rows = [t for t in targets if keep(t)]
            if not rows:
                continue
            fig, axes = plt.subplots(len(rows), len(tables), figsize=(5.5 * len(tables), 2.6 * len(rows) + .8),
                                     sharex=True, sharey='row', squeeze=False)
            for column, (source, frames) in enumerate(tables.items()):
                color = ENSEMBLE_COLOR if source == 'Hub ensemble' else MODEL_COLOR
                for r, target in enumerate(rows):
                    ax = axes[r, column]
                    ax.plot(truth_dates[shown], panel['targets'][shown, index, CHANNEL[target]], color='black', lw=1)
                    for held in held_seasons:
                        if (held, target) not in frames:
                            continue
                        table = frames[(held, target)]
                        part = table[table.location.eq(fips[location]) & table.reference_date.isin(dates)]
                        for _, fan in part.sort_values('horizon').groupby('reference_date'):
                            x = pd.to_datetime(fan.target_end_date)
                            ax.fill_between(x, fan['q0.05'], fan['q0.95'], color=color, alpha=.15, lw=0)
                            ax.fill_between(x, fan['q0.25'], fan['q0.75'], color=color, alpha=.3, lw=0)
                            ax.plot(x, fan['q0.5'], color=color, lw=1)
                    if r == 0:
                        ax.set_title(source, fontsize=9)
                    if column == 0:
                        ax.set_ylabel(target.removeprefix('wk inc '), fontsize=9)
                    ax.tick_params(labelsize=7)
            fig.suptitle(f'{location}: 50%/90% intervals and median (horizons 0-3) every {FAN_EVERY} weeks; '
                         'black = finalized truth; Hub ensemble only where frozen support exists', fontsize=10)
            fig.tight_layout()
            paths.append(save(fig, out / f'fans-{location}-{kind}.png'))
    return paths


def cv_layout(panel, scenario, out):
    dates = np.array([str(d) for d in panel['dates']])
    x = pd.to_datetime(dates)
    us = list(panel['locations']).index('US')
    seasons = scenario.scored_seasons
    fig, axes = plt.subplots(len(seasons), 1, figsize=(12, 1.8 * len(seasons) + .6), sharex=True, squeeze=False)
    for row, held in enumerate(seasons):
        roles = week_roles(dates, scenario, held)
        starts = np.flatnonzero(np.r_[True, roles[1:] != roles[:-1]])
        ax = axes[row, 0]
        for a, b in zip(starts, np.r_[starts[1:], len(roles)] - 1):
            ax.axvspan(x[a] - pd.Timedelta(days=3.5), x[b] + pd.Timedelta(days=3.5), color=ROLE_COLORS[roles[a]], lw=0)
        ax.plot(x, panel['targets'][:, us, 0], color='black', lw=1)
        ax.set_ylabel(f'held out\n{held}', fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=ROLE_LABELS[r]) for r, c in ROLE_COLORS.items()
               if scenario.patience or r != 'validation']
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=8, frameon=False)
    fig.suptitle('Week roles per fold; black = finalized US flu admissions', fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, 1))
    return save(fig, out / 'cv-layout.png')


def write_report(folder, ranking, runs, root='docs/experiments'):
    """`<root>/<experiment>/index.md` and its figures and tables, from one complete ranking."""
    from chromantis.dataset.build import load
    folder, ranking = Path(folder), Path(ranking)
    out = Path(root) / folder.name
    out.mkdir(parents=True, exist_ok=True)
    page = out / 'index.md'
    writeup = '_Not written yet._'
    if page.exists():
        text = page.read_text()
        if WRITEUP_START not in text or WRITEUP_END not in text:
            raise ValueError(f'{page} exists without both write-up markers; add them around the hand-written text '
                             '(or move the page away) so it is not overwritten')
        writeup = text.split(WRITEUP_START, 1)[1].split(WRITEUP_END, 1)[0].strip()
    write_scenario_key(Path(root).parent / 'reference' / 'scenario.md')
    # Summary tables only (user rule, 2026-10-09): tables over 1 MB stay in the ranking folder.
    for path in ranking.glob('*.csv'):
        if path.stat().st_size <= MAX_TABLE_BYTES:
            shutil.copy2(path, out / path.name)
    for name in ('study.json', 'experiment.json', 'jobs.csv'):
        if (folder / name).exists():
            shutil.copy2(folder / name, out / name)
    summary = pd.read_csv(ranking / 'headline-rankings.csv')
    seeds = pd.read_csv(ranking / 'headline-seed-scores.csv')
    details = pd.read_csv(ranking / 'score-details.csv', dtype={'value': str})
    pairwise = pd.read_csv(ranking / 'hub-pairwise-scores.csv')
    # Each configuration in its recipe's own forecast view (Scenario.forecast_view): no selection on scores.
    deployed = summary[summary.deployed & (summary.target == 'wk inc flu hosp') & (summary.scale == 'log')
                       & (summary.geography == 'states_dc')].sort_values('mean')
    selected = deployed[['name', 'history']].drop_duplicates()
    selected = selected[selected.name.isin({r['name'] for r in runs})]
    selected.to_csv(out / 'forecast-views.csv', index=False)
    settings = json.loads((folder / 'experiment.json').read_text())
    panel = load(settings['dataset'])
    first = Scenario.from_string(runs[0]['config_id'])  # fold layout figure only
    metrics = pd.read_csv(ranking / 'metric-seed-scores.csv')
    figures = [model_comparison(summary, seeds, selected, out), *dotplot(metrics, selected, out),
               monthly(details, selected, out),
               *hub_ranking(pairwise, selected, out), *fans(panel, runs, selected, settings['frozen'], out),
               cv_layout(panel, first, out)]
    # Finalized fills exist only where actual Wednesday reports were read, per evaluation-input group.
    notes = []
    for inputs in dict.fromkeys(Scenario.from_string(r['config_id']).evaluation_inputs for r in runs):
        member = next(r for r in runs if Scenario.from_string(r['config_id']).evaluation_inputs == inputs)
        notes.append('Configurations evaluated on artificial histories read no archived report.' if inputs == 'prescribed' else
                     star_note(input_fills(Path(member['path']), settings['frozen'])))
    study = json.loads((folder / 'study.json').read_text()) if (folder / 'study.json').exists() else {}
    metadata = report_metadata(out, folder.name)
    table = summary.merge(selected, on=['name', 'history'])
    get = lambda name, target, scale, geography: table[(table.name == name) & (table.target == target) & (table.scale == scale)
                                                         & (table.geography == geography)]['mean']
    fmt = lambda v: f'{float(v.iloc[0]):.4g}' if len(v) else ''
    count = lambda name: int(seeds[seeds.name == name].seed.nunique())
    # Folds actually evaluated (from each configuration's run manifest), not the scenario default.
    folds = lambda config: json.loads((Path(next(r for r in runs if r['config_id'] == config)['path']) / 'manifest.json').read_text())['folds']
    named = [(name, Scenario.from_string(config), dict(seasons=tuple(folds(config))))
             for name, config in dict.fromkeys((r['name'], r['config_id']) for r in runs)]
    lines = [f'# {metadata["date"]} · {metadata["title"]}', '', study.get('description', ''), '',
             f'{len(runs)} runs, {len(selected)} configurations. Each configuration is retrained in every fold listed in '
             'its protocol below and scored on October-May reference dates, horizons 0-3, against the latest panel values. '
             'Mean WIS per task; states/DC and US separate; evaluated seasons averaged equally, then seeds. Lower is better. '
             'Every configuration is ranked in its own forecast view (F); the other views are diagnostics in the tables. '
             'Generated by `planner rank`; the write-up below is kept across regenerations.', '',
             *notes, '', '## Model choices', '', choices_table(named), '',
             '## Write-up', '', WRITEUP_START, writeup, WRITEUP_END, '',
             '## Ranking', '',
             '| Configuration | Seeds | Forecast view | States/DC log adm. | States/DC adm. | States/DC ED | US log adm. | US adm. |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    for row in selected.itertuples():
        lines.append(f'| {row.name} | {count(row.name)} | {row.history} | '
                     f'{fmt(get(row.name, "wk inc flu hosp", "log", "states_dc"))} | {fmt(get(row.name, "wk inc flu hosp", "natural", "states_dc"))} | '
                     f'{fmt(get(row.name, "wk inc flu prop ed visits", "natural", "states_dc"))} | '
                     f'{fmt(get(row.name, "wk inc flu hosp", "log", "US"))} | {fmt(get(row.name, "wk inc flu hosp", "natural", "US"))} |')
    lines += ['', 'Input views: ' + '; '.join(f'`{k}` = {v}' for k, v in VIEW_MEANING.items()) + '.', '']
    lines += [f'![{p.stem}]({p.name})\n' for p in figures]
    tables = [('All views', 'headline-rankings.csv'), ('seasons and draws', 'headline-season-scores.csv'),
              ('seeds', 'headline-seed-scores.csv'), ('WIS components and coverage per seed', 'metric-seed-scores.csv'),
              ('month, horizon, location', 'score-details.csv'), ('Hub-relative', 'hub-relative-scores.csv'),
              ('Hub pairwise', 'hub-pairwise-scores.csv'), ('coverage', 'distribution-scores.csv')]
    shown = [f'[{label}]({name})' if (out / name).exists() else f'{label} (`{name}`, in `{ranking}`)' for label, name in tables]
    lines += ['## Tables', '', ' · '.join(shown) + '. Tables over 1 MB stay in the ranking folder; '
              '`planner rank` regenerates them.', '',
              '## Appendix: configurations', '', 'Field meanings: [scenario field key](../../reference/scenario.md).', '',
              '| Configuration | Scenario string | Seeds |', '|---|---|---|']
    for name, config in dict.fromkeys((r['name'], r['config_id']) for r in runs):
        lines.append(f'| {name} | `{config}` | {", ".join(str(r["seed"]) for r in runs if r["name"] == name)} |')
    page.write_text('\n'.join(lines) + '\n')
    return page
