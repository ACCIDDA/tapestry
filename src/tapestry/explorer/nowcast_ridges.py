"""Render completed nowcasts as date-offset revision lines; no fitting or scoring."""
import argparse
import base64
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tapestry.dataset.build import load
from tapestry.experiment.provenance import sha256


def percent(value, report):
    value, report = np.asarray(value), np.asarray(report)
    out = np.full(value.shape, np.nan)
    valid = np.isfinite(value) & np.isfinite(report)
    np.divide(100 * (value - report), report, out=out, where=valid & (report != 0))
    out[valid & (value == 0) & (report == 0)] = 0
    return out


def packed(a):
    return base64.b64encode(np.asarray(a, dtype='<f4').tobytes()).decode()


def build(experiment, output, fold):
    settings = json.loads((experiment / 'experiment.json').read_text())
    dataset = Path(settings['dataset'])
    if sha256(dataset) != settings['dataset_sha256']:
        raise ValueError('Panel does not match the completed experiment')
    panel = load(dataset)
    runs = pd.read_csv(experiment / 'runs.csv')
    chosen = runs[runs.status.eq('complete') & runs.scenario.str.contains('finalization_cv=season')]
    if len(chosen) != 1:
        raise ValueError('Expected one completed seasonal configuration')
    run = chosen.iloc[0]
    source = experiment / run.attempt / f'eval_{fold}' / 'finalizations.csv.gz'
    frame = pd.read_csv(source)
    if frame.duplicated(['signal', 'location', 'issuance', 'age']).any():
        raise ValueError('Ambiguous prediction rows')
    dates = panel['dates'].astype(str).tolist()
    issues = sorted(frame.issuance.unique())
    locations = panel['locations'].astype(str).tolist()
    ages = int(frame.age.max()) + 1
    date_index = {d: i for i, d in enumerate(dates)}
    issue_index = {d: i for i, d in enumerate(issues)}
    location_index = {d: i for i, d in enumerate(locations)}
    names = panel['target_names'].astype(str).tolist()
    origins = [date_index[str((pd.Timestamp(d) - pd.Timedelta(days=4)).date())] for d in issues]
    records = []
    for name, g in frame.groupby('signal', sort=False):
        shape = (len(issues), ages, len(locations))
        arrays = {key: np.full(shape, np.nan) for key in ('report', 'truth', 'prediction')}
        w = g.issuance.map(issue_index).to_numpy()
        k = g.age.to_numpy()
        l = g.location.map(location_index).to_numpy()
        expected = np.array(origins)[w] - k
        if not np.array_equal(np.array(dates)[expected], g.boundary.to_numpy()):
            raise ValueError('Observation dates do not align with Wednesday minus age')
        for key, values in arrays.items():
            values[w, k, l] = g[key]
        history = panel['targets'][:, :, names.index(name)]
        # Saved scoring labels must be in the same raw units as the plotted curve.
        if not np.allclose(history[expected, l], g.truth, rtol=2e-6, equal_nan=True):
            raise ValueError('Reference curve differs from saved prediction labels')
        records.append(dict(name=name, history=packed(history), **{k: packed(v) for k, v in arrays.items()}))
    latest = frame.loc[frame.age.eq(0), 'issuance'].max()
    metadata = json.loads(str(panel['metadata']))
    payload = dict(experiment=experiment.name, fold=fold, scenario=run.scenario, source=str(source),
                   reference_day=metadata['truth_day'], dates=dates, issues=issues,
                   origins=origins, locations=locations, ages=ages, series=records,
                   default_end=latest, dataset_sha256=settings['dataset_sha256'])
    output.mkdir(parents=True, exist_ok=True)
    template = (Path(__file__).parent / 'static' / 'nowcast-ridges.html').read_text()
    (output / 'nowcast.html').write_text(template.replace('__PAYLOAD__', json.dumps(payload, separators=(',', ':'))))
    static_figure(frame, panel, output)
    print(f'Exported {len(records)} targets, {len(locations)} locations, {len(issues)} rounds: {output / "nowcast.html"}')


def static_figure(frame, panel, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as md
    from matplotlib.lines import Line2D
    name = 'nhsn_flu_admissions'
    g = frame[frame.signal.eq(name) & frame.location.eq('US') & frame.issuance.between('2025-11-05', '2026-03-04')].copy()
    if g.empty:
        return
    g['actual_revision'] = percent(g.truth, g.report)
    g['model_revision'] = percent(g.prediction, g.report)
    issues = sorted(g.issuance.unique(), reverse=True)
    finite = g[['actual_revision', 'model_revision']].to_numpy()
    bound = max(5, np.nanmax(np.abs(finite)))
    bound = np.ceil(bound / 5) * 5
    fig, (curve, ax) = plt.subplots(2, 1, figsize=(13, 3 + .53 * len(issues)), sharex=True,
                                  gridspec_kw={'height_ratios': [3, .53 * len(issues)], 'hspace': .06})
    index = panel['target_names'].astype(str).tolist().index(name)
    loc = panel['locations'].astype(str).tolist().index('US')
    curve.plot(pd.to_datetime(panel['dates'].astype(str)), panel['targets'][:, loc, index], color='#263445', lw=1.7)
    newest = g[g.age.eq(0)].sort_values('boundary')
    curve.plot(pd.to_datetime(newest.boundary), newest.prediction, color='#07877b', lw=1.2, marker='o', ms=3)
    curve.set_ylabel('Admissions'); curve.grid(alpha=.15)
    curve.set_title('Flu admissions · US · selected seasonal nowcaster\nReference curve and newest-week nowcasts; dated rows compare predicted and actual revisions', loc='left', fontsize=12)
    for row, issue in enumerate(issues):
        v = g[g.issuance.eq(issue)].sort_values('age', ascending=False).set_index('age').reindex(range(7, -1, -1))
        x = [pd.Timestamp(issue) - pd.Timedelta(days=4 + 7 * age) for age in v.index]
        ax.plot([x[0], pd.Timestamp(issue)], [row, row], color='#dce3e9', lw=.7)
        for key, color, style in [('actual_revision', '#9c4fbc', '-'), ('model_revision', '#07877b', '--')]:
            ax.plot(x, row - .38 * v[key].to_numpy() / bound, color=color, ls=style, lw=1.4, marker='o', ms=2)
        ax.plot([pd.Timestamp(issue)] * 2, [row - .10, row + .10], color='#82909e', lw=.8)
    ax.set_yticks(range(len(issues)), issues, fontsize=8)
    ax.set_ylim(len(issues) - .5, -.65)
    ax.set_ylabel('Forecast Wednesday · each row is its own 0% baseline')
    ax.set_xlim(pd.to_datetime(g.boundary.min()) - pd.Timedelta(days=3), pd.to_datetime(g.issuance.max()) + pd.Timedelta(days=5))
    ax.xaxis.set_major_locator(md.MonthLocator()); ax.xaxis.set_major_formatter(md.DateFormatter('%b %Y'))
    ax.grid(axis='x', alpha=.15); ax.set_xlabel('Observation date (Saturday); row-end tick marks issuance Wednesday')
    handles = [Line2D([], [], color='#263445', label='Reference curve'), Line2D([], [], color='#07877b', ls='--', label='Model correction / newest nowcast'), Line2D([], [], color='#9c4fbc', label='Actual revision')]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.text(.13, .055, f'Both revisions = 100 × (value − report) / report. Same scale on every row: ±{bound:g}% spans 76% of row spacing.\nPositive revisions go upward. Missing/undefined values break lines. Saved 2025–26 development replay; no refitting.', fontsize=8)
    fig.subplots_adjust(left=.13, right=.97, bottom=.11, top=.94)
    fig.savefig(output / 'nowcast-rows-us-flu.png', dpi=170)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment', type=Path, default=Path('data/experiments/seasonal-nowcast-causal-20261001'))
    p.add_argument('--output', type=Path, default=Path('docs/experiments/seasonal-nowcast-20261001'))
    p.add_argument('--fold', default='2025-2026')
    a = p.parse_args()
    build(a.experiment, a.output, a.fold)


if __name__ == '__main__':
    main()
