"""Export a separate revision explorer and the actual finalization split calendars.

Run: python -m tapestry.explorer.revisions --experiment data/experiments/reporting-triangle-v11-20261001
This reads the pinned panel and completed manifests; it neither fits nor scores models.
"""
import argparse
import base64
import json
from pathlib import Path
import shutil

import numpy as np

from tapestry.dataset.build import load, context_end, LAG_ONE_COVARIATES
from tapestry.dataset.finalization import signals, boundary_rows, split, validation_split
from tapestry.experiment.planner import read_jobs, seed_state
from tapestry.experiment.provenance import sha256
from tapestry.model.scenario import Scenario


def packed(values):
    return base64.b64encode(np.asarray(values, dtype='<f4').tobytes()).decode('ascii')


def spans(roles):
    starts = np.flatnonzero(np.r_[True, roles[1:] != roles[:-1]])
    ends = np.r_[starts[1:], len(roles)]
    return [[int(a), int(b), int(roles[a])] for a, b in zip(starts, ends) if roles[a]]


def selections(experiment):
    choices = {}
    for job in read_jobs(experiment):
        scenario = Scenario.from_string(job['scenario'])
        if scenario.task != 'finalize':
            continue
        key = (scenario.finalization_cv, scenario.finalization_weeks, scenario.finalization_maturity,
               scenario.scored_seasons)
        if key not in choices or scenario.lookback == 3:
            choices[key] = (scenario, job)
    result = []
    for key, (scenario, job) in sorted(choices.items()):
        attempt = None
        for seed in job['seeds']:
            candidate, _, done = seed_state(experiment, job['scenario'], seed)
            if done:
                attempt = candidate
                break
        if attempt is None:
            raise ValueError(f'No completed run for {job["scenario"]}')
        for fold in scenario.scored_seasons:
            manifest = json.loads((attempt / f'eval_{fold}' / 'manifest.json').read_text())
            result.append((scenario, fold, manifest))
    return result


def role_data(panel, name, truth, asof, locations, scenario, fold, manifest):
    rows = boundary_rows(panel, name, truth, asof, locations, scenario.lookback, scenario.finalization_weeks)
    training, scoring, info = split(panel, rows, scenario, fold)
    saved = next(r for r in manifest['coverage'] if r['signal'] == name)
    if int(training.sum()) != saved['training_cells'] or int(scoring.sum()) != saved['eligible_score_cells']:
        raise ValueError(f'{name}/{fold}: split differs from the saved experiment')
    trained = np.bincount(rows['location'][training], minlength=len(locations)) > 0
    scoring &= trained[rows['location']]
    if saved['status'] == 'complete' and int(scoring.sum()) != saved['score_cells']:
        raise ValueError(f'{name}/{fold}: scored support differs from the saved experiment')
    roles = np.zeros((scenario.finalization_weeks, len(panel['dates']), len(locations)), np.uint8)
    if saved['status'] == 'complete':
        date_index = np.searchsorted(panel['dates'].astype(str), rows['boundary'])
        for keep, code in ((training, 1), (scoring, 2)):
            roles[rows['age'][keep], date_index[keep], rows['location'][keep]] = code
    if saved['status'] == 'complete' and 'validation_cells' in saved:
        _, validation = validation_split(rows, training, maturity=scenario.finalization_maturity)
        roles[rows['age'][validation], date_index[validation], rows['location'][validation]] = 3
    # For a fixed age, a reference week maps to exactly one Wednesday; no role
    # aggregation hides a mixture of training and held-out examples here.
    runs = [[spans(roles[age, :, l]) for l in range(len(locations))]
            for age in range(scenario.finalization_weeks)]
    return dict(runs=runs, status=saved['status'], reason=saved.get('reason', ''), **info)


def build(experiment, output):
    settings = json.loads((experiment / 'experiment.json').read_text())
    dataset = Path(settings['dataset'])
    if sha256(dataset) != settings['dataset_sha256']:
        raise ValueError('Panel differs from the experiment-pinned panel')
    panel = load(dataset)
    metadata = json.loads(str(panel['metadata']))
    dates, issues = panel['dates'].astype(str), panel['issuance_dates'].astype(str)
    ends = np.array([context_end(d) for d in issues])
    origin = np.searchsorted(dates, ends)
    origin[ends < dates[0]] = -1
    week_index = origin[:, None] - np.arange(12)[None, :]
    choices = selections(experiment)
    descriptors = [dict(id=f'cv{i}', protocol=s.finalization_cv, weeks=s.finalization_weeks,
                        fold=fold, maturity=s.finalization_maturity, scenario=s.scenario_string,
                        fit_cutoff=m['fit_cutoff']) for i, (s, fold, m) in enumerate(choices)]
    output.mkdir(parents=True, exist_ok=True)
    static = Path(__file__).parent / 'static'
    shutil.copy2(static / 'revisions.html', output / 'index.html')
    shutil.copy2(static / 'revisions.js', output / 'revisions.js')
    shutil.copy2(static / 'revision-comparison.html', output / 'comparison.html')
    catalog, exported = [], {}
    for name, truth, asof, locations in signals(panel):
        values = asof[np.arange(len(issues))[:, None], week_index.clip(0)]
        values = np.where((week_index >= 0)[..., None], values, np.nan)
        payload = dict(name=name, locations=locations.tolist(), truth=packed(truth), reports=packed(values),
                       lag=int(name in LAG_ONE_COVARIATES),
                       cv={d['id']: role_data(panel, name, truth, asof, locations, s, fold, manifest)
                           for d, (s, fold, manifest) in zip(descriptors, choices)})
        (output / f'{name}.json').write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False))
        catalog.append(dict(name=name, file=f'{name}.json', locations=locations.tolist()))
        exported[name] = payload
    manifest = dict(dates=dates.tolist(), issuances=issues.tolist(), origins=origin.tolist(),
                    truth_day=metadata['truth_day'], dataset_sha256=settings['dataset_sha256'],
                    experiment=experiment.name, series=catalog, cv=descriptors,
                    cutoff_policy='Wednesday panel snapshot; no additional publication-time reconstruction',
                    encoding='base64 little-endian float32; reports[issuance, T-offset 0..11, location]; truth[date, location]')
    (output / 'manifest.json').write_text(json.dumps(manifest, separators=(',', ':')))
    static_calendars(panel, descriptors, exported, output)
    print(json.dumps(dict(output=str(output), signals=len(catalog), Wednesdays=len(issues),
                          cv_layouts=len(descriptors), reference_day=metadata['truth_day'])))


def static_calendars(panel, descriptors, exported, output):
    """US target examples, matching the forecast calendar's colors and layout."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    from tapestry.evaluation.plots import ROLE_COLORS
    colors = [ROLE_COLORS['unused'], ROLE_COLORS['fit'], ROLE_COLORS['score'], ROLE_COLORS['validation']]
    x = pd.to_datetime(panel['dates'].astype(str))
    names = panel['target_names'].astype(str)
    for protocol in ('rolling', 'season'):
        folds = [d for d in descriptors if d['protocol'] == protocol and d['weeks'] == max(x['weeks'] for x in descriptors)]
        if not folds:
            continue
        fig, axes = plt.subplots(len(folds), len(names), figsize=(20, 2.7 * len(folds)), squeeze=False, sharex=True)
        for row, descriptor in enumerate(folds):
            for col, name in enumerate(names):
                data = exported[name]
                li = data['locations'].index('US')
                layout = data['cv'][descriptor['id']]
                ax = axes[row, col]
                ax.set_facecolor(colors[0])
                for a, b, role in layout['runs'][0][li]:
                    ax.axvspan(x[a] - pd.Timedelta(days=3.5), x[b-1] + pd.Timedelta(days=3.5), color=colors[role], lw=0)
                ax.plot(x, panel['targets'][:, li, col], color='black', lw=.8)
                if layout['status'] != 'complete':
                    ax.text(.5, .82, 'No fit: insufficient vintage support', transform=ax.transAxes, ha='center', fontsize=7)
                if row == 0:
                    ax.set_title(name.replace('_', ' '), fontsize=8)
                if col == 0:
                    ax.set_ylabel(descriptor['fold'], fontsize=9)
                ax.tick_params(labelsize=7)
        handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=l) for c, l in zip(colors,
                   ('Not used', 'Training statistics', 'Held-out evaluation', 'Missing-cell validation'))]
        fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False)
        fig.suptitle(f'{protocol}: actual finalization split, US targets, R={folds[0]["weeks"]}, newest reconstructed week\n'
                     'Reference-value series; orange: training-only missing-cell validation; reported correction strength uses up to 52 training weeks. Interactive view covers every state, signal and age.', fontsize=11)
        fig.tight_layout(rect=(0, .06, 1, .94))
        fig.savefig(output / f'cv-layout-{protocol}-us.png', dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, default=Path('data/experiments/reporting-triangle-v11-20261001'))
    parser.add_argument('--output', type=Path, default=Path('docs/data/revisions'))
    args = parser.parse_args()
    build(args.experiment, args.output)


if __name__ == '__main__':
    main()
