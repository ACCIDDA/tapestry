"""Forward support and diagnostics using the shared scientific scorer."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tapestry.data.geography import STATE_FIPS
from tapestry.model_data.wednesday import WednesdayDataset
from tapestry.model_data.forward import TARGET_START, TARGET_END, TEST_START
from tapestry.models.provenance import save
from .hubs import HUBS, KEY, QCOLS
from .scoring import match_forecasts
from .totals import quantile_scores, case_cells, cells_totals, frozen_cases, METRICS

# Declared before fitting: mean baseline absolute error / training Q95 <= 1e-4.
DENOMINATOR_THRESHOLD = 1e-4
TARGETS = {c: target for spec in HUBS.values() for target, c in spec['targets'].items()}


def task_frame(ds, indices, c):
    a = ds.arrays
    n, l = len(indices), len(ds.locations)
    dates = a['target_dates'][indices, 2:]
    codes = {v:k for k,v in STATE_FIPS.items()} | {'US':'US'}
    return pd.DataFrame(dict(reference_date=np.repeat(dates[:, 0], 4*l),
        target_end_date=np.repeat(dates.reshape(-1), l),
        location=np.tile([codes[v] for v in ds.locations], n*4),
        horizon=np.tile(np.repeat(np.arange(4), l), n),
        observed=a['Y_future'][indices, :, c].reshape(-1),
        valid=a['Y_future_valid'][indices, :, c].reshape(-1)))


def prepare_support(dataset, source, destination):
    """Freeze pre-existing ensemble support intersected with pinned forward labels."""
    from tapestry.models.backends import sha
    destination, source = Path(destination), Path(source)
    identity = dict(dataset_sha256=sha(dataset), source_manifest_sha256=sha(source/'manifest.json'))
    if (destination/'manifest.json').exists():
        if json.loads((destination/'manifest.json').read_text())['identity'] != identity:
            raise ValueError('Forward scoring support changed; choose a new frozen path')
        return
    ds = WednesdayDataset.load(dataset)
    indices = np.flatnonzero(ds.arrays['issuance_dates'] >= TEST_START)
    cases = []
    counts = []
    for case in frozen_cases(source):
        if case['season'] != '2025-2026':
            continue
        c = next(c for c, target in TARGETS.items() if target == case['target'])
        truth = task_frame(ds, indices, c)
        truth = truth[truth.valid].drop(columns='valid')
        original = pd.read_parquet(source/case['directory']/'units.parquet')
        units = original[KEY].merge(truth, on=KEY, validate='one_to_one')
        folder = destination/case['directory']; folder.mkdir(parents=True, exist_ok=True)
        quantiles = pd.read_parquet(source/case['directory']/'quantiles.parquet')
        quantiles = quantiles[quantiles.model == case['ensemble']].merge(units[KEY], on=KEY, validate='many_to_one')
        match_forecasts(quantiles, units, case['target'])
        units.to_parquet(folder/'units.parquet', index=False)
        quantiles.to_parquet(folder/'quantiles.parquet', index=False)
        cases.append(case)
        counts.append(dict(target=case['target'], tasks=len(units), first=units.reference_date.min(), last=units.reference_date.max()))
    save(destination/'manifest.json', dict(identity=identity, quantiles=QCOLS, cases=cases,
        support='Existing Hub ensemble tasks intersected with pinned forward labels; truth replaced by September 16 reference', counts=counts))


def score(run, frozen):
    manifest = json.loads((run/'manifest.json').read_text())
    ds = WednesdayDataset.load(manifest['dataset'])
    with np.load(run/'forecasts.npz') as saved:
        q, indices, scales = saved['quantiles'], saved['indices'], saved['scales']
    if not np.isfinite(q[:, :, 2:]).all():
        raise ValueError('Nonfinite forward prediction')
    parts, full = [], []
    for c, target in TARGETS.items():
        frame = task_frame(ds, indices, c)
        frame[QCOLS] = q[:, :, 2:, c].reshape(len(QCOLS), -1).T
        frame = frame[frame.valid].drop(columns='valid').reset_index(drop=True)
        metrics = quantile_scores(frame[QCOLS], frame.observed)
        diagnostics = pd.concat((frame[KEY+['observed']], metrics), axis=1)
        diagnostics['scale'] = np.tile(scales[c], len(indices)*4)[ds.arrays['Y_future_valid'][indices, :, c].reshape(-1)]
        diagnostics['scaled_wis'] = diagnostics.wis/diagnostics.scale
        full.append(diagnostics.assign(target=target, season='2025-2026', geography=np.where(diagnostics.location.eq('US'),'US','states_dc')))
        for case in frozen_cases(frozen):
            if case['target'] != target:
                continue
            units = pd.read_parquet(frozen/case['directory']/'units.parquet')
            ensemble = pd.read_parquet(frozen/case['directory']/'quantiles.parquet')
            parts.append(case_cells(match_forecasts(frame, units, target), match_forecasts(ensemble, units, target), case))
    cells = pd.concat(parts, ignore_index=True)
    cells.to_parquet(run/'forecast-cells.parquet', index=False)
    pd.concat(full, ignore_index=True).to_parquet(run/'full-forecast-cells.parquet', index=False)
    totals = cells_totals(cells); totals.to_csv(run/'totals.csv', index=False)
    recent_parts = []
    a = ds.arrays
    for c, target in TARGETS.items():
        for h, age in enumerate((11, 4)):
            for li, location in enumerate(ds.locations):
                for kind, field in [('revision','revision_eligible'), ('reconstruction','reconstruction_eligible')]:
                    valid = a['Y_recent_valid'][indices,h,c,li] & a[field][indices,h,c,li]
                    # Direct has no recent predictor; its own preliminary report is the revision baseline only.
                    if manifest['candidate'] == 'direct':
                        continue
                    values = q[:,valid,h,c,li].T
                    truth = a['Y_recent'][indices[valid],h,c,li]
                    if not len(truth):
                        continue
                    m = quantile_scores(values, truth)
                    preliminary = a['X_values'][indices[valid],-2+h,c,li]
                    m['baseline_ae'] = np.abs(truth-preliminary) if kind == 'revision' else np.nan
                    m['scaled_ae'] = m.ae_median/scales[c,li]
                    m['scaled_wis'] = m.wis/scales[c,li]
                    m['baseline_scaled_ae'] = m.baseline_ae/scales[c,li]
                    m['issuance_date'] = a['issuance_dates'][indices[valid]]
                    m['observation_week'] = a['target_dates'][indices[valid],h]
                    recent_parts.append(m.assign(target=target, location=location, age_days=age, kind=kind, scale=scales[c,li],
                        season='2025-2026', geography='US' if location=='US' else 'states_dc'))
    pd.concat(recent_parts, ignore_index=True).to_parquet(run/'recent-cells.parquet', index=False) if recent_parts else pd.DataFrame(columns=['target']).to_parquet(run/'recent-cells.parquet', index=False)
    return totals


def geographic_mean(group, fields):
    # Equal locations inside geography; 80/20 when both groups are present.
    locations = group.groupby(['geography','location'])[fields].mean().reset_index()
    geo = locations.groupby('geography')[fields].mean()
    weights = pd.Series({'states_dc':.8,'US':.2}).reindex(geo.index)
    return (geo.mul(weights,axis=0).sum(min_count=1)/weights.sum()).to_dict()


def markdown(table):
    rows = ['| ' + ' | '.join(table.columns) + ' |', '| ' + ' | '.join(['---'] * len(table.columns)) + ' |']
    for row in table.itertuples(index=False, name=None):
        rows.append('| ' + ' | '.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in row) + ' |')
    return '\n'.join(rows)


def report(runs, destination, experiment):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination = Path(destination)
    page = Path('docs/results')/experiment
    page.mkdir(parents=True, exist_ok=True)
    rankings = pd.read_csv(destination/'configuration_ranking.csv')
    labels = {r['config_id']:json.loads((Path(r['path'])/'manifest.json').read_text())['candidate'] for r in runs}
    rankings['candidate'] = rankings.config_id.map(labels)
    forecast, recent = [], []
    for run in runs:
        for filename, bucket in [('full-forecast-cells.parquet',forecast),('recent-cells.parquet',recent)]:
            table = pd.read_parquet(Path(run['path'])/filename)
            if len(table):
                bucket.append(table.assign(candidate=labels[run['config_id']], seed=run['seed']))
    f = pd.concat(forecast, ignore_index=True)
    metrics = ['wis','scaled_wis',*[f'covered_{c}' for c in (50,80,90,95)]]
    rows = []
    for keys,g in f.groupby(['candidate','seed','target','season']):
        rows.append(dict(zip(['candidate','seed','target','season'],keys), n=len(g), **geographic_mean(g,metrics)))
    summary = pd.DataFrame(rows)
    summary.to_csv(page/'forecast-by-target-seed.csv',index=False)
    f.groupby(['candidate','seed','target','season','geography','location'])[metrics].mean().to_csv(page/'forecast-by-location.csv')
    f.groupby(['candidate','seed','target','season','geography','horizon'])[metrics].mean().to_csv(page/'forecast-by-horizon.csv')
    matched = pd.read_csv(destination/'season_scores.csv')
    matched['candidate'] = matched.config_id.map(labels)
    matched.to_csv(page/'matched-hub-scores.csv',index=False)
    fig, ax = plt.subplots(figsize=(11,5))
    matched[matched.geography == 'all'].groupby(['target','candidate']).wis_ratio.mean().unstack().plot.bar(ax=ax)
    ax.axhline(1,color='black',ls='--');ax.set_ylabel('Matched Hub-relative WIS');ax.tick_params(axis='x',labelsize=8)
    fig.tight_layout();fig.savefig(page/'targets.png',dpi=150);plt.close(fig)
    rankings.to_csv(page/'ranking.csv',index=False)
    fig, ax = plt.subplots(figsize=(7,4))
    ax.bar(rankings.candidate,rankings.combined_mean,yerr=rankings.combined_sd.fillna(0),capsize=5)
    ax.axhline(1,color='black',ls='--');ax.set_ylabel('Shared location-relative WIS (mean ± seed SD)')
    fig.tight_layout();fig.savefig(page/'ranking.png',dpi=150);plt.close(fig)
    fig, axes=plt.subplots(2,3,figsize=(13,7))
    for (target,g),ax in zip(summary.groupby('target'),axes.flat):
        for candidate,p in g.groupby('candidate'):
            ax.plot([50,80,90,95],[100*p[f'covered_{c}'].mean() for c in (50,80,90,95)],marker='o',label=candidate)
        ax.plot([50,95],[50,95],color='black',ls='--');ax.set_title(target.replace('wk inc ',''));ax.legend(fontsize=7)
        ax.set_xlabel('Nominal coverage (%)');ax.set_ylabel('Observed coverage (%)')
    fig.tight_layout();fig.savefig(page/'coverage.png',dpi=150);plt.close(fig)
    recent_text = ''
    if recent:
        r = pd.concat(recent,ignore_index=True)
        fields = ['wis','ae_median','scaled_wis','scaled_ae','baseline_ae','baseline_scaled_ae',*[f'covered_{c}' for c in (50,80,90,95)]]
        loc = r.groupby(['candidate','seed','target','season','kind','age_days','geography','location'])[fields].mean().reset_index()
        loc['n'] = r.groupby(['candidate','seed','target','season','kind','age_days','geography','location']).size().to_numpy()
        loc['denominator_near_zero'] = loc.kind.eq('revision') & (loc.baseline_scaled_ae <= DENOMINATOR_THRESHOLD)
        eligible = loc.kind.eq('revision') & ~loc.denominator_near_zero
        loc['mae_ratio'] = loc.ae_median.div(loc.baseline_ae.where(eligible))
        loc['wis_ratio'] = loc.wis.div(loc.baseline_ae.where(eligible))
        loc.to_csv(page/'recent-by-location.csv',index=False)
        rows=[]
        for keys,g in r.groupby(['candidate','seed','target','kind','age_days']):
            rows.append(dict(zip(['candidate','seed','target','kind','age_days'],keys),n=len(g),**geographic_mean(g,fields)))
        rs=pd.DataFrame(rows);rs.to_csv(page/'recent-summary.csv',index=False)
        fig,axes=plt.subplots(2,2,figsize=(12,7))
        for (kind,age),ax in zip([('revision',4),('revision',11),('reconstruction',4),('reconstruction',11)],axes.flat):
            part=rs[(rs.kind==kind)&(rs.age_days==age)]
            pivot=part.groupby(['target','candidate']).scaled_ae.mean().unstack()
            if kind == 'revision':
                pivot['own preliminary'] = part.groupby('target').baseline_scaled_ae.mean()
            pivot.plot.bar(ax=ax);ax.set_title(f'{kind}, {age} days');ax.set_ylabel('MAE / training Q95');ax.tick_params(axis='x',labelsize=6)
        fig.tight_layout();fig.savefig(page/'recent.png',dpi=150);plt.close(fig)
        recent_text = f'\n## Recent predictions\n\n{int(loc.denominator_near_zero.sum())} candidate/seed/target/age/location ratios were undefined under the predeclared threshold (mean preliminary absolute error / training Q95 ≤ {DENOMINATOR_THRESHOLD:g}). These are not assigned zero or epsilon denominators. No revision ratio exists for missing reports.\n\n![Recent scaled errors](recent.png)\n\n[Recent diagnostics](recent-summary.csv) · [Location diagnostics and denominator flags](recent-by-location.csv)\n'
    if experiment.endswith('-check'):
        recent_text += '\n**Execution check only: one epoch. Do not interpret these scores as benchmark conclusions.**\n'
    table = markdown(rankings[['candidate','seeds','combined_mean','combined_sd','states_dc_combined_mean','US_combined_mean']])
    target_table = markdown(summary.groupby(['candidate','target'])[['wis','scaled_wis','covered_50','covered_95']].mean().reset_index())
    (page/'index.md').write_text(f'''# Frozen forward development benchmark: 2025–26

Training cutoff: July 26, 2025 end of day UTC; mature labels through June 28. Test Wednesdays July 30, 2025–July 29, 2026; forecast targets August 2, 2025–August 1, 2026. Parameters frozen. This season was previously explored; this is **forward development**, not an untouched final test.

[Experiment specification](../../workflows/forward-2025.md) · [Data assumptions](../../data/forward-2025.md)

## Matched Hub-supported forecast tasks

{table}

![Forecast ranking](ranking.png)

![Matched target results](targets.png)

Mean and sample SD describe fitting variability only. One season provides no independent season replication; seeds and locations are not independent seasons. Native WIS cannot be compared across targets' units. Hub support varies by target and is a subset of the full test period. [Matched target/season/geography scores](matched-hub-scores.csv).

## Full forward-period diagnostics

{target_table}

All candidates share identical available forecast labels. Tables use equal locations within states/DC (80%) and US (20%). Target-native WIS, stable training-Q95-scaled WIS and coverage are reported separately from the Hub-relative ranking.

![Coverage](coverage.png)

[Target and seed diagnostics](forecast-by-target-seed.csv) · [Location diagnostics](forecast-by-location.csv) · [Horizon diagnostics](forecast-by-horizon.csv)
{recent_text}

Archive dates are accepted availability proxies, with assumed interior completeness; strict provider-publication availability is not independently certified. Finalized values are 28-day-mature cutoff proxies. Four-day NSSP visible correction training begins only June 18, 2025. Older-history fitting uses cutoff-final proxies; deployment uses Wednesday reports. The separate pipeline additionally learns forecasting on exact cutoff-final recent inputs and deploys on estimates; sampling propagates uncertainty but does not eliminate that mismatch. No calibration, artificial masking or test-driven epoch selection was used.
''')
    if (page/'analysis.md').exists():
        with (page/'index.md').open('a') as stream:
            stream.write('\n[**Analysis, ranking against B1, fan plots and heatmaps**](analysis.md)\n')
    import hashlib
    save(page/'report-provenance.json', dict(experiment=experiment,
        report_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runs=[dict(candidate=labels[r['config_id']], seed=r['seed'], path=str(r['path'])) for r in runs],
        denominator_threshold=DENOMINATOR_THRESHOLD))
    print(f'Results page: {page}/index.md',flush=True)
