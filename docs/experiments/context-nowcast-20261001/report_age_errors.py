"""Paired age-specific errors against leaving the current vintage unchanged."""
from pathlib import Path
import numpy as np
import pandas as pd
from tapestry.dataset.build import load
from tapestry.experiment.finalization import reporting_support

ROOT = Path('data/experiments/context-nowcast-v6-20261001')
OUT = Path(__file__).parent
KEY = ['signal', 'issuance', 'location', 'age']
MODELS = {
    'Seasonal nowcast': 'mlp-all-vintaged-ed46d3e36e5f',
    'Balanced nowcast': 'mlp-all-vintaged-4498052c695b',
    'Full correction': 'mlp-all-vintaged-eb672d2cc4de',
}
runs = pd.read_csv(ROOT / 'runs.csv')
panel = load('data/processed/panel.npz')
records = []
for fold in ['2024-2025', '2025-2026']:
    frames = {}
    for label, run_id in MODELS.items():
        attempt = runs.loc[runs.name.eq(run_id), 'attempt'].iloc[0]
        f = pd.read_csv(ROOT / attempt / f'eval_{fold}/finalizations.csv.gz')
        frames[label] = f[f.age.isin([0, 1, 2])].sort_values(KEY).reset_index(drop=True)
    base = frames['Seasonal nowcast']
    for f in frames.values():
        assert base[KEY].equals(f[KEY])
        for column in ['truth', 'report', 'scale']:
            np.testing.assert_allclose(base[column], f[column], equal_nan=True)
    # Cohorts refer to the current issuance's newest-week history, shared across ages.
    support = reporting_support(panel, base[base.age.eq(0)])
    support = support[['signal', 'issuance', 'location', 'complete_history_12', 'uninterrupted_8']]
    base = base.merge(support, on=KEY[:-1], validate='many_to_one', how='left')
    for label, f in frames.items():
        assert base[KEY].equals(f[KEY])
        base[label] = f.prediction.to_numpy()
    base['Unchanged report'] = base.report
    base['geography'] = np.where(base.location.eq('US'), 'US', 'states')
    methods = ['Unchanged report', *MODELS]
    finite = np.isfinite(base[['truth', *methods]]).all(axis=1)
    # Require matched reports for all three ages, so age comparisons share issuances.
    eligible = base.assign(valid=finite).groupby(KEY[:-1]).valid.agg(['all', 'size'])
    eligible = eligible[eligible['all'] & eligible['size'].eq(3)].reset_index()[KEY[:-1]]
    base = base.merge(eligible, on=KEY[:-1], validate='many_to_one')
    for cohort in ['complete_history_12', 'complete_and_uninterrupted']:
        keep = base.complete_history_12
        if cohort == 'complete_and_uninterrupted':
            keep = keep & base.uninterrupted_8
        for (signal, geography, age), g in base[keep].groupby(['signal', 'geography', 'age']):
            for method in methods:
                error = g[method] - g.truth
                loc = pd.DataFrame(dict(location=g.location, ae=error.abs(), bias=error,
                    truth=g.truth, nae=error.abs()/g.scale)).groupby('location')
                means = loc.mean().mean()
                sums = loc.sum()
                records.append(dict(fold=fold, cohort=cohort, geography=geography,
                    signal=signal, age=age, method=method, cells=len(g),
                    mae=means.ae, bias=means.bias, normalized_mae=means.nae,
                    wape=error.abs().sum()/g.truth.sum(),
                    location_wape=(sums.ae/sums.truth.replace(0,np.nan)).mean()))
detail = pd.DataFrame(records)
detail.to_csv(OUT/'age-error-by-target.csv', index=False)
summary = detail.groupby(['fold','cohort','geography','age','method'],as_index=False).agg(
    wape=('wape','mean'), location_wape=('location_wape','mean'),
    normalized_mae=('normalized_mae','mean'), cells=('cells','sum'))
summary.to_csv(OUT/'age-error-summary.csv',index=False)
recent = summary[summary.fold.eq('2025-2026')]
print((100*recent.pivot(index=['cohort','geography','age'],columns='method',values='wape')).round(3).to_string())
print('\nNative-unit mean absolute errors (ED proportions multiplied by 100 = percentage points):')
native = detail[detail.fold.eq('2025-2026') & detail.cohort.eq('complete_history_12') & detail.geography.eq('states')].copy()
native.loc[native.signal.str.startswith('nssp'), ['mae', 'bias']] *= 100
print(native.pivot(index=['signal','age'],columns='method',values='mae').round(5).to_string())
