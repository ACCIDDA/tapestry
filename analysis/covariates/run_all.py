"""Evaluate every candidate Delphi covariate under one protocol. See covsources.py."""
import pandas as pd
import covsources as C
import ww_asarchive as W

SOURCES = [
    ('claims_inpatient',  'flu',   'cov/claims_inpatient_adm_pct_claims_flu.csv',    'chunk'),
    ('claims_inpatient',  'covid', 'cov/claims_inpatient_adm_pct_claims_covid.csv',  'chunk'),
    ('claims_outpatient', 'flu',   'cov/claims_outpatient_ov_pct_claims_flu.csv',    'chunk'),
    ('claims_outpatient', 'covid', 'cov/claims_outpatient_ov_pct_claims_covid.csv',  'chunk'),
    ('pophive_ed',        'flu',   'cov/pophive_flu_pct_ed.csv',                     'pophive'),
    ('pophive_ed',        'covid', 'cov/pophive_covid_pct_ed.csv',                   'pophive'),
    ('pophive_ed',        'rsv',   'cov/pophive_rsv_pct_ed.csv',                     'pophive'),
]

rows, lat = [], []
for label, pathogen, path, how in SOURCES:
    cov = (C.load_source_chunked(path) if how == 'chunk'
           else C.load_source(path, age_all=True, weekly_only=False))
    cov = cov[cov['reference_time'] > '2024-01-01']
    window = '2025-07-01' if how == 'pophive' else '2024-11-01'
    k, share, med = C.choose_lag(cov[cov['reference_time'] > window])
    lat.append({'source': label, 'pathogen': pathogen, 'median_latency_d': med,
                'vintages': cov['report_time'].nunique(), 'lag_wk': k,
                'available_at_origin': round(share, 3)})
    rows.append(C.evaluate(pathogen, cov, k, label))

for pathogen in ['flu', 'covid', 'rsv']:
    cov = W.wastewater_archive(pathogen)
    cov = cov[cov['reference_time'] > '2024-01-01']
    lat.append({'source': 'wastewater', 'pathogen': pathogen, 'median_latency_d': 11,
                'vintages': None, 'lag_wk': 2, 'available_at_origin': 0.90})
    rows.append(C.evaluate(pathogen, cov, 2, 'wastewater', log_cov=False))

res = pd.concat(rows, ignore_index=True)
res.to_csv('covariate_results.csv', index=False)
pd.DataFrame(lat).to_csv('covariate_latency.csv', index=False)
print(pd.DataFrame(lat).to_string(index=False))
print()
print(res.round(3).to_string(index=False))
