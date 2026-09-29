"""Reporting availability on actual scored target/location/issuance support."""
from pathlib import Path
import numpy as np
import pandas as pd
from tapestry.dataset.build import load
from tapestry.evaluation.totals import frozen_cases
from tapestry.evaluation.hubs import CHANNEL
from tapestry.data.geography import STATE_FIPS
p=load('data/processed/panel.npz')
wi={str(d):i for i,d in enumerate(p['issuance_dates'])}
ti={str(d):i for i,d in enumerate(p['dates'])}
li={str(d):i for i,d in enumerate(p['locations'])}
rows=[]
root=Path('data/evaluation/b0_hub_comparison_q23')
for case in frozen_cases(root):
    u=pd.read_parquet(root/case['directory']/'units.parquet')
    u=u[['reference_date','location']].drop_duplicates()
    for row in u.itertuples():
        reference=pd.Timestamp(row.reference_date)
        issuance=(reference-pd.Timedelta(days=3)).date().isoformat()
        loc='US' if str(row.location)=='US' else STATE_FIPS[str(row.location).zfill(2)]
        for lag in [0,1]:
            day=(reference-pd.Timedelta(days=7+7*lag)).date().isoformat()
            w,t,l,c=wi[issuance],ti[day],li[loc],CHANNEL[case['target']]
            final=np.isfinite(p['targets'][t,l,c]);seen=np.isfinite(p['asof_targets'][w,t,l,c])
            rows.append(dict(season=case['season'],target=case['target'],issuance=issuance,location=loc,
                lag=lag,final=int(final),seen=int(seen),missing_with_final=int(final and not seen)))
d=pd.DataFrame(rows)
a=d.groupby(['season','target','lag'],as_index=False)[['final','seen','missing_with_final']].sum()
a['percent_missing_of_final']=100*a.missing_with_final/a.final
out=Path('docs/results/forecast-geography-v2')
a.to_csv(out/'scored_target_availability.csv',index=False)
d.to_csv(out/'scored_target_availability_cells.csv',index=False)
print(a.to_string(index=False))
