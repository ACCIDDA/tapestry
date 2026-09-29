"""Count finite publisher statements separately from conflicting-value rejection."""
from pathlib import Path
import numpy as np,pandas as pd
from tapestry.dataset import extract as ex
from tapestry.dataset.build import load
from tapestry.dataset.cv import season
out=Path('docs/data/availability')
p=load('data/processed/panel-b2-deadline.npz');iss=pd.to_datetime(p['issuance_dates'])
base=pd.read_csv(out/'cells.csv.gz');base=base[base.covariate.isin(ex.DELPHI_COVARIATES)].copy()
original=ex._without_conflicts
for name in ex.DELPHI_COVARIATES:
 def capture(rows):
  keys=['tier','release','day','location']
  g=rows.groupby(keys,sort=False).value.agg(['min','max','count','size']).reset_index()
  g=g[(g.day>='2025-05-01')&(g.day<='2026-08-01')].copy()
  g.to_pickle('/tmp/tapestry-'+name+'-statements.pkl')
  return original(rows)
 ex._without_conflicts=capture
 ex.revisions(name,'data')
 g=pd.read_pickle('/tmp/tapestry-'+name+'-statements.pkl')
 release=pd.to_datetime(g.release,utc=True);midnight=release.eq(release.dt.normalize());g['release']=release+pd.to_timedelta(midnight.astype(int),unit='D')-pd.to_timedelta(midnight.astype(int),unit='ns')
 chunks=[]
 for issuance,c in base[base.covariate.eq(name)].groupby('issuance',sort=False):
  eligible=g[g.release<=pd.Timestamp(c.cutoff.iloc[0])].sort_values('release').drop_duplicates(['day','location'],keep='last')
  z=c.merge(eligible[['day','location','min','max','count','size']],left_on=['observation_week','location'],right_on=['day','location'],how='left')
  z['raw_finite']=z['count'].fillna(0).gt(0)
  z['conflicting']=z.raw_finite & (z['min'].ne(z['max']) | z['count'].ne(z['size']))
  z['mixed_null']=z.raw_finite & z['count'].ne(z['size'])
  chunks.append(z)
 result=pd.concat(chunks);result.to_csv(out/(name+'-statements.csv.gz'),index=False)
 print(name,'lag1',result[result.lag.eq(1)][['available','raw_finite','conflicting','mixed_null']].sum().to_dict(),flush=True)
ex._without_conflicts=original
frames=[pd.read_csv(out/(n+'-statements.csv.gz')) for n in ex.DELPHI_COVARIATES]
x=pd.concat(frames);s=x.groupby(['covariate','lag']).agg(opportunities=('available','size'),pipeline_available=('available','sum'),raw_finite=('raw_finite','sum'),conflicting=('conflicting','sum'),mixed_null=('mixed_null','sum')).reset_index();s.to_csv(out/'conflict-audit.csv',index=False)
