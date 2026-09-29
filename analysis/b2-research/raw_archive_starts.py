from pathlib import Path
import pandas as pd,numpy as np
from tapestry.dataset import extract as ex
rows=[]
def scan(name,files,valuecol='value'):
 firstreport=firstobs=firstrow=None;count=0
 for path in files:
  for x in ex._read(path,path.name,['report_time','reference_time',valuecol]):
   rt=x.report_time.astype(str).str[:10];dt=x.reference_time.astype(str).str[:10]
   gooddate=rt.str.match(r'^\d{4}-\d{2}-\d{2}$')
   if gooddate.any():firstrow=min(firstrow or '9999',rt[gooddate].min())
   good=gooddate & dt.str.match(r'^\d{4}-\d{2}-\d{2}$') & np.isfinite(pd.to_numeric(x[valuecol],errors='coerce'))
   if good.any():firstreport=min(firstreport or '9999',rt[good].min());firstobs=min(firstobs or '9999',dt[good].min());count+=int(good.sum())
 row=dict(covariate=name,earliest_raw_report=firstrow,earliest_finite_report=firstreport,earliest_observation=firstobs,finite_raw_rows=count,files=len(files));rows.append(row);print(row,flush=True)
for name,(dataset,signal) in ex.DELPHI_COVARIATES.items():
 root=ex._latest_snapshot('data',dataset);scan(name,list(root.glob('signal='+signal+'/geo_type=*/archive.csv.gz')))
scan('kinsa_ili',[ex._latest_snapshot('data',ex.KINSA_DATASET)/'archive.csv.gz'],ex.KINSA_SIGNAL)
p=ex._latest_snapshot('data',ex.NWSS_DERIVED_DATASET)/'data.csv.gz';x=pd.read_csv(p)
for pathogen,g in x.groupby('pathogen'):
 for metric in ['wval_like','pct_rank']:
  q=g[np.isfinite(pd.to_numeric(g[metric],errors='coerce'))];rows.append(dict(covariate='nwss_'+str(pathogen).lower()+'_'+metric,earliest_raw_report=g.report_time.astype(str).str[:10].min(),earliest_finite_report=q.report_time.astype(str).str[:10].min(),earliest_observation=q.reference_time.astype(str).str[:10].min(),finite_raw_rows=len(q),files=1))
out=Path('docs/data/availability/raw-archive-starts.csv');pd.DataFrame(rows).to_csv(out,index=False);print(pd.DataFrame(rows).to_string(index=False),flush=True)
