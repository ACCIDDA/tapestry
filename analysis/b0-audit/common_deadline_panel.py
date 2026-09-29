"""One shared forecast issuance must not mix inputs from different Hub deadlines.

For the two extended rounds, use the earliest participating Hub deadline for ALL
six inputs. Per-Hub missingness is still reported separately in deadline tables.
"""
from datetime import date,timedelta,timezone
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from tapestry.dataset.build import load,save
from deadline_availability import HUBS,CHANNELS,deadline,state,table,delphi_reports

source=Path('data/audits/b0/original-panel-deadline2025.npz');panel=load(source)
delphi=delphi_reports();rows=[]
for reference in ['2025-12-27','2026-01-03']:
 ref=date.fromisoformat(reference);cut=min(deadline(ref,h) for h in HUBS);utc=cut.astimezone(timezone.utc).isoformat()
 issuance=(ref-timedelta(days=3)).isoformat();w=list(panel['issuance_dates'].astype(str)).index(issuance);end=ref-timedelta(days=7)
 context=[(end-timedelta(weeks=i)).isoformat() for i in range(12)];observed={}
 for hub in HUBS:
  commit=state(hub,utc)
  paths=('target-data/time-series.csv','target-data/target-hospital-admissions.csv','target-data/target-ed-visits-prop.csv') if hub=='flusight' else ('target-data/time-series.parquet','auxiliary-data/nssp-raw-data/latest.parquet','auxiliary-data/nssp-raw-data/latest.csv')
  for path in paths:
   frame,blob=table(hub,commit,path)
   if frame is None:continue
   for row in frame.loc[frame.week.isin(context)].itertuples(index=False):
    observed.setdefault((row.week,row.location,row.target),(row.value,hub,commit,path))
 available=delphi.loc[delphi.week.isin(context)&(delphi.release<=pd.Timestamp(cut))].sort_values('release').drop_duplicates(['week','location','target'],keep='last')
 for row in available.loc[lambda f:f.value.notna()&(f.value>=0)].itertuples(index=False):
  observed.setdefault((row.week,row.location,row.target),(row.value,'delphi_archive',row.release.isoformat(),row.path))
 for week in context:
  t=list(panel['dates'].astype(str)).index(week)
  for c,target in enumerate(CHANNELS):
   for l,loc in enumerate(panel['locations']):
    found=observed.get((week,str(loc),target));before=bool(np.isfinite(panel['asof_targets'][w,t,l,c]))
    panel['asof_targets'][w,t,l,c]=np.nan if found is None else found[0]
    rows.append(dict(reference_date=reference,common_deadline=cut.isoformat(),observation_week=week,target=target,location=str(loc),per_hub_present=before,common_present=found is not None,source='' if found is None else found[1],source_version='' if found is None else found[2]))
metadata=json.loads(str(panel['metadata']));metadata.update(purpose='Original B0 values; corrected2025 availability with one shared safe issuance cutoff',common_deadline='Earliest participating Hub deadline for every input channel; Christmas Dec29 23:00 ET, NewYear Jan4 23:00 ET',source_panel_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
panel['metadata']=json.dumps(metadata)
out=Path('data/audits/b0/original-panel-commondeadline2025.npz');save(panel,out)
records=pd.DataFrame(rows);records.to_csv('docs/results/b1-to-b0-chain/common_deadline_holiday_cells.csv',index=False)
metadata['output_sha256']=hashlib.sha256(out.read_bytes()).hexdigest();Path('docs/results/b1-to-b0-chain/common_deadline_policy.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Visibility changes from per-Hub cutoffs:',int((records.per_hub_present!=records.common_present).sum()))
print(out)
