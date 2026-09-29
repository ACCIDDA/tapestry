"""Audit actual Git-visible target inputs at 2025–26 deadlines; pin a corrected panel.

Only 2025–26 reference rounds are reconstructed. Earlier seasons retain the saved
archive policy explicitly. Presence is unioned across the three public Hubs and
raw NSSP files; finalized numerical values are NOT inserted where no report exists.
"""
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import io, json, subprocess, hashlib
from functools import lru_cache
import numpy as np
import pandas as pd
from tapestry.dataset.build import load,save
from tapestry.dataset.cv import season

ROOT=Path('docs/results/b1-to-b0-chain'); ROOT.mkdir(parents=True,exist_ok=True)
CACHE=Path('data/audits/b0/deadline-cache-v2'); CACHE.mkdir(parents=True,exist_ok=True)
PANEL=Path('data/audits/b0/original-panel-unified.npz')
CHANNELS=['wk inc flu hosp','wk inc covid hosp','wk inc rsv hosp','wk inc flu prop ed visits','wk inc covid prop ed visits','wk inc rsv prop ed visits']
HUBS=['flusight','covid','rsv']
loc=pd.read_csv('data/metadata/locations.csv',dtype={'location':str})
fips=dict(zip(loc.location,loc.abbreviation)); names=dict(zip(loc.abbreviation,loc.location_name))
geos=dict(zip(loc.location_name,loc.abbreviation));geos.update({'United States':'US','United States of America':'US','USA':'US'})


def git(hub,*args,check=True):
 r=subprocess.run(['git',f'--git-dir=data/mirrors/hub_{hub}_current.git',*args],capture_output=True)
 if check and r.returncode:raise RuntimeError(r.stderr.decode())
 return r.stdout


def deadline(ref,hub):
 day=ref-timedelta(days=3)
 if ref==date(2025,12,27):day=date(2025,12,30) if hub=='flusight' else date(2025,12,29)
 if ref==date(2026,1,3):day=date(2026,1,5) if hub=='flusight' else date(2026,1,4)
 return datetime.combine(day,time(23),ZoneInfo('America/New_York'))


@lru_cache(maxsize=None)
def state(hub,cutoff):
 return git(hub,'rev-list','--first-parent','-1',f'--before={cutoff}','HEAD').decode().strip()


def table(hub,commit,path):
 if not commit:return None,None
 blob=git(hub,'rev-parse',f'{commit}:{path}',check=False).decode().strip()
 if len(blob)!=40 or ':' in blob:return None,None
 cache=CACHE/f'{blob}.pkl'
 if cache.exists():return pd.read_pickle(cache),blob
 content=git(hub,'show',f'{commit}:{path}')
 raw=pd.read_parquet(io.BytesIO(content)) if path.endswith('parquet') else pd.read_csv(io.BytesIO(content),dtype={'location':str})
 if 'target_end_date' not in raw and 'date' in raw:raw=raw.rename(columns={'date':'target_end_date'})
 if path=='target-data/target-hospital-admissions.csv':raw=raw.rename(columns={'value':'observation'}).assign(target='wk inc flu hosp')
 if path=='target-data/target-ed-visits-prop.csv':raw=raw.rename(columns={'value':'observation'}).assign(target='wk inc flu prop ed visits')
 if 'as_of' in raw:raw=raw.sort_values('as_of').drop_duplicates(['target_end_date','location','target'],keep='last')
 if 'percent_visits_covid' in raw:
  raw=raw.loc[raw['county'].astype(str).str.lower().eq('all')].copy()
  raw['location']=raw.geography.map(geos)
  raw['week']=pd.to_datetime(raw.week_end).dt.strftime('%Y-%m-%d')
  frames=[]
  for pathogen,col in [('flu','percent_visits_influenza'),('covid','percent_visits_covid'),('rsv','percent_visits_rsv')]:
   if col not in raw:continue
   f=raw[['week','location']].copy();f['target']=f'wk inc {pathogen} prop ed visits';f['value']=pd.to_numeric(raw[col],errors='coerce')/100;frames.append(f)
  out=pd.concat(frames,ignore_index=True)
 elif {'target_end_date','target','observation','location'} <= set(raw):
  out=pd.DataFrame({'week':pd.to_datetime(raw.target_end_date).dt.strftime('%Y-%m-%d'),'location':raw.location.astype(str).map(fips),'target':raw.target.astype(str),'value':pd.to_numeric(raw.observation,errors='coerce')})
 else:raise ValueError((hub,path,raw.columns.tolist()))
 out=out.loc[out.location.notna() & out.value.notna() & out.target.isin(CHANNELS)].drop_duplicates(['week','location','target'])
 out.to_pickle(cache);return out,blob


def delphi_reports():
 frames=[]
 for c,signal in enumerate(['confirmed_admissions_flu_ew','confirmed_admissions_covid_ew','confirmed_admissions_rsv_ew','pct_ed_visits_influenza','pct_ed_visits_covid','pct_ed_visits_rsv']):
  provider='delphi_nhsn' if c<3 else 'delphi_nssp'
  snapshot=json.loads(Path(f'data/raw/{provider}/latest.json').read_text())['snapshot_id']
  for geo in ['state','nation']:
   path=Path(f'data/raw/{provider}/snapshots/{snapshot}/signal={signal}/geo_type={geo}/archive.csv.gz')
   raw=pd.read_csv(path,low_memory=False)
   if 'fill_method' in raw:raw=raw.loc[raw.fill_method.fillna('none').astype(str).str.lower().isin(['source','none','native'])]
   frame=pd.DataFrame({'week':raw.reference_time.astype(str).str[:10],'location':raw.geo_value.astype(str).str.upper(),'target':CHANNELS[c],'value':pd.to_numeric(raw.value,errors='coerce')/(100 if c>=3 else 1),'release':pd.to_datetime(raw.report_time,utc=True)+pd.Timedelta(days=1)-pd.Timedelta(microseconds=1),'path':str(path)})
   frames.append(frame)
 return pd.concat(frames,ignore_index=True)


def run():
 original=load(PANEL);panel={k:v.copy() if isinstance(v,np.ndarray) else v for k,v in original.items()}
 dates=panel['dates'].astype(str).tolist();locations=panel['locations'].astype(str).tolist();issuances=panel['issuance_dates'].astype(str).tolist()
 scored=pd.read_csv('docs/results/forecast-geography-v2/scored_target_availability_cells.csv')
 scored=set(map(tuple,scored.loc[(scored.season=='2025-2026') & (scored.lag==0),['target','issuance','location']].values))
 records=[];provenance=[];delphi=delphi_reports()
 for w,issuance in enumerate(issuances):
  nominal=date.fromisoformat(issuance[:10]);ref=nominal+timedelta(days=3);end=ref-timedelta(days=7)
  if season(end)!='2025-2026':continue
  context=[(end-timedelta(weeks=i)).isoformat() for i in range(12)]
  for h,target_hub in enumerate(HUBS):
   cut=deadline(ref,target_hub);utc=cut.astimezone(timezone.utc).isoformat();observed={}
   for source in HUBS:
    commit=state(source,utc)
    for path in (('target-data/time-series.csv','target-data/target-hospital-admissions.csv','target-data/target-ed-visits-prop.csv') if source=='flusight' else ('target-data/time-series.parquet','auxiliary-data/nssp-raw-data/latest.parquet','auxiliary-data/nssp-raw-data/latest.csv')):
     frame,blob=table(source,commit,path)
     if frame is None:continue
     cols=[CHANNELS[h],CHANNELS[h+3]]
     select=frame.loc[frame.week.isin(context)&frame.target.isin(cols)]
     for r in select.itertuples(index=False):
      key=(r.week,r.location,r.target)
      observed.setdefault(key,(r.value,source,commit,path))
     provenance.append(dict(reference_date=ref.isoformat(),target_hub=target_hub,deadline_local=cut.isoformat(),deadline_utc=utc,source_hub=source,commit=commit,path=path,blob=blob))
   git_observed=set(observed)
   available_delphi=delphi.loc[delphi.week.isin(context)&delphi.target.isin([CHANNELS[h],CHANNELS[h+3]])&(delphi.release<=pd.Timestamp(cut))].sort_values('release').drop_duplicates(['week','location','target'],keep='last')
   for row in available_delphi.loc[lambda f:f.value.notna() & (f.value>=0)].itertuples(index=False):
    observed.setdefault((row.week,row.location,row.target),(row.value,'delphi_archive',row.release.isoformat(),row.path))
   for c in (h,h+3):
    for lag,week in enumerate(context):
     if week not in dates:continue
     t=dates.index(week)
     for l,location in enumerate(locations):
      found=observed.get((week,location,CHANNELS[c]));final=bool(np.isfinite(panel['targets'][t,l,c]));before=bool(np.isfinite(panel['asof_targets'][w,t,l,c]))
      panel['asof_targets'][w,t,l,c]=np.nan if found is None else found[0]
      records.append(dict(reference_date=ref.isoformat(),nominal_wednesday=issuance,deadline_local=cut.isoformat(),observation_week=week,lag=lag,target=CHANNELS[c],location=location,location_name=names[location],present=found is not None,git_present=(week,location,CHANNELS[c]) in git_observed,final_present=final,previously_present=before,scored_origin=(CHANNELS[c],issuance,location) in scored,source_hub='' if found is None else found[1],source_commit='' if found is None else found[2],source_file='' if found is None else found[3]))
  print(ref,flush=True)
 frame=pd.DataFrame(records);frame.to_csv(ROOT/'deadline_context_cells.csv',index=False)
 latest=frame.loc[frame.lag==0];latest.to_csv(ROOT/'deadline_latest_cells.csv',index=False)
 missing=latest.loc[~latest.present];missing.to_csv(ROOT/'missing_latest_2025-2026.csv',index=False)
 frame.loc[~frame.present].to_csv(ROOT/'missing_context_2025-2026.csv',index=False)
 pd.DataFrame(provenance).drop_duplicates().to_csv(ROOT/'deadline_sources.csv',index=False)
 out=Path('data/audits/b0/original-panel-deadline2025.npz')
 metadata=dict(purpose='Original B0 finalized values; 2025–26 target availability reconstructed at Hub deadlines from Git',source_sha256=hashlib.sha256(PANEL.read_bytes()).hexdigest(),earlier_seasons='Unchanged saved Wednesday archive policy',numerical_inputs='Finalized values where report exists; not preliminary values',deadlines='23:00 America/New_York; Christmas flu Dec30, COVID/RSV Dec29; New Year flu Jan5, COVID/RSV Jan4',sources='Union three Hub series/direct target files, raw NSSP and dated Delphi reports; no inference of worldwide nonpublication from absence',unchanged='Finalized values, calendar, covariates, locations')
 panel['metadata']=json.dumps(metadata);save(panel,out)
 np.testing.assert_equal(panel['targets'],original['targets'])
 metadata['output_sha256']=hashlib.sha256(out.read_bytes()).hexdigest();(ROOT/'deadline_policy.json').write_text(json.dumps(metadata,indent=2)+'\n')
 print(latest.groupby('target').agg(cells=('present','size'),present=('present','sum')).to_string())
 print('Scored latest:',latest.loc[latest.scored_origin].groupby('target').agg(cells=('present','size'),present=('present','sum')).to_string())
 print(out,flush=True)

if __name__=='__main__':run()
