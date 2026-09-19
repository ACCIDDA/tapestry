"""Aggregate existing per-cell scores; no fitting, predictions or new model scoring."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
sys.path.insert(0,'scripts')
from plot_b1_revisions import parse, nowcast_locations, nowcast_scores, nowcast_runs
OUT=Path('docs/results/b1-overnight/revisions')
root=Path('data/experiments/B1-revisions-20260917')
ranking=next(root.glob('ranking-*'))
runs=json.loads((ranking/'manifest.json').read_text())['runs']
columns=['stress','recent_kind','target','season','location','horizon','observed','loss_scale','crps','wis','coverage_50','coverage_95','median_error','ae_median','report_ae','issuance_date','target_date']
parts=[];paired=[];outages=[]
for run in runs:
 if parse(run['name'])[1] not in ('C','two_stage','gated'):continue
 for path in Path(run['path']).glob('eval_*/scores-*.parquet'):
  f=pd.read_parquet(path,columns=columns,filters=[('task','==','nowcast')])
  f['scaled_crps']=f.crps/f.loss_scale
  f['scaled_mae']=f.ae_median/f.loss_scale
  f['scaled_report_ae']=f.report_ae/f.loss_scale
  f['scaled_zero_crps']=f.observed.abs()/f.loss_scale
  f['median']=f.observed+f.median_error
  f['near_zero_median']=(f['median'].abs()<=1e-8).astype(float)
  keys=['stress','recent_kind','target','season','location','horizon']
  g=f.groupby(keys,dropna=False)[['scaled_crps','scaled_mae','scaled_report_ae','coverage_50','coverage_95','scaled_zero_crps','near_zero_median']].mean().reset_index()
  g['name']=run['name'];g['seed']=run['seed'];parts.append(g)
  # Exact matched report cells: natural visible report versus same cell hidden by stress.
  key=['issuance_date','target_date','target','location','horizon']
  natural=f[f.stress.eq('natural')&f.recent_kind.eq('revision')]
  for stress in ['recent','outage']:
   hidden=f[f.stress.eq(stress)&f.recent_kind.eq('artificial_reconstruction')]
   q=hidden.merge(natural,on=key,suffixes=('_hidden','_visible'),validate='one_to_one')
   for side in ['hidden','visible']:
    z=q[key+['season_'+side,'scaled_crps_'+side,'coverage_50_'+side,'coverage_95_'+side]].rename(columns={v+'_'+side:v for v in ['season','scaled_crps','coverage_50','coverage_95']})
    z=z.groupby(['target','season','location','horizon'])[['scaled_crps','coverage_50','coverage_95']].mean().reset_index()
    z['stress']=stress;z['condition']=side;z['name']=run['name'];z['seed']=run['seed'];paired.append(z)
  o=f[f.stress.eq('outage')&f.recent_kind.eq('artificial_reconstruction')]
  for target,p in o.groupby('target'):
   outages.append(dict(name=run['name'],seed=run['seed'],season=p.season.iloc[0],target=target,n=len(p),median_abs=float(p['median'].abs().median()),zero_median_fraction=p.near_zero_median.mean(),crps=p.crps.mean(),zero_crps=p.observed.abs().mean()))
 print(run['name'],run['seed'],flush=True)
loc=pd.concat(parts,ignore_index=True)

def aggregate(f,extra,metrics):
 # Input rows are means within location / target / season / age, then aggregate
 # with scientific geography/target/season weights. Keep age separate throughout.
 base=['name','seed',*extra,'horizon']
 rows=[]
 for keys,g in f.groupby([*base,'season','target'],dropna=False):
  us=g.location.eq('US');w=np.where(us,.2/max(1,us.sum()),.8/max(1,(~us).sum()));w=w/w.sum()
  rows.append(dict(zip([*base,'season','target'],keys),**{m:float(np.dot(w,g[m])) for m in metrics}))
 a=pd.DataFrame(rows);rows=[]
 for keys,g in a.groupby([*base,'season'],dropna=False):
  w=np.where(g.target.str.startswith('nhsn'),1.,.5);w=w/w.sum()
  rows.append(dict(zip([*base,'season'],keys),**{m:float(np.dot(w,g[m])) for m in metrics}))
 a=pd.DataFrame(rows)
 return a.groupby(base,dropna=False)[metrics].mean().reset_index()
metrics=['scaled_crps','scaled_mae','scaled_report_ae','coverage_50','coverage_95','scaled_zero_crps','near_zero_median']
summary=aggregate(loc,['stress','recent_kind'],metrics)
summary.to_csv(OUT/'recent-scaled-run-diagnostics.csv',index=False)
p=aggregate(pd.concat(paired,ignore_index=True),['stress','condition'],['scaled_crps','coverage_50','coverage_95']);p.to_csv(OUT/'matched-reconstruction.csv',index=False)
pd.DataFrame(outages).to_csv(OUT/'outage-by-target.csv',index=False)
# Retain cutoff sensitivity rather than presenting one post-hoc cutoff as a fix.
l,excluded=nowcast_locations(runs)
rows=[]
for floor in [0,1e-12,1e-9,1e-8,1e-7,1e-6,1e-5]:
 baseline=l.groupby(['target','season','location']).ensemble_wis.max()
 dropped=set(baseline[baseline<floor].index)
 take=np.array([(t,s,c) not in dropped for t,s,c in zip(l.target,l.season,l.location)])
 a=nowcast_runs(nowcast_scores(l[take]),'wis_ratio')
 for name,g in a[a.geography.eq('all')].groupby('name'):
  rows.append(dict(floor=floor,dropped_groups=len(dropped),name=name,ratio=g.combined.mean()))
pd.DataFrame(rows).to_csv(OUT/'nowcast-threshold-sensitivity.csv',index=False)
b=l[['target','season','location','n','ensemble_wis']].drop_duplicates().sort_values('ensemble_wis')
b.head(15).to_csv(OUT/'small-nowcast-denominators.csv',index=False)
# Fair paired comparison to B, fixed backbone/augmentation/seed. Also controls.
f=pd.read_csv(OUT/'forecast-run-scores.csv');results=[]
for (backbone,rev,weight),g in f[f.form.eq('gated')].groupby(['backbone','rev','nw_weight']):
 b=f[f.form.eq('B')&f.backbone.eq(backbone)&f.rev.eq(rev)]
 q=g.merge(b,on='seed',suffixes=('_gated','_B'));d=q.combined_gated-q.combined_B
 results.append(dict(backbone=backbone,augmentation=rev,nowcast_weight=weight,B=q.combined_B.mean(),gated=q.combined_gated.mean(),delta=d.mean(),seed_delta_sd=d.std(),improved=int((d<0).sum()),pairs=len(d)))
pd.DataFrame(results).to_csv(OUT/'gated-vs-matched-B.csv',index=False)
print('DONE',flush=True)
