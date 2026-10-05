"""Saved best C1 forecasts against Google Hub submissions, matched task by task."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from tapestry.evaluation.hubs import export, KEY, QCOLS
from tapestry.evaluation.totals import quantile_scores, frozen_cases

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
FROZEN=ROOT/'data/evaluation/b0_hub_comparison_q23'
RUNS=ROOT/'data/analysis/overnight-google-20261005/forecasts'
ID='mlp-pathogen-scheduled_final-1ba45a22690c'
SEASON='2025-2026'
frames={s:export(RUNS/ID/f's{s}'/'attempt-001') for s in (42,43,44)}
rows=[];support=[];locations=[]
for case in frozen_cases(FROZEN):
 if case['season']!=SEASON: continue
 path=FROZEN/case['directory']
 allq=pd.read_parquet(path/'quantiles.parquet')
 names=[m for m in allq.model.unique() if m.startswith('Google')]
 if not names:continue
 assert len(names)==1
 google=allq[allq.model==names[0]]
 units=pd.read_parquet(path/'units.parquet')
 common=units[KEY+['observed']].merge(google[KEY+QCOLS],on=KEY,validate='one_to_one').sort_values(KEY).reset_index(drop=True)
 ensemble=allq[allq.model==case['ensemble']]
 for scope in ['frozen_all','frozen_states','cdc_flu_window_states']:
  if scope.startswith('cdc') and case['target']!='wk inc flu hosp':continue
  c=common.copy()
  if scope!='frozen_all':c=c[c.location!='US']
  if scope.startswith('cdc'):c=c[(c.reference_date>='2025-11-22')&(c.reference_date<='2026-05-23')]
  c=c.sort_values(KEY).reset_index(drop=True)
  assert len(c)>0 and not c.duplicated(KEY).any()
  support.append(dict(target=case['target'],google=names[0],scope=scope,n=len(c),full_frozen_tasks=len(units),dates=c.reference_date.nunique(),locations=c.location.nunique(),first=c.reference_date.min(),last=c.reference_date.max(),target_first=c.target_end_date.min(),target_last=c.target_end_date.max()))
  eq=c[KEY+['observed']].merge(ensemble[KEY+QCOLS],on=KEY,validate='one_to_one').sort_values(KEY).reset_index(drop=True)
  assert c[KEY].equals(eq[KEY])
  for seed in (42,43,44):
   ours=c[KEY+['observed']].merge(frames[seed][(SEASON,case['target'])][KEY+QCOLS],on=KEY,validate='one_to_one').sort_values(KEY).reset_index(drop=True)
   assert c[KEY].equals(ours[KEY])
   for scale,offset in [('native',None),('log1p_sensitivity',1.),('log_half_sensitivity',.5)]:
    y=c.observed.to_numpy(); gs=c[QCOLS].to_numpy();ts=ours[QCOLS].to_numpy();es=eq[QCOLS].to_numpy()
    if offset is not None:y,gs,ts,es=[np.log(x+offset) for x in (y,gs,ts,es)]
    scores=c[KEY].copy()
    for label,q in [('tapestry',ts),('google',gs),('ensemble',es)]:
     assert np.isfinite(q).all()
     scores[label]=quantile_scores(q,y).wis.to_numpy()
    loc=scores.groupby('location')[['tapestry','google','ensemble']].sum()
    assert (loc[['google','ensemble']]>0).all().all()
    us=loc.index=='US';w=np.where(us,.2,.8/max((~us).sum(),1));w=w/w.sum()
    tr=float(np.dot(w,loc.tapestry/loc.ensemble));gr=float(np.dot(w,loc.google/loc.ensemble))
    rows.append(dict(target=case['target'],google_model=names[0],scope=scope,scale=scale,seed=seed,n=len(c),tapestry_to_ensemble=tr,google_to_ensemble=gr,change_percent=100*(tr/gr-1),tapestry_to_google_location_mean=float(np.dot(w,loc.tapestry/loc.google)),pooled_tapestry_wis=scores.tapestry.mean(),pooled_google_wis=scores.google.mean(),pooled_ratio=scores.tapestry.mean()/scores.google.mean(),locations_won=int((loc.tapestry<loc.google).sum()),locations=len(loc)))
    if scale=='native':
     locations.append(loc.reset_index().assign(target=case['target'],scope=scope,seed=seed))
    if seed==42 and scope=='frozen_all' and scale=='native':
     c.to_parquet(OUT/(case['directory']+'-google-matched.parquet'),index=False)
results=pd.DataFrame(rows);results.to_csv(OUT/'per-seed.csv',index=False)
pd.DataFrame(support).to_csv(OUT/'support.csv',index=False)
pd.concat(locations).to_csv(OUT/'native-location-scores.csv',index=False)
summary=results.groupby(['target','google_model','scope','scale']).agg(tapestry_to_ensemble=('tapestry_to_ensemble','mean'),google_to_ensemble=('google_to_ensemble','mean'),change_percent=('change_percent','mean'),seed_min_change=('change_percent','min'),seed_max_change=('change_percent','max'),pooled_ratio=('pooled_ratio','mean'),seeds_won=('change_percent',lambda x:int((x<0).sum())),n=('n','first')).reset_index()
summary.to_csv(OUT/'summary.csv',index=False)
meta=json.loads((FROZEN/'manifest.json').read_text())
(OUT/'provenance.json').write_text(json.dumps(dict(model_id=ID,training_seasons=['2022-2023','2023-2024','2024-2025'],error_donor='2024-2025',evaluation=SEASON,source_hubs={k:{a:v[a] for a in ['commit','url','truth_vintages']} for k,v in meta['hubs'].items()},forecast_hashes={str(p.relative_to(RUNS)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (RUNS/ID).glob('s*/attempt-001/eval_2025-2026/forecasts.npz')},scoring='Native: existing Tapestry location-relative Hub-ensemble WIS, US20%/states80%, equal state weighting, on Google intersection. Log offsets are sensitivities, NOT official CDC replication.'),indent=2))
print(summary.to_string(index=False))
print(pd.DataFrame(support).to_string(index=False))
