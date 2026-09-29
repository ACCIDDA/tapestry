"""Supporting measurements for the consecutive comparison, without fitting anything."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from tapestry.experiment.planner import read_jobs,seed_state
from tapestry.dataset.build import load
from tapestry.dataset.cv import season

out=Path('docs/results/b1-to-b0-chain')
labels=['00','01','02','03','03b','04','05','06','07','09']
rows=[]
for stage in labels:
 root=Path('data/experiments')/f'b1-b0-score-{stage}'
 protocol=json.loads((root/'reforecast-protocol.json').read_text())
 source=Path('data/experiments')/protocol['source_experiment']
 for seed in [42,43,44]:
  attempt,_,_=seed_state(source,protocol['source_scenario'],seed)
  if attempt is None:continue
  for model in sorted(attempt.glob('eval_*/model.pt')):
   m=torch.load(model,map_location='cpu',weights_only=False)['metadata']
   for record in m.get('records',[]):
    if record.get('phase')!='select':continue
    rows.append(dict(stage=stage,seed=seed,season=model.parent.name[5:],channels=str(record.get('channels')),epoch=record.get('selected_epoch',record.get('best_epoch'))))
pd.DataFrame(rows).to_csv(out/'selected_epochs.csv',index=False)
# How much finalized historical information does the presence restriction remove?
p=load('data/audits/b0/original-panel-deadline2025.npz')
from datetime import date,timedelta
wi={str(d):i for i,d in enumerate(p['issuance_dates'])}
rows=[]
for t,day in enumerate(p['dates']):
 cutoff=(date.fromisoformat(str(day))+timedelta(days=4)).isoformat()
 if cutoff not in wi:continue
 for lag in range(12):
  if t-lag<0:continue
  valid=np.isfinite(p['targets'][t-lag])
  present=np.isfinite(p['asof_targets'][wi[cutoff],t-lag])&valid
  for c,target in enumerate(['Flu admissions','COVID admissions','RSV admissions','Flu ED','COVID ED','RSV ED']):
   rows.append(dict(season=season(day),target=target,lag=lag,eligible=int(valid[:,c].sum()),present=int(present[:,c].sum())))
f=pd.DataFrame(rows).groupby(['season','target','lag'])[['eligible','present']].sum().reset_index()
f['percent_present']=100*f.present/f.eligible
f.to_csv(out/'historical_input_presence.csv',index=False)
print('Wrote selected epochs and historical input presence.')
