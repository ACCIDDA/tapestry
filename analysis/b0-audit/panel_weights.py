"""Compare finalized panels and the independent-fit training weight contracts."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
from tapestry.dataset.build import load
from tapestry.dataset.cv import fold,season
from tapestry.model.scenario import Scenario
from tapestry.model.objective import loss_cell_weights
out=Path('docs/results/b0-audit');out.mkdir(parents=True,exist_ok=True)
old=np.load('data/processed/build_b_finalized.npz');new=load('data/processed/panel.npz')
idx=[list(new['dates'].astype(str)).index(d) for d in old['dates']]
loc=[list(new['locations']).index(l) for l in old['locations']]
a=np.where(old['panel'][:,:,1].astype(bool),old['panel'][:,:,0],np.nan)
b=new['targets'][idx][:,loc].transpose(0,2,1)
rows=[]
for c,name in enumerate(new['target_names']):
 for s in ('2023-2024','2024-2025','2025-2026'):
  take=np.array([season(d)==s for d in old['dates']]);x,y=a[take,c],b[take,c];both=np.isfinite(x)&np.isfinite(y)
  rows.append(dict(target=name,season=s,old_n=int(np.isfinite(x).sum()),new_n=int(np.isfinite(y).sum()),common=int(both.sum()),changed=int((~np.isclose(x[both],y[both],rtol=1e-6,atol=1e-8)).sum()),mae=float(np.abs(x[both]-y[both]).mean()),max_abs=float(np.abs(x[both]-y[both]).max())))
pd.DataFrame(rows).to_csv(out/'panel_comparison.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
rows=[]
for held in ('2023-2024','2024-2025','2025-2026'):
 eps=fold(new,Scenario(ed_transform='logit',patience=30),held).train
 dates=np.array([[season(d) for d in e['target_dates']] for e in eps])
 for label,channels in [('flu',[0,3]),('covid',[1,4]),('rsv',[2,5])]:
  current=loss_cell_weights(eps)[:,:,channels]
  selected=[]
  for e in eps:
   y=e['Y'].copy();y[:,[c for c in range(6) if c not in channels]]=0
   selected.append(dict(e,Y=y))
  legacy=loss_cell_weights(selected)[:,:,channels]
  for s in np.unique(dates):
   take=dates==s
   if current[take].sum() or legacy[take].sum():
    rows.append(dict(held=held,component=label,season=s,current_mass=float(current[take].sum()),b0_mass=float(legacy[take].sum()),current_season_share=float(current[take].sum()/current.sum()),b0_season_share=float(legacy[take].sum()/legacy.sum())))
pd.DataFrame(rows).to_csv(out/'training_weight_comparison.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
