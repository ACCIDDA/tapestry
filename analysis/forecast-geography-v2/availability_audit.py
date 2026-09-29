"""Verify retrospective finalized-input availability, including unpublished wastewater."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tapestry.dataset.build import load, covariate_names_for
from tapestry.dataset.episodes import episodes

p=load('data/processed/panel.npz')
names=covariate_names_for('ww_wval_like')
es=episodes(p,12,'finalized_available',names)
final={e['context_dates'][-1]:e for e in episodes(p,12,'finalized',names)}
rows=[];examples=[]
for e in es:
    if e['context_dates'][-1] not in final:continue
    truth=final[e['context_dates'][-1]]
    for lag in (0,1):
        i=-1-lag
        for c,name in enumerate(p['target_names']):
            known=truth['available'][i,c]
            rows.append(dict(kind='target',name=str(name),lag_weeks=lag,finite_final=int(known.sum()),masked_by_cutoff=int((known&~e['available'][i,c]).sum())))
        for c,name in enumerate(names):
            known=truth['covariates'][i,c,1].astype(bool)
            seen=e['covariates'][i,c,1].astype(bool)
            assert not (seen & ~known).any()
            rows.append(dict(kind='covariate',name=name,lag_weeks=lag,finite_final=int(known.sum()),masked_by_cutoff=int((known&~seen).sum())))
            if lag==0 and '2024-10-01'<e['issuance']<'2025-04-01' and len(examples)<3:
                nc=list(e['locations']).index('NC')
                if known[nc] and not seen[nc]:
                    examples.append(dict(issuance=e['issuance'],reference_week=e['context_dates'][i],location='NC',covariate=name,final_value=float(truth['covariates'][i,c,0,nc]),model_available=False,model_value=float(e['covariates'][i,c,0,nc])))
a=pd.DataFrame(rows).groupby(['kind','name','lag_weeks'],as_index=False)[['finite_final','masked_by_cutoff']].sum()
a['percent_masked']=100*a.masked_by_cutoff/a.finite_final
out=Path('docs/results/forecast-geography-v2');a.to_csv(out/'availability_audit.csv',index=False)
(out/'availability_examples.json').write_text(json.dumps(examples,indent=2)+'\n')
print(a.to_string(index=False));print(json.dumps(examples,indent=2))
