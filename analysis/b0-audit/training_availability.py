"""Actual training context availability under the two controlled input policies."""
from pathlib import Path
import numpy as np
import pandas as pd
from tapestry.dataset.build import load
from tapestry.dataset.cv import fold,season
from tapestry.model.scenario import Scenario
panel=load('data/audits/b0/original-panel-unified.npz')
rows=[]
for held in ('2023-2024','2024-2025','2025-2026'):
    full=fold(panel,Scenario(ed_transform='logit',patience=30,fit_partition='pathogen'),held).train
    wed=fold(panel,Scenario(ed_transform='logit',patience=30,fit_partition='pathogen',input_mode='finalized_available'),held).train
    lookup={e['context_dates'][-1]:e for e in wed}
    for label in ('2023-2024','2024-2025','2025-2026'):
        episodes=[e for e in full if season(e['context_dates'][-1])==label]
        if not episodes:continue
        a=np.stack([e['available'] for e in episodes]);b=np.stack([lookup[e['context_dates'][-1]]['available'] if e['context_dates'][-1] in lookup else np.zeros_like(e['available']) for e in episodes])
        for c,target in enumerate(panel['target_names']):
            for which,aa,bb in [('all_12_weeks',a[:,:,c],b[:,:,c]),('latest_week',a[:,-1:,c],b[:,-1:,c])]:
                eligible=int(aa.sum());present=int((aa&bb).sum())
                rows.append(dict(held_out=held,training_season=label,target=target,context=which,episodes=len(episodes),full_cells=eligible,wednesday_cells=present,retained_fraction=present/eligible if eligible else np.nan))
out=Path('docs/results/b0-reproduction');out.mkdir(parents=True,exist_ok=True)
table=pd.DataFrame(rows);table.to_csv(out/'training_availability.csv',index=False)
print(table.groupby(['training_season','target','context']).retained_fraction.mean().unstack('context').round(3).to_string())
