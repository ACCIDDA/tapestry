"""Paired effects by season and outcome for interpretation of the final chain."""
from pathlib import Path
import pandas as pd
out=Path('docs/results/b1-to-b0-chain')
order=['00','01','02','03','03b','04','05','06','07','09']
f=pd.read_csv(out/'season_scores.csv');f=f[f.geography=='all']
rows=[]
for before,after in zip(order,order[1:]):
 a=f[f.config_id==f'stage{before}'].set_index(['seed','season','target'])
 b=f[f.config_id==f'stage{after}'].set_index(['seed','season','target'])
 d=(b.wis_ratio-a.wis_ratio).rename('delta').reset_index()
 for (season,target),part in d.groupby(['season','target']):
  rows.append(dict(stage=after,season=season,target=target,delta=part.delta.mean(),improved_seeds=int((part.delta<0).sum()),delta_min=part.delta.min(),delta_max=part.delta.max()))
pd.DataFrame(rows).to_csv(out/'target_effects.csv',index=False)
f=pd.read_csv(out/'season_composite_scores.csv');f=f[f.geography=='all'];rows=[]
for before,after in zip(order,order[1:]):
 a=f[f.config_id==f'stage{before}'].set_index(['seed','season']).combined
 b=f[f.config_id==f'stage{after}'].set_index(['seed','season']).combined
 for season,part in (b-a).rename('delta').reset_index().groupby('season'):
  rows.append(dict(stage=after,season=season,delta=part.delta.mean(),improved_seeds=int((part.delta<0).sum())))
pd.DataFrame(rows).to_csv(out/'season_effects.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
