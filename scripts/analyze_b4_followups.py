import pandas as pd,json
from pathlib import Path
for exp in ['b4-flu-refine-20261006','b4-flu-heads-20261006']:
 p=Path('docs/experiments')/exp;design=pd.DataFrame(json.loads((p/'design.json').read_text())['rows']).rename(columns={'scenario':'config_id'})
 d=pd.read_csv(p/'analysis/pilot-rankings.csv').merge(design,on='config_id',validate='many_to_one');assert (d['count']==2).all()
 t=d[d.season.isin(['equal_season_mean','available_season_mean'])].pivot(index=[*design.columns,'history'],columns='metric',values='mean').reset_index()
 t.to_csv(p/'analysis/labeled-summary.csv',index=False)
 print('\n',exp)
 if 'heads' in exp: print(t.drop(columns=['config_id']).to_string(index=False))
 else:print(t[t.history=='corrected'].drop(columns=['config_id']).sort_values('flu_native').to_string(index=False))
 s=pd.read_csv(p/'analysis/pilot-seed-scores.csv').merge(design,on='config_id',validate='many_to_one');s.to_csv(p/'analysis/labeled-seeds.csv',index=False)
 if 'heads' in exp:
  x=pd.read_csv(p/'analysis/distribution-scores.csv').merge(design,on='config_id',validate='many_to_one');print('DIAGNOSTIC ROWS',len(x))
  metrics=[c for c in x if 'wis' in c or 'coverage' in c]
  x['geography']=x.location.map(lambda v:'US' if v=='US' else 'states_dc')
  g=x.groupby(['anchor','variant','seed','history','season','target','geography'])[metrics].mean().reset_index()
  g.to_csv(p/'analysis/distribution-geography.csv',index=False)
  metrics=[c for c in metrics if 'coverage' in c]
  g[metrics]=g[metrics].mul(g.geography.map({'US':.2,'states_dc':.8}),axis=0)
  h=g.groupby(['anchor','variant','seed','history','season','target'])[metrics].sum(min_count=1).reset_index()
  h.to_csv(p/'analysis/distribution-season-seed.csv',index=False)
  h=h.groupby(['anchor','variant','history','target'])[metrics].mean().reset_index()
  h.to_csv(p/'analysis/distribution-summary.csv',index=False)
  print(h[(h.history=='corrected')&(h.target=='flu_admissions')].to_string(index=False))
