"""Summarize saved raw-history coverage; same geographic/target/season weights as rank."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
root=Path(__file__).resolve().parent
names={r['scenario']:r['name'] for r in json.loads((root.parent/'design.json').read_text())['rows']}
rows=[]
for p in Path('/tmp/b3-pilot-analysis/artifacts').glob('*/s*/attempt-*/totals.csv'):
    m=json.loads((p.parent/'manifest.json').read_text());d=pd.read_csv(p)
    cols=['model_covered_50','model_covered_90','ensemble_covered_50','ensemble_covered_90']
    loc=d.groupby(['season','target','geography','location'])[['n']+cols].sum()
    for c in cols:loc[c]/=loc.n
    geo=loc.groupby(['season','target','geography'])[cols].mean().reset_index()
    for (season,target),g in geo.groupby(['season','target']):
        w=np.where(g.geography.eq('US'),.2,.8)
        rows.append(dict(name=names[m['scenario']],seed=m['seed'],season=season,target=target,
                         **{c:np.average(g[c],weights=w) for c in cols}))
d=pd.DataFrame(rows);d.to_csv(root/'coverage-target-season.csv',index=False)
out=[]
for (name,seed,season),g in d.groupby(['name','seed','season']):
    w=np.where(g.target.str.contains('prop ed'),.5,1.)
    out.append(dict(name=name,seed=seed,season=season,**{c:np.average(g[c],weights=w) for c in cols}))
s=pd.DataFrame(out).groupby('name')[cols].mean();s.to_csv(root/'coverage-summary.csv')
print(s.round(3).to_string())
