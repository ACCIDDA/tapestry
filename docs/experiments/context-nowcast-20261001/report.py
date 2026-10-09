"""Compact trajectory comparison from the shared manager's scored artifacts."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from chromantis.model.scenario import Scenario

p=argparse.ArgumentParser();p.add_argument('experiment',nargs='?',default='context-nowcast-v1-20261001')
p.add_argument('--output',type=Path,default=Path(__file__).parent);a=p.parse_args()
root=Path('data/experiments')/a.experiment
out=a.output;out.mkdir(parents=True,exist_ok=True)
rank=root/'finalization-ranking'
t=pd.read_csv(rank/'trajectory-ranking.csv')
def label(text):
    s=Scenario.from_string(text)
    return ('Seasonal control' if s.finalization_model=='adaptive_chain' else
        f'Residual · p{s.finalization_penalty:g}, g{s.finalization_growth_weight:g}, h{s.finalization_residual_halflife:g}'+
        (f', strength {s.finalization_strength:g}' if s.finalization_strength!=1 else '')+
        (f', {s.finalization_features}, {s.finalization_gate}' if s.finalization_features!='basic' or s.finalization_gate!='none' else '') if s.finalization_model=='context_residual' else
        f'{s.finalization_model} · growth {s.finalization_growth:g}')
t['model']=t.scenario.map(label)
t.to_csv(out/'trajectory-summary.csv',index=False)
c=t[t.method.eq('prediction')&t.fold.eq('2025-2026')&t.stratum.eq('complete_history_12')]
w=c.pivot(index=['model','scenario'],columns='geography',values='trajectory')
w['selection_score']=.8*w.states+.2*w.US
w=w.sort_values('selection_score').reset_index()
w.to_csv(out/'selection.csv',index=False)
choice=w.iloc[0];s=Scenario.from_string(choice.scenario)
(out/'selection.json').write_text(json.dumps(dict(experiment=a.experiment,scenario=choice.scenario,
    run_id=s.run_id,features=s.finalization_features,gate=s.finalization_gate,strength=s.finalization_strength,model=s.finalization_model,growth=s.finalization_growth,penalty=s.finalization_penalty,growth_weight=s.finalization_growth_weight,residual_halflife=s.finalization_residual_halflife,criterion='2025-2026 complete history: 0.8 states + 0.2 US trajectory',
    score=choice.selection_score),indent=2)+'\n')
print(w[['model','states','US','selection_score']].to_string(index=False))
fig,axes=plt.subplots(2,3,figsize=(14,8),sharex=True)
models=list(w.model);colors=['#237e83' if x!='Seasonal control' else '#8e969f' for x in models]
for i,geo in enumerate(['states','US']):
    g=c[c.geography.eq(geo)].set_index('model').reindex(models)
    for ax,metric,title in zip(axes[i],['point','level','growth'],['Newest week','Four-week level','Two-week change']):
        control=float(g.loc['Seasonal control',metric]);values=g[metric]/control
        ax.barh(np.arange(len(models)),values,color=colors)
        ax.axvline(1,color='#444',ls='--',lw=1)
        ax.set_yticks(np.arange(len(models)),models if ax is axes[i,0] else [])
        ax.invert_yaxis();ax.set_title(f'{geo}: {title}');ax.set_xlabel('Error / seasonal control')
        for j,v in enumerate(values):ax.text(v+.004,j,f'{v:.3f}',va='center',fontsize=8)
        ax.set_xlim(0,max(1.1,values.max()*1.13));ax.grid(axis='x',alpha=.15)
fig.suptitle('Recent trajectory reconstruction · 2025–26 complete target histories\nEqual-location, training-scale errors; lower is better')
fig.tight_layout();fig.savefig(out/'trajectory-comparison.png',dpi=170);plt.close(fig)
point=pd.read_csv(rank/'target-ranking.csv');point['model']=point.scenario.map(label)
point.to_csv(out/'point-summary.csv',index=False)
print(point[point.fold.eq('2025-2026')&point.method.eq('prediction')&point.stratum.eq('complete_history_12')][['model','geography','wape','location_wape','within5']].to_string(index=False))
