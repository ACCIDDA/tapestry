"""Summarize completed manager outputs, including all-cell and missing-cell errors."""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tapestry.experiment.finalization import metrics
root=Path(sys.argv[1]);out=Path(__file__).parent
records=[]
for p in root.glob('*/s*/attempt-*/eval_*/finalizations.csv.gz'):
    f=pd.read_csv(p)
    if not (p.parent/'manifest.json').exists(): continue
    m=pd.concat([metrics(f),metrics(f.assign(kind='all'))],ignore_index=True)
    m['cv']='rolling' if 'rolling' in p.parent.name else 'season'
    m['fold']=p.parent.name;records.append(m)
d=pd.concat(records,ignore_index=True)
d.to_csv(out/'current-all-scores.csv',index=False)
a=d.groupby(['cv','age','kind','signal','method']).normalized_mae.mean()
a=a.groupby(['cv','age','kind','method']).mean()
s=a.groupby(['cv','kind','method']).mean().unstack();s['reduction_pct']=100*(1-s.prediction/s.persistence)
s.to_csv(out/'current-error-reduction.csv');print(s.to_string())
fig,axes=plt.subplots(1,2,figsize=(13,5))
keys=['reported','missing_recent_history','missing_no_recent_history','all']
labels=['Reported','Missing,\nprior history','Missing,\nno prior history','All cells']
for ax,cv in zip(axes,['rolling','season']):
    values=s.loc[cv].reindex(keys).reduction_pct
    ax.bar(labels,values,color=['#568bad' if v>=0 else '#d77863' for v in values])
    ax.axhline(0,color='black',lw=.8);ax.set_title(cv);ax.set_ylabel('MAE reduction vs baseline (%)');ax.grid(axis='y',alpha=.2)
    for i,v in enumerate(values): ax.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')
fig.tight_layout();fig.savefig(out/'current-error-reduction.png',dpi=160)
