"""Report raw versus calibrated estimates already emitted by the manager run."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tapestry.experiment.finalization import metrics
root=Path(sys.argv[1]); output=Path(__file__).parent
records=[]
for p in root.glob('*/s*/attempt-*/eval_*/finalizations.csv.gz'):
    f=pd.read_csv(p)
    cv='rolling' if 'rolling' in p.parent.name else 'season'
    for label,column in [('Nowcaster','prediction'),('Raw triangle','triangle_raw')]:
        m=metrics(f.assign(prediction=f[column]));m=m[m.method.isin(['prediction','persistence'])].copy()
        m['model']=label;m['cv']=cv;m['fold']=p.parent.name;records.append(m)
d=pd.concat(records,ignore_index=True)
d.to_csv(output/f'current-raw-and-calibrated.csv',index=False)
g=d.groupby(['cv','model','age','kind','signal','method']).normalized_mae.mean()
s=g.groupby(['cv','model','age','kind','method']).mean().unstack('method');s['ratio']=s.prediction/s.persistence
fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
for ax,cv in zip(axes,['rolling','season']):
    for model in ['Raw triangle','Nowcaster']:
        t=s.loc[(cv,model,slice(None),'reported'),:].reset_index()
        ax.plot(t.age,t.ratio,'o-',label=model)
    ax.axhline(1,color='black',ls='--');ax.set_title(cv);ax.set_xlabel('Weeks behind source T-X');ax.grid(alpha=.2)
axes[0].set_ylabel('Normalized MAE / unchanged reports');axes[1].legend();fig.tight_layout()
fig.savefig(output/f'current-calibration.png',dpi=160)
print(s.to_string())
reported=d[d.kind=='reported'].groupby(['cv','model','signal','method']).normalized_mae.mean().unstack('method')
reported['ratio']=reported.prediction/reported.persistence
reported.to_csv(output/f'current-source-ratios.csv')
print(reported.to_string())
# Native-unit traces for the latest completed forward-season evaluation.
paths=list(root.glob('*/s*/attempt-*/eval_2025-2026/finalizations.csv.gz'))
if paths:
    f=pd.read_csv(paths[0]);f=f[(f.age==0)&(f.kind=='reported')]
    fig,axes=plt.subplots(2,2,figsize=(14,7))
    for row,signal in enumerate(['nhsn_flu_admissions','nssp_flu_proportion']):
        for col,location in enumerate(['US','NC']):
            ax=axes[row,col];t=f[(f.signal==signal)&(f.location==location)].sort_values('boundary')
            for column,label,style in [('truth','Reference','k--'),('report','Wednesday report','C1-'),('prediction','Triangle nowcast','C0-')]:
                ax.plot(pd.to_datetime(t.boundary),t[column],style,label=label,lw=1.3)
            ax.set_title(f'{location}: {signal}');ax.grid(alpha=.2);ax.tick_params(axis='x',rotation=25)
    axes[0,0].legend();fig.suptitle('Newest reconstructed week: forward 2025–26 evaluation')
    fig.tight_layout();fig.savefig(output/f'current-traces.png',dpi=160)
