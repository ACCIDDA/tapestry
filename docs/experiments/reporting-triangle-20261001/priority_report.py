"""B2-aligned diagnostic groups, fixed by the saved B2 top-three configurations."""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tapestry.experiment.finalization import metrics
root=Path(sys.argv[1]);out=Path(__file__).parent
covariates={'inpatient_flu','inpatient_covid','nwss_flu_wval_like','nwss_covid_wval_like',
            'nwss_rsv_wval_like','kinsa_ili','ilinet_ili','clinical_lab_flu_pct_positive','flusurv_flu_rate'}
records=[]
for p in root.glob('*/s*/attempt-*/eval_*/finalizations.csv.gz'):
    if not (p.parent/'manifest.json').exists():continue
    f=pd.read_csv(p)
    for label,keep in [('Targets',f.signal.str.startswith(('nhsn_','nssp_'))),('B2 covariates',f.signal.isin(covariates))]:
        for kind,frame in [('all',f[keep].assign(kind='all')),('strata',f[keep])]:
            m=metrics(frame);m['priority']=label;m['fold']=p.parent.name
            m['cv']='rolling' if 'rolling' in p.parent.name else 'season';records.append(m)
d=pd.concat(records,ignore_index=True);d.to_csv(out/'current-priority-scores.csv',index=False)
s=d.groupby(['priority','cv','age','kind','signal','method']).normalized_mae.mean()
s=s.groupby(['priority','cv','age','kind','method']).mean()
s=s.groupby(['priority','cv','kind','method']).mean().unstack()
s['reduction_pct']=100*(1-s.prediction/s.persistence);s.to_csv(out/'current-priority-summary.csv');print(s.to_string())
fold=d[d.kind=='all'].groupby(['priority','cv','fold','age','method']).normalized_mae.mean()
fold=fold.groupby(['priority','cv','fold','method']).mean().unstack()
fold['reduction_pct']=100*(1-fold.prediction/fold.persistence)
fold.to_csv(out/'current-priority-folds.csv');print(fold.to_string())
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,group in zip(axes,['Targets','B2 covariates']):
    values=[s.loc[(group,'rolling','all'),'reduction_pct'],
            fold.loc[(group,'season','eval_2025-2026'),'reduction_pct']]
    ax.bar(['Recent rolling','2025–26'],values,color='#568bad');ax.axhline(0,color='black',lw=.8)
    for i,v in enumerate(values):ax.annotate(f'{v:.2f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')
    ax.set_title(group if group=='Targets' else 'B2 covariates: 9 rolling / 4 seasonal');ax.set_ylabel('All-cell MAE reduction (%)');ax.grid(axis='y',alpha=.2)
fig.tight_layout();fig.savefig(out/'current-priority.png',dpi=160)
