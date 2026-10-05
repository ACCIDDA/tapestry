"""Summarize manager-produced scores; no fitting and no alternative scoring protocol."""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('experiment', type=Path)
args=parser.parse_args()
output=Path(__file__).parent
new=pd.read_csv(args.experiment/'finalization-ranking/signal-scores.csv')
rows=[]
for label,frame in [('Nowcaster',new)]:
    by_signal=frame.groupby(['cv','age','kind','signal','method']).normalized_mae.mean()
    means=by_signal.groupby(['cv','age','kind','method']).mean().unstack('method')
    means['ratio']=means.prediction/means.persistence.replace(0,np.nan)
    means['model']=label;rows.append(means.reset_index())
summary=pd.concat(rows,ignore_index=True)
summary.to_csv(output/f'current-summary.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
for ax,cv in zip(axes,['rolling','season']):
    for label,g in summary[(summary.cv==cv)&(summary.kind=='reported')].groupby('model'):
        ax.plot(g.age,g.ratio,'o-',label=label)
    ax.axhline(1,color='black',ls='--',label='Unchanged reports');ax.set_title(cv)
    ax.set_xlabel('Weeks behind source T-X');ax.set_xticks(range(8));ax.grid(alpha=.2)
axes[0].set_ylabel('Normalized MAE / unchanged reports');axes[-1].legend()
fig.tight_layout();fig.savefig(output/f'current-reported.png',dpi=160)
print(summary[['model','cv','age','kind','ratio']].to_string(index=False))
by=new.groupby(['cv','signal','age','kind','method']).normalized_mae.mean().unstack('method')
by['ratio']=by.prediction/by.persistence.replace(0,np.nan)
by.to_csv(output/f'current-signals.csv')
geo=pd.read_csv(args.experiment/'finalization-ranking/geography-season-scores.csv')
# The geography file does not carry CV explicitly; derive from saved scenario.
geo['cv']=np.where(geo.scenario.str.contains('finalization_cv=season'),'season','rolling')
g=geo.groupby(['cv','geography','age','kind','signal','method']).normalized_mae.mean()
g=g.groupby(['cv','geography','age','kind','method']).mean().unstack('method')
g['ratio']=g.prediction/g.persistence.replace(0,np.nan)
g.to_csv(output/f'current-geography.csv')
# Distinguish correction benefit from damage to reports that were already correct.
retention=[]
for p in args.experiment.glob('*/s*/attempt-*/eval_*/finalizations.csv.gz'):
    if not (p.parent/'manifest.json').exists():
        continue
    frame=pd.read_csv(p)
    frame=frame[frame.kind=='reported'].copy()
    frame['already_correct']=frame.report==frame.truth
    frame['changed']=(frame.prediction-frame.report).abs()>1e-8*frame.scale
    frame['normalized_ae']=(frame.prediction-frame.truth).abs()/frame.scale
    frame['baseline_ae']=(frame.report-frame.truth).abs()/frame.scale
    frame['cv']='rolling' if 'rolling' in p.parent.name else 'season'
    g=frame.groupby(['cv','signal','age','already_correct','location']).agg(
        model_error=('normalized_ae','mean'),baseline_error=('baseline_ae','mean'),
        changed_fraction=('changed','mean'),cells=('changed','size')).reset_index()
    g['fold']=p.parent.name
    retention.append(g)
retention=pd.concat(retention,ignore_index=True)
retention.to_csv(output/f'current-correct-report-audit.csv',index=False)
