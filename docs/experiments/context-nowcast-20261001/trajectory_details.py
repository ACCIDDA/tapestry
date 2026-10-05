"""Paired trajectory uncertainty and native-unit plots; no model fitting."""
import argparse,json
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tapestry.dataset.build import load
from tapestry.model.scenario import Scenario
from tapestry.experiment.finalization import reporting_support
from tapestry.evaluation.trajectory import trajectory_cells

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path(__file__).parent);a=p.parse_args()
out=a.output
selection=json.loads((out/'selection.json').read_text())
root=Path('data/experiments')/selection['experiment'];runs=pd.read_csv(root/'runs.csv')
chosen=selection['scenario'];control=next(s for s in runs.scenario if Scenario.from_string(s).finalization_model=='adaptive_chain')
panel=load('data/processed/panel.npz');frames={};cells={};season='2025-2026'
for label,scenario in [('Seasonal control',control),('Context residual',chosen)]:
    attempt=root/runs.loc[runs.scenario.eq(scenario),'attempt'].iloc[0]
    f=pd.read_csv(attempt/f'eval_{season}'/'finalizations.csv.gz')
    f=reporting_support(panel,f);frames[label]=f
    c=trajectory_cells(f);c=c[c.method.eq('prediction')].drop(columns='method')
    cells[label]=c.sort_values(['signal','issuance','location']).reset_index(drop=True)
keys=['signal','issuance','location'];before=cells['Seasonal control'];after=cells['Context residual']
if not before[keys].equals(after[keys]):raise ValueError('Trajectory support changed')
metrics=['point','level','growth','log_growth','trajectory']
records=[];rng=np.random.default_rng(42)
calendar=sorted(frames['Seasonal control'].issuance.unique())
for stratum in ['complete_history_12','complete_and_uninterrupted']:
    keep=before.complete_history_12
    if stratum.endswith('uninterrupted'):keep=keep&before.uninterrupted_8
    for geo in ['states','US']:
        use=keep&before.geography.eq(geo)
        b=before[use];a=after[use]
        columns=pd.MultiIndex.from_frame(b[['signal','location']].drop_duplicates())
        signals=columns.get_level_values('signal').unique()
        def array(f,metric):
            return f.pivot(index='issuance',columns=['signal','location'],values=metric).reindex(index=calendar,columns=columns).to_numpy()
        n=len(calendar);draws=((rng.integers(0,n,(1000,int(np.ceil(n/4))))[:,:,None]+np.arange(4))%n).reshape(1000,-1)[:,:n]
        for metric in metrics:
            x=array(b,metric);y=array(a,metric)
            def macro(z):
                return np.mean([np.nanmean(z[...,columns.get_level_values('signal')==s],axis=-1) for s in signals],axis=0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                old=macro(np.nanmean(x[draws],axis=1));new=macro(np.nanmean(y[draws],axis=1))
            change=100*(new/old-1);valid=np.isfinite(change)
            records.append(dict(stratum=stratum,geography=geo,metric=metric,
                change_pct=100*(macro(np.nanmean(y,axis=0))/macro(np.nanmean(x,axis=0))-1),
                low=np.quantile(change[valid],.025),high=np.quantile(change[valid],.975),draws=int(valid.sum()),
                trajectories=len(b),signals=len(signals),locations=b.location.nunique()))
pd.DataFrame(records).to_csv(out/'trajectory-bootstrap.csv',index=False)
print(pd.DataFrame(records).to_string(index=False))
# Latest-week curves keep native units and date gaps visible.
fig,axes=plt.subplots(6,2,figsize=(15,17))
for i,signal in enumerate(panel['target_names'].astype(str)):
    for j,loc in enumerate(['US','NC']):
        ax=axes[i,j]
        b=frames['Seasonal control'];b=b[b.signal.eq(signal)&b.location.eq(loc)&b.age.eq(0)].sort_values('boundary')
        a=frames['Context residual'];a=a[a.signal.eq(signal)&a.location.eq(loc)&a.age.eq(0)].sort_values('boundary')
        dates=pd.to_datetime(b.boundary);factor=100 if signal.startswith('nssp_') else 1
        ax.plot(dates,b.truth*factor,color='#222',lw=1.6,label='Later reference')
        ax.plot(dates,b.report*factor,color='#b8bdc2',lw=1,label='Reported')
        ax.plot(dates,b.prediction*factor,color='#5b8fb0',lw=1,label='Seasonal')
        ax.plot(pd.to_datetime(a.boundary),a.prediction*factor,color='#008577',lw=1.3,label='Residual')
        ax.set_title(f'{loc} · {signal.replace("nhsn_", "").replace("nssp_", "").replace("_", " ")}',fontsize=10)
        ax.set_ylabel('ED visits (%)' if factor==100 else 'Admissions');ax.grid(alpha=.15)
        ax.tick_params(axis='x',rotation=25)
axes[0,1].legend(fontsize=8);fig.suptitle('Newest-week nowcasts · 2025–26\nFull seasonal curves include gaps; primary scores use complete-history cohorts')
fig.tight_layout();fig.savefig(out/'native-nowcasts.png',dpi=155);plt.close(fig)

# Show the actual trajectory quantities, not just newest-week values.
for loc in ['US','NC']:
    fig,axes=plt.subplots(6,2,figsize=(14,17))
    for i,signal in enumerate(panel['target_names'].astype(str)):
        factor=100 if signal.startswith('nssp_') else 1
        for label,color in [('Seasonal control','#5b8fb0'),('Context residual','#008577')]:
            f=frames[label];f=f[f.signal.eq(signal)&f.location.eq(loc)&f.age.lt(4)]
            table=f.pivot(index='issuance',columns='age',values=['truth','prediction']).dropna()
            table=table.reindex(calendar)
            x=table.prediction.to_numpy();y=table.truth.to_numpy()
            for j,func in enumerate([lambda a:a.mean(1),lambda a:a[:,:2].mean(1)-a[:,2:].mean(1)]):
                ax=axes[i,j]
                if label=='Seasonal control':ax.plot(pd.to_datetime(table.index),func(y)*factor,color='#222',lw=1.5,label='Later reference')
                ax.plot(pd.to_datetime(table.index),func(x)*factor,color=color,lw=1.2,label=label)
        for j,title in enumerate(['Four-week level','Two-week change']):
            ax=axes[i,j];ax.set_title(f'{signal.replace("nhsn_", "").replace("nssp_", "").replace("_", " ")} · {title}',fontsize=10)
            ax.set_ylabel(('Percent of ED visits' if j==0 else 'Percentage points') if factor==100 else 'Admissions')
            ax.grid(alpha=.15);ax.tick_params(axis='x',rotation=25)
            if j:ax.axhline(0,color='#777',lw=.5)
    axes[0,1].legend(fontsize=8);fig.suptitle(f'{loc} · Reconstructed recent level and growth · 2025–26\nEvery point is a four-week history available at its issuance')
    fig.tight_layout();fig.savefig(out/f'level-growth-{loc}.png',dpi=150);plt.close(fig)

r=pd.DataFrame(records);fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharey=True)
for ax,stratum,title in zip(axes,['complete_history_12','complete_and_uninterrupted'],['Complete histories','Also eight uninterrupted reports']):
    for geo,offset,color in [('states',-.09,'#16877f'),('US',.09,'#466183')]:
        q=r[r.stratum.eq(stratum)&r.geography.eq(geo)].set_index('metric').loc[['point','level','growth']]
        values=-q.change_pct.to_numpy();lo=-q.high.to_numpy();hi=-q.low.to_numpy()
        ax.errorbar(values,np.arange(3)+offset,xerr=[values-lo,hi-values],fmt='o',capsize=3,color=color,label=geo)
    ax.axvline(0,color='#444',lw=1);ax.set_yticks(range(3),['Newest week','Four-week level','Two-week change']);ax.invert_yaxis()
    ax.set_title(title);ax.set_xlabel('Error reduction versus seasonal control (%)');ax.grid(axis='x',alpha=.2)
axes[1].legend();fig.suptitle('Selected age-dependent admissions correction · 2025–26\nPaired four-week block bootstrap intervals; reused development season')
fig.tight_layout();fig.savefig(out/'selected-performance.png',dpi=180);plt.close(fig)
