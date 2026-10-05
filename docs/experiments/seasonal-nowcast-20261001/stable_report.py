"""Primary comparison on complete history and uninterrupted reporting; no refitting."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tapestry.dataset.build import load
from tapestry.experiment.finalization import reporting_support,target_percent_metrics
p=argparse.ArgumentParser();p.add_argument('roots',nargs='+',type=Path);a=p.parse_args()
out=Path(__file__).parent;panel=load('data/processed/panel.npz')
records=[];seen=set();coverage=[];calendars={}
selected='adaptive_chain,h8,p4,median,grid'
for root in a.roots:
    runs=pd.read_csv(root/'runs.csv')
    for run in runs[runs.status.eq('complete')].itertuples():
        config=dict(t.split('=',1) for t in run.scenario.split(','))
        model=config.get('finalization_model','triangle')
        if model!='triangle':
            model+=f",h{float(config.get('finalization_halflife',8)):g},p{float(config.get('finalization_pool',4)):g},{config.get('finalization_statistic','mean')}"
            if config.get('finalization_quantize')=='1':model+=',grid'
        for path in (root/run.attempt).glob('eval_*/finalizations.csv.gz'):
            fold=path.parent.name.removeprefix('eval_');key=(model,fold)
            if key in seen:continue
            seen.add(key)
            f=pd.read_csv(path);f=f[f.age.eq(0)]
            calendars.setdefault(fold,sorted(f.issuance.unique()))
            f=reporting_support(panel,f)
            scored=target_percent_metrics(f);scored['model'],scored['fold']=model,fold;records.append(scored)
            if model in ['triangle',selected]:
                for stratum,keep in [('reported',f.kind.eq('reported')),('complete_history_12',f.complete_history_12),('uninterrupted_8',f.uninterrupted_8),('complete_and_uninterrupted',f.complete_history_12&f.uninterrupted_8)]:
                    for (signal,geo),g in f[keep].assign(geography=np.where(f[keep].location.eq('US'),'US','states')).groupby(['signal','geography']):
                        for method in ('prediction','persistence'):
                            t=g.assign(ae=abs(g[method]-g.truth)).groupby('issuance',as_index=False).agg(ae=('ae','sum'),truth_sum=('truth','sum'))
                            t['model'],t['fold'],t['stratum'],t['signal'],t['geography'],t['method']=model,fold,stratum,signal,geo,method
                            coverage.extend(t.to_dict('records'))
d=pd.concat(records,ignore_index=True);d.to_csv(out/'stable-target-scores.csv',index=False)
s=d.groupby(['model','fold','stratum','geography','method'],as_index=False).agg(wape=('wape','mean'),location_wape=('location_wape','mean'),normalized_mae=('normalized_mae','mean'),within5=('within5','mean'),cells=('cells','sum'),signals=('signal','nunique'))
s.to_csv(out/'stable-summary.csv',index=False)
w=pd.DataFrame(coverage);w.to_csv(out/'stable-weekly.csv.gz',index=False)
print(s[s.fold.eq('2025-2026')&s.stratum.isin(['complete_history_12','complete_and_uninterrupted'])&s.method.eq('prediction')].to_string(index=False))
fig,axes=plt.subplots(1,2,figsize=(12,5.6),sharey=True)
signals=['nhsn_flu_admissions','nhsn_covid_admissions','nhsn_rsv_admissions','nssp_flu_proportion','nssp_covid_proportion','nssp_rsv_proportion']
labels=['Flu admissions','COVID admissions','RSV admissions','Flu ED','COVID ED','RSV ED']
for ax,geo,title in zip(axes,['states','US'],['States pooled within each target','National targets']):
    for offset,model,label,color in [(-.18,'triangle','Previous model','#a5aab2'),(.18,selected,'Selected model','#168c89')]:
        g=d[d.fold.eq('2025-2026')&d.stratum.eq('complete_history_12')&d.geography.eq(geo)&d.method.eq('prediction')&d.model.eq(model)].set_index('signal').reindex(signals)
        vals=100*g.wape.to_numpy();ax.barh(np.arange(6)+offset,vals,height=.34,label=label,color=color)
        for j,v in enumerate(vals):ax.text(v+.1,j+offset,f'{v:.1f}%',va='center',fontsize=8)
    ax.axvline(5,color='#c86037',ls='--',lw=1,label='5% target');ax.set_title(title);ax.set_xlabel('Absolute error / observed total (%)');ax.set_yticks(np.arange(6),labels);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True);ax.margins(x=.18)
axes[0].invert_yaxis();axes[0].legend(loc='lower right',fontsize=8)
fig.suptitle('2025–26 newest-week nowcasts with complete visible 12-week histories',fontsize=13)
fig.tight_layout();fig.savefig(out/'stable-performance.png',dpi=180);plt.close(fig)

# Paired four-week blocks over the full seasonal calendar, preserving empty weeks.
rng=np.random.default_rng(42);intervals=[]
for stratum in ('complete_history_12','complete_and_uninterrupted'):
    for geo in ('states','US'):
        z=w[w.fold.eq('2025-2026')&w.stratum.eq(stratum)&w.geography.eq(geo)&w.method.eq('prediction')]
        idx=calendars['2025-2026'];n=len(idx)
        def matrix(model,column):
            return z[z.model.eq(model)].pivot(index='issuance',columns='signal',values=column).reindex(index=idx,columns=signals).fillna(0).to_numpy()
        truth=matrix('triangle','truth_sum');before=matrix('triangle','ae');after=matrix(selected,'ae')
        draws=((rng.integers(0,n,(2000,int(np.ceil(n/4))))[:,:,None]+np.arange(4))%n).reshape(2000,-1)[:,:n]
        den=truth[draws].sum(1);valid=(den>0).all(1)
        old=np.mean(before[draws][valid].sum(1)/den[valid],axis=1)
        new=np.mean(after[draws][valid].sum(1)/den[valid],axis=1)
        gain=1-new/old
        intervals.append(dict(stratum=stratum,geography=geo,reduction=1-np.mean(after.sum(0)/truth.sum(0))/np.mean(before.sum(0)/truth.sum(0)),low=np.quantile(gain,.025),high=np.quantile(gain,.975),draws=int(valid.sum()),block_weeks=4))
pd.DataFrame(intervals).to_csv(out/'stable-bootstrap.csv',index=False)
