"""Compact display of the selected candidate; selection remains explicit."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('--root',type=Path,default=Path(__file__).parent);a=p.parse_args()
f=pd.read_csv(a.root/'target-scores.csv')
x=f[f.model.isin(['triangle',a.model])&f.fold.eq('2025-2026')&f.window.eq('newest')&f.kind.eq('all')&f.method.eq('prediction')]
x.to_csv(a.root/'selected-target-scores.csv',index=False)
signals=['nhsn_flu_admissions','nhsn_covid_admissions','nhsn_rsv_admissions','nssp_flu_proportion','nssp_covid_proportion','nssp_rsv_proportion']
labels=['Flu admissions','COVID admissions','RSV admissions','Flu ED','COVID ED','RSV ED']
fig,axes=plt.subplots(1,2,figsize=(12,5.6),sharey=True)
for ax,geo,title in zip(axes,['states','US'],['States pooled within each target','National targets']):
    for offset,model,label,color in [(-.18,'triangle','Previous model','#a5aab2'),(.18,a.model,'Selected model','#168c89')]:
        g=x[x.geography.eq(geo)&x.model.eq(model)].set_index('signal').reindex(signals)
        vals=100*g.wape.to_numpy();ax.barh(np.arange(6)+offset,vals,height=.34,label=label,color=color)
        for j,v in enumerate(vals):ax.text(v+.12,j+offset,f'{v:.1f}%',va='center',fontsize=8)
    ax.axvline(5,color='#c86037',ls='--',lw=1,label='5% target');ax.set_title(title);ax.set_xlabel('Newest-week absolute error / observed total (%)');ax.set_yticks(np.arange(6),labels);ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True);ax.margins(x=.15)
axes[0].invert_yaxis()
axes[0].legend(loc='lower right',fontsize=8)
fig.suptitle('2025–26 full-season nowcasting, including reporting outages',fontsize=14)
fig.tight_layout();fig.savefig(a.root/'selected-performance.png',dpi=180);plt.close(fig)
w=pd.read_csv(a.root/'weekly-scores.csv.gz');w=w[w.model.isin(['triangle',a.model])&w.fold.eq('2025-2026')&w.method.eq('prediction')]
fig,axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
for ax,geo in zip(axes,['states','US']):
    for model,label,color in [('triangle','Previous model','#9197a0'),(a.model,'Selected model','#168c89')]:
        g=w[w.model.eq(model)&w.geography.eq(geo)].groupby('issuance').wape.mean()
        ax.plot(pd.to_datetime(g.index),g*100,label=label,color=color,lw=1.8)
    ax.axhline(5,color='#c86037',ls='--',lw=.8);ax.axvspan(pd.Timestamp('2025-10-01'),pd.Timestamp('2025-11-12'),color='#d9b067',alpha=.16)
    ax.set_ylabel(('States' if geo=='states' else 'US')+' mean WAPE (%)');ax.grid(alpha=.18)
axes[0].legend();axes[0].set_title('Newest-week errors through the season · shaded interval: target reporting outage')
fig.tight_layout();fig.savefig(a.root/'selected-season.png',dpi=180);plt.close(fig)
print(x.groupby(['model','geography'])[['wape','location_wape','normalized_mae','within5']].mean().to_string())
c=pd.read_csv(a.root/'learned-curves.csv.gz')
c=c[c.model.eq(a.model)&c.fold.eq('2025-2026')&c.geography.eq('states')]
fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True)
for ax,signal,label in zip(axes.flat,signals,labels):
    for date in ['2025-11-26','2026-01-07','2026-02-25']:
        g=c[c.signal.eq(signal)&c.issuance.eq(date)].sort_values('age')
        ax.plot(g.age,g.factor,marker='o',markersize=3,label=date)
    ax.axhline(1,color='black',lw=.7);ax.set_title(label);ax.set_ylabel('Median state development factor');ax.set_xlabel('Weeks after initial scheduled report');ax.grid(alpha=.18)
axes[0,0].legend(fontsize=8);fig.suptitle('Learned reporting corrections update through the season',fontsize=14)
fig.tight_layout();fig.savefig(a.root/'selected-delay-curves.png',dpi=180);plt.close(fig)
