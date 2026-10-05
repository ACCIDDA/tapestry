"""Describe observed reporting curves and residual reference differences; no fit."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tapestry.dataset.build import load
from tapestry.dataset.cv import season
p=load('data/processed/panel.npz');dates=p['dates'].astype('datetime64[D]');issues=p['issuance_dates'].astype('datetime64[D]')
seasons=np.array([season(str(d)) for d in dates]);out=Path(__file__).parent
records=[]
for k,name in enumerate(p['target_names'].astype(str)):
    for d in range(0,27):
        want=dates+np.timedelta64(4+7*d,'D');wi=np.searchsorted(issues,want)
        valid=(wi<len(issues))&(issues[wi.clip(0,len(issues)-1)]==want)
        for s in ['2023-2024','2024-2025','2025-2026']:
            for geo,loc in [('states',p['locations']!='US'),('US',p['locations']=='US')]:
                ti=np.flatnonzero(valid&(seasons==s));a=p['asof_targets'][wi[ti],ti,:,k][:,loc];y=p['targets'][ti,:,k][:,loc]
                ok=np.isfinite(a)&np.isfinite(y);den=np.where(ok,y,0).sum();err=np.where(ok,abs(a-y),0).sum()
                records.append(dict(signal=name,season=s,geography=geo,delay=d,cells=int(ok.sum()),events=len(ti),
                    completeness=np.where(ok,a,0).sum()/den if den>0 else np.nan,wape=err/den if den>0 else np.nan))
f=pd.DataFrame(records);f.to_csv(out/'maturity-audit.csv',index=False)
print(f[f.delay.isin([0,4,12,26])&f.geography.eq('states')].to_string(index=False))
fig,axes=plt.subplots(2,3,figsize=(14,8))
for ax,name in zip(axes.flat,p['target_names'].astype(str)):
    for s,g in f[f.signal.eq(name)&f.geography.eq('states')].groupby('season'):
        ax.plot(g.delay,g.completeness,label=s)
    ax.axhline(1,color='black',lw=.7);ax.set_title(name);ax.set_xlabel('Weeks after first scheduled Wednesday');ax.set_ylabel('Reported / later reference, pooled states');ax.grid(alpha=.2)
axes[0,0].legend();fig.tight_layout();fig.savefig(out/'observed-delay-curves.png',dpi=150)
