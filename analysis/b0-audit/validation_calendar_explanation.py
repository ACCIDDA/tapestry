"""Describe actual hidden validation dates relative to observed national peaks."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tapestry.dataset.build import load
from tapestry.dataset.cv import season,season_start
from datetime import date

out=Path('docs/results/b1-to-b0-chain');out.mkdir(parents=True,exist_ok=True)
panel=load('data/audits/b0/original-panel-unified.npz');dates=panel['dates'].astype(str);labels=np.array([season(d) for d in dates]);us=list(panel['locations']).index('US')
names=['Flu admissions','COVID admissions','RSV admissions','Flu ED','COVID ED','RSV ED'];seasons=['2023-2024','2024-2025','2025-2026'];rows=[];peaks=[]
fig,axes=plt.subplots(3,2,figsize=(13,10),sharex='row')
colors=['#cc6b24','#4169a8','#4b9067']
for r,s in enumerate(seasons):
 ids=np.flatnonzero(labels==s);d=pd.to_datetime(dates[ids]);b0=(np.arange(len(ids))%16>=4)&(np.arange(len(ids))%16<7)
 pos=np.array([(date.fromisoformat(day)-season_start(int(s[:4]))).days//7 for day in dates[ids]])%16
 current=(pos>=4)&(pos<7)
 for c,name in enumerate(names):
  y=panel['targets'][ids,us,c];valid=np.isfinite(y)
  if not valid.any():continue
  peak=float(np.nanmax(y));peak_days=dates[ids][valid&(y==peak)]
  pd0=pd.Timestamp(str(peak_days[0]));peaks.append(dict(season=s,target=name,peak_week=';'.join(peak_days),peak_value=peak))
  for i,day in enumerate(dates[ids]):
   if b0[i] or current[i]:
    delta=(pd.Timestamp(str(day))-pd0).days//7
    rows.append(dict(season=s,target=name,week=day,b0_validation=bool(b0[i]),current_validation=bool(current[i]),national_value=None if not np.isfinite(y[i]) else float(y[i]),fraction_of_peak=None if not np.isfinite(y[i]) or peak==0 else float(y[i]/peak),weeks_from_peak=delta,phase='within 2 weeks of peak' if abs(delta)<=2 else ('before peak' if delta<0 else 'after peak')))
  axes[r,c//3].plot(d,y/peak,label=name.split()[0],color=colors[c%3],lw=1.4)
 for j in range(2):
  ax=axes[r,j]
  for mask,color,low,high in [(current,'#d484ad',0,.04),(b0,'#555555',.05,.09)]:
   for day in d[mask]:ax.axvspan(day-pd.Timedelta(days=3),day+pd.Timedelta(days=3),ymin=low,ymax=high,color=color,alpha=.8)
  ax.set_title(f'{s}: '+('admissions' if j==0 else 'ED'));ax.set_ylabel('Fraction of national season peak');ax.set_ylim(-.13,1.08);ax.tick_params(axis='x',rotation=25);ax.grid(alpha=.15)
axes[0,0].legend(ncol=3,loc='upper right');fig.suptitle('Validation weeks and national epidemic timing\nBottom strips: pink = current weeks; gray = B0 weeks. Curves scaled separately to each season peak.',fontsize=12)
fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(out/'validation-weeks.png',dpi=170);plt.close(fig)
pd.DataFrame(rows).to_csv(out/'validation_week_peak_positions.csv',index=False);pd.DataFrame(peaks).to_csv(out/'national_peak_weeks.csv',index=False)
print(pd.DataFrame(peaks).to_string(index=False))
print(pd.DataFrame(rows).loc[lambda x:(x.season=='2023-2024')&(x.target=='Flu admissions'),['week','b0_validation','current_validation','weeks_from_peak','phase']].to_string(index=False))
