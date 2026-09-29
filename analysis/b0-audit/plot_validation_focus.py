"""Plain-language plot of the validation calendar change that actually differs."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from tapestry.dataset.build import load
p=load('data/audits/b0/original-panel-unified.npz');u=list(p['locations']).index('US')
dates=pd.to_datetime(p['dates']);keep=(dates>='2023-09-02')&(dates<='2024-07-27');d=dates[keep]
v=pd.read_csv('docs/results/b1-to-b0-chain/validation_week_peak_positions.csv')
v=v[(v.season=='2023-2024')&(v.target=='Flu admissions')]
fig,ax=plt.subplots(figsize=(11,4.5))
for c,name,color in [(0,'Flu admissions','#c26316'),(1,'COVID admissions','#395e9c')]:
 y=p['targets'][keep,u,c];ax.plot(d,y/np.nanmax(y),label=name,color=color,lw=2)
 peak=d[np.nanargmax(y)];ax.annotate(peak.strftime('%b %d'),(peak,1),xytext=(0,10 if c==0 else 25),textcoords='offset points',ha='center',color=color,fontsize=9)
for column,y,color,label in [('current_validation',-.07,'#b63f75','Newer validation weeks'),('b0_validation',-.14,'#2d7966','B0 validation weeks')]:
 for day in pd.to_datetime(v.loc[v[column],'week']):
  ax.plot([day-pd.Timedelta(days=3),day+pd.Timedelta(days=3)],[y,y],lw=7,solid_capstyle='butt',color=color)
 ax.text(pd.Timestamp('2024-06-08'),y,label,va='center',fontsize=9,color=color)
ax.set_ylim(-.20,1.22);ax.set_yticks([0,.25,.5,.75,1],['0%','25%','50%','75%','100%'])
ax.set_ylabel('National admissions relative to each disease’s peak')
ax.set_title('The calendar change: which 2023–24 weeks choose training duration?\nThe newer winter block includes the peak; B0’s block comes later in the decline.',loc='left',fontsize=12,pad=12)
ax.spines[['top','right','bottom']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.legend(loc='upper right',frameon=False)
ax.set_xlim(pd.Timestamp('2023-08-25'),pd.Timestamp('2024-07-27'));fig.tight_layout()
out=Path('docs/results/b1-to-b0-chain');fig.savefig(out/'validation-timing.png',dpi=180);fig.savefig(out/'validation-timing.svg');plt.close(fig)
