"""Full-season target errors and state-versus-shared reporting curves."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(sys.argv[1]);out=Path(__file__).parent
paths=list(root.glob('*/s*/attempt-*/eval_2025-2026/finalizations.csv.gz'))
if len(paths)!=1:raise ValueError('Need one completed 2025–26 result')
f=pd.read_csv(paths[0]);f=f[f.signal.str.startswith(('nhsn_','nssp_'))].copy()
f['baseline_error']=(f.persistence-f.truth).abs()/f.scale
f['model_error']=(f.prediction-f.truth).abs()/f.scale
names=sorted(f.signal.unique())
fig,axes=plt.subplots(3,2,figsize=(15,10))
for ax,name in zip(axes.ravel(),names):
    # Newest reconstructed week only: no averaging with mostly-complete old weeks.
    g=f[(f.signal==name)&(f.age==0)&(f.location!='US')]
    weekly=g.groupby('issuance')[['baseline_error','model_error']].mean()
    for column,label in [('baseline_error','Unchanged report / fallback'),('model_error','Nowcaster')]:
        ax.plot(pd.to_datetime(weekly.index),weekly[column],label=label,lw=1)
    ax.set_title(name);ax.set_ylabel('Equal-state normalized MAE');ax.grid(alpha=.2)
    ax.tick_params(axis='x',rotation=20)
axes[0,0].legend(fontsize=8);fig.suptitle('Full 2025–26 season: newest target week, states/DC')
fig.tight_layout();fig.savefig(out/'current-season-weekly.png',dpi=160)
# Per-state full-season results, retained separately from the national row.
g=f.groupby(['signal','location'])[['baseline_error','model_error']].mean()
g['reduction_pct']=100*(1-g.model_error/g.baseline_error.replace(0,np.nan))
g.to_csv(out/'current-season-state-scores.csv')
t=g.reduction_pct.unstack('signal').reindex(columns=names)
fig,ax=plt.subplots(figsize=(11,15));im=ax.imshow(t,aspect='auto',cmap='RdBu',vmin=-25,vmax=25)
ax.set_yticks(range(len(t)),t.index,fontsize=7);ax.set_xticks(range(len(names)),names,rotation=35,ha='right',fontsize=8)
ax.set_title('Full-season state MAE reduction (%), all eight reconstructed ages\nColor limited to ±25%; exact values in CSV, grey = zero baseline error')
im.cmap.set_bad('#dddddd');fig.colorbar(im,ax=ax,label='Positive = improved');fig.tight_layout()
fig.savefig(out/'current-season-states.png',dpi=160)
# Compare local and common estimated remaining development, not hindsight labels.
fig,axes=plt.subplots(3,2,figsize=(15,10))
for ax,name in zip(axes.ravel(),names):
    g=f[(f.signal==name)&(f.age==0)]
    common=g.groupby('issuance').shared_development_factor.first()
    ax.plot(pd.to_datetime(common.index),100*(common-1),'k--',label='Shared state curve',lw=1.8)
    for location in ['NC','CA','TX']:
        z=g[g.location==location].sort_values('issuance')
        ax.plot(pd.to_datetime(z.issuance),100*(z.development_factor-1),label=location,lw=1)
    ax.set_title(name);ax.set_ylabel('Estimated remaining development (%)');ax.grid(alpha=.2)
    ax.tick_params(axis='x',rotation=20)
axes[0,0].legend(fontsize=8);fig.suptitle('52-week partial pooling: common curve and example states\nRaw development factors before calibration; not measured final revision percentages')
fig.tight_layout();fig.savefig(out/'current-season-shared.png',dpi=160)
