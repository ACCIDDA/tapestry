"""Static audit figures; no image inspection required."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
out=Path('docs/results/b0-audit')
r=pd.read_csv(out/'configuration_ranking.csv')
s=pd.read_csv(out/'season_scores.csv');s=s[s.geography=='all']
labels={'b0-target100':'B0 target · cap 100','b0-pathogen300':'B0 pathogen · cap 300','latest-best':'Latest best covariates','latest-control':'Latest no covariates'}
order=list(labels)
fig,ax=plt.subplots(figsize=(8,3.7))
for i,name in enumerate(order):
 row=r[r.config_id==name].iloc[0]
 ax.errorbar(row.combined_mean,i,xerr=row.combined_sd,fmt='o',capsize=4,color=['#227c9d','#227c9d','#c45b34','#c45b34'][i])
 ax.text(row.combined_mean+.008,i-.18,f'{row.combined_mean:.3f}',fontsize=10)
ax.axvline(1,color='0.4',ls='--');ax.set_yticks(range(4),[labels[x] for x in order]);ax.invert_yaxis()
ax.set_xlabel('Relative WIS · mean ± seed SD (lower is better)');ax.set_title('Saved forecasts rescored on identical frozen tasks')
ax.spines[['top','right']].set_visible(False);fig.tight_layout();fig.savefig(out/'scores.png',dpi=180);plt.close(fig)
wide=s.groupby(['season','target','config_id']).wis_ratio.mean().unstack('config_id')[order]
fig,ax=plt.subplots(figsize=(9,6.2));im=ax.imshow(wide.to_numpy(),vmin=.7,vmax=1.35,cmap='RdBu_r',aspect='auto')
ax.set_xticks(range(4),['B0 target','B0 pathogen','Latest best','Latest control'])
ax.set_yticks(range(len(wide)),[f"{season}  {target.replace('wk inc ','').replace('prop ed visits','ED').replace('hosp','admissions')}" for season,target in wide.index])
for i in range(len(wide)):
 for j in range(4):ax.text(j,i,f'{wide.iloc[i,j]:.3f}',ha='center',va='center',color='white' if wide.iloc[i,j]>1.25 or wide.iloc[i,j]<.78 else 'black')
ax.set_title('No B0 leader wins every available target–season\nThree-seed mean relative WIS; 1 = ensemble')
fig.colorbar(im,ax=ax,label='Relative WIS (color capped at 1.35)',shrink=.7);fig.tight_layout();fig.savefig(out/'target-season.png',dpi=180);plt.close(fig)
old=pd.read_csv('data/audits/b0/old-ranking/season_scores.csv');old=old[old.geography=='all']
check=old.groupby(['config_id','target','season']).wis_ratio.mean().groupby('config_id').agg(worst_target_season='max',comparisons='count')
check['wins_every_comparison']=check.worst_target_season<1
check.to_csv(out/'all_b0_unicorn_check.csv')
print('Strict unicorns:',int(check.wins_every_comparison.sum()),'of',len(check))
