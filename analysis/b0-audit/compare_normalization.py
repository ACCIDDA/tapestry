"""Paired comparison of the completed normalization intervention and original runs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('data/experiments/b0-normalization-audit')
out=Path('docs/results/b0-audit')
rankings=list(root.glob('ranking-*/configuration_ranking.csv'))
assert rankings, 'Fetch completed ranking first'
ranking=max(rankings,key=lambda p:p.stat().st_mtime).parent
labels={s:('latest-best' if 'covariate_set=' in s else 'latest-control') for s in pd.read_csv(ranking/'configuration_ranking.csv').config_id}
summary={}
for name,keys in [('run_scores',['config_id','seed','geography']),('season_scores',['config_id','seed','geography','season','target'])]:
    old=pd.read_csv(out/f'{name}.csv');old=old[old.config_id.isin(labels.values())]
    new=pd.read_csv(ranking/f'{name}.csv');new.config_id=new.config_id.map(labels)
    paired=old.merge(new,on=keys,suffixes=('_before','_after'),validate='one_to_one')
    assert len(paired)==len(old)==len(new)
    metric='combined' if name=='run_scores' else 'wis_ratio'
    if name=='season_scores':
        assert np.array_equal(paired.n_before,paired.n_after)
        np.testing.assert_allclose(paired.ensemble_wis_before,paired.ensemble_wis_after,rtol=1e-12)
    paired['delta']=paired[f'{metric}_after']-paired[f'{metric}_before']
    paired.to_csv(out/f'normalization_{name}_paired.csv',index=False)
    if name=='run_scores':
        summary=paired.groupby(['config_id','geography']).agg(before=('combined_before','mean'),after=('combined_after','mean'),after_sd=('combined_after','std'),delta=('delta','mean'),improved_seeds=('delta',lambda x:int((x<0).sum())))
        summary.to_csv(out/'normalization_summary.csv');print(summary.to_string())
        allruns=paired[paired.geography=='all'].copy()
    else:
        print(paired[paired.geography=='all'].groupby(['config_id','season','target']).agg(before=('wis_ratio_before','mean'),after=('wis_ratio_after','mean'),delta=('delta','mean')).to_string())

fig,axes=plt.subplots(1,2,figsize=(9,4.4),sharey=True)
for ax,label,title in zip(axes,['latest-control','latest-best'],['No covariates','Best covariate formulation']):
    part=allruns[allruns.config_id==label]
    for row in part.itertuples():
        ax.plot([0,1],[row.combined_before,row.combined_after],'-o',alpha=.6,label=f'Seed {row.seed}')
    means=[part.combined_before.mean(),part.combined_after.mean()]
    ax.plot([0,1],means,'kD',markersize=8,label='Mean')
    for x,y in enumerate(means):ax.annotate(f'{y:.3f}',(x,y),xytext=(8,6),textcoords='offset points',weight='bold')
    ax.axhline(1,color='0.4',ls='--',label='Ensemble')
    ax.axhline(.8947671964566201,color='#227c9d',ls=':',label='Historical B0 pathogen')
    ax.set_xticks([0,1],['Before','Restored normalization']);ax.set_xlim(-.2,1.45)
    ax.set_title(title);ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Combined relative WIS (lower is better)')
axes[1].legend(fontsize=8,loc='best');fig.suptitle('Normalization-only intervention · matched seeds and frozen scoring')
fig.tight_layout();fig.savefig(out/'normalization_comparison.png',dpi=180)
print('\nPaired seeds:\n',allruns[['config_id','seed','combined_before','combined_after','delta']].to_string(index=False))
