"""Plot completed reproduction and training controls without inspecting images."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
out=Path('docs/results/b0-reproduction')
r=pd.read_csv(out/'configuration_ranking.csv').set_index('config_id')
labels={'historical-b0-target100':'B0 target · saved',
        'reproduced-b0-target100':'B0 target · fresh reproduction',
        'historical-b0-pathogen300':'B0 pathogen · saved',
        'reproduced-b0-pathogen300':'B0 pathogen · fresh reproduction',
        'old-training-l40':'B0 training on L40 · finalized',
        'current-unscaled-finalized':'Current training · finalized',
        'current-normalized-finalized':'Current training + normalization · finalized',
        'current-normalized-wednesday':'Same training + normalization · Wednesday mask'}
fig,ax=plt.subplots(figsize=(10,5))
for i,(name,label) in enumerate(labels.items()):
    row=r.loc[name];color='#227c9d' if 'b0-' in name else '#c45b34'
    ax.errorbar(row.combined_mean,i,xerr=row.combined_sd,fmt='o',capsize=4,color=color)
    ax.annotate(f'{row.combined_mean:.3f}',(row.combined_mean,i),xytext=(7,-15),textcoords='offset points')
ax.axvline(1,color='0.4',ls='--');ax.set_yticks(range(len(labels)),labels.values());ax.invert_yaxis()
ax.set_xlabel('Relative WIS · mean ± seed SD (lower is better)')
ax.set_title('B0 reproduction and controlled training comparisons')
ax.spines[['top','right']].set_visible(False);fig.tight_layout()
fig.savefig(out/'combined.png',dpi=180);fig.savefig(out/'combined.svg');plt.close(fig)
s=pd.read_csv(out/'season_scores.csv');s=s[s.geography=='all']
order=['old-training-l40','current-unscaled-finalized','current-normalized-finalized','current-normalized-wednesday']
wide=s.groupby(['season','target','config_id']).wis_ratio.mean().unstack('config_id')[order]
fig,ax=plt.subplots(figsize=(10,6));im=ax.imshow(wide.to_numpy(),vmin=.7,vmax=1.3,cmap='RdBu_r',aspect='auto')
ax.set_xticks(range(4),['B0 / L40','Current / final','Normalized / final','Normalized / Wednesday'])
ax.set_yticks(range(len(wide)),[f"{season}  {target.replace('wk inc ','').replace('prop ed visits','ED').replace('hosp','admissions')}" for season,target in wide.index])
for i in range(len(wide)):
    for j in range(4):
        v=wide.iloc[i,j];ax.text(j,i,f'{v:.3f}',ha='center',va='center',color='white' if v>1.22 or v<.78 else 'black')
ax.set_title('Pathogen models on identical frozen target–season support')
fig.colorbar(im,ax=ax,label='Relative WIS (color capped at 1.3)',shrink=.7);fig.tight_layout();fig.savefig(out/'target-season.png',dpi=180)
print(out/'combined.png')
paired=pd.read_csv(out/'paired_seed_deltas.csv')
paired=paired[paired.geography=='all']
fig,axes=plt.subplots(1,3,figsize=(11,4),sharey=True)
for ax,effect,title in zip(axes,['training-code','normalization','availability'],
                         ['Training code','Restore normalization','Wednesday availability']):
    for row in paired[paired.effect==effect].itertuples():
        ax.plot([0,1],[row.combined_before,row.combined_after],marker='o',label=f'Seed {row.seed}')
    ax.axhline(1,color='0.5',ls='--');ax.set_xticks([0,1],['Before','After'])
    ax.set_title(title);ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Combined relative WIS (lower is better)');axes[-1].legend(frameon=False)
fig.suptitle('Paired controls · L40 · three seeds · 2,048 evaluation draws')
fig.tight_layout();fig.savefig(out/'paired-controls.png',dpi=180);plt.close(fig)
