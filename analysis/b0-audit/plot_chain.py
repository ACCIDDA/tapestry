"""Single consecutive comparison, not a collection of branching experiment plots."""
from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path('docs/results/b1-to-b0-chain')
f=pd.read_csv(out/'chain_table.csv',dtype={'stage':str});f=f[f.stage!='09'].reset_index(drop=True);seeds=pd.read_csv(out/'run_scores.csv')
labels=['B1 recipe','Normalize','No artificial\nmasking','No finality\nflag','Complete forecast\ninputs','Complete training\nhistory','B0 validation\nweeks','B0 validation\ndraws','B0 error weight\n(B0 endpoint)']
fig,ax=plt.subplots(figsize=(14,5))
for seed in [42,43,44]:
 s=seeds.loc[(seeds.seed==seed)&(seeds.geography=='all')].set_index('config_id').combined
 y=[s[f'stage{x}'] for x in f.stage]
 ax.plot(range(len(f)),y,color='#8997a7',alpha=.45,lw=.8,marker='.',ms=4)
ax.plot(range(len(f)),f['mean'],color='#174b70',lw=2.4,marker='o',label='Overall mean (three fits)')
ax.plot(range(len(f)),f['season2025_mean'],color='#b76621',lw=2.2,marker='s',label='2025–26 mean (three fits)')
ax.axhline(1,color='#666666',ls='--',lw=1,label='Hub ensemble')
ax.set_xticks(range(len(f)),labels,rotation=20,ha='right');ax.set_ylabel('Relative WIS (lower is better)')
ax.set_title('One consecutive path from the B1-derived recipe to B0\nEach point changes only the named choice from the preceding point')
for i,r in f.iterrows():ax.annotate(f'{r["mean"]:.3f}',(i,r['mean']),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.legend(frameon=False);fig.tight_layout();fig.savefig(out/'chain.png',dpi=180);fig.savefig(out/'chain.svg');plt.close(fig)
