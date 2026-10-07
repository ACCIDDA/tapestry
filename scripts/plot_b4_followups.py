from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=Path('docs/experiments/b4-flu-heads-20261006/analysis')
d=pd.read_csv(p/'labeled-summary.csv');d=d[d.history=='corrected']
variants=['samples','samples_sum_wis','quantiles','compact_quantiles'];labels=['Samples','Samples + total WIS','23 quantiles','Compact quantiles']
colors=['#2166ac','#1b9e77','#d95f02','#7570b3']
fig,axes=plt.subplots(2,3,figsize=(12,7),sharey=False)
for i,(anchor,title) in enumerate([('flu_native','MLP trained on tree-corrected synthetic histories + Kinsa'),('flu_admissions_log','MLP trained on reporting errors + recent-history reconstruction')]):
 t=d[d.anchor==anchor].set_index('variant').loc[variants]
 for j,(metric,label) in enumerate([('flu_admissions_native','Admissions: native relative WIS'),('flu_admissions_log','Admissions: log relative WIS'),('flu_ed_native','ED: forward-season relative WIS')]):
  ax=axes[i,j];vals=t[metric].values;ax.barh(labels,vals,color=colors);ax.invert_yaxis();ax.axvline(1,color='gray',linestyle='--');ax.set_xlim(0,1.22)
  for k,v in enumerate(vals):ax.text(v+.012,k,f'{v:.3f}',va='center',fontsize=9)
  ax.set_title(label,fontsize=10);ax.spines[['top','right']].set_visible(False)
  if j>0:ax.set_yticklabels([])
 axes[i,1].text(.5,1.22,title,transform=axes[i,1].transAxes,ha='center',fontsize=11,fontweight='bold')
fig.suptitle('Matched output heads: lower WIS is better',fontsize=15,y=.99)
fig.text(.04,.02,'Mean of seeds 42/43. Finalized flu labels; training 2022–25 → evaluation 2025–26, and 2022–24 + 2025–26 → 2024–25.\nEvaluation: Wednesday reports, newest two admission weeks tree-corrected, ED reported. Admissions seasons equal; ED relative score has 2025–26 only.',fontsize=9)
fig.tight_layout(rect=[0,.09,1,.93]);fig.savefig(p/'head-comparison.png',dpi=180);fig.savefig(p/'head-comparison.pdf');plt.close(fig)
g=pd.read_csv(p/'distribution-geography.csv');g=g[(g.history=='corrected')&(g.target=='flu_admissions')&(g.anchor=='flu_admissions_log')&g.variant.isin(['samples','samples_sum_wis'])]
fig,axes=plt.subplots(1,2,figsize=(10,4.6));x=np.arange(4)
keys=[('2024-2025','states_dc'),('2025-2026','states_dc'),('2024-2025','US'),('2025-2026','US')]
a=g.groupby(['season','geography','variant'])[['four_week_sum_wis','four_week_sum_coverage_90']].mean()
change=[100*(a.loc[(s,geo,'samples_sum_wis'),'four_week_sum_wis']/a.loc[(s,geo,'samples'),'four_week_sum_wis']-1) for s,geo in keys]
axes[0].bar(x,change,color='#1b9e77');axes[0].axhline(0,color='gray');axes[0].set_ylabel('Change in total WIS (%)');axes[0].set_title('Adding sum WIS improves cumulative forecasts')
for k,v in enumerate(change):axes[0].text(k,v-1,f'{v:.1f}%',ha='center',va='top')
for offset,var,label,color in [(-.18,'samples','Samples','#2166ac'),(.18,'samples_sum_wis','Samples + total WIS','#1b9e77')]:
 axes[1].bar(x+offset,[100*a.loc[(s,geo,var),'four_week_sum_coverage_90'] for s,geo in keys],width=.34,label=label,color=color)
axes[1].axhline(90,color='gray',linestyle='--');axes[1].set_ylim(0,105);axes[1].set_ylabel('90% total-interval coverage (%)');axes[1].legend(fontsize=8)
for ax in axes:ax.set_xticks(x,['2024–25\nstates/DC','2025–26\nstates/DC','2024–25\nUS','2025–26\nUS']);ax.spines[['top','right']].set_visible(False)
fig.suptitle('MLP trained with reporting errors + reconstruction: four-week admissions',fontsize=12)
fig.text(.025,.02,'Same finalized labels, folds and corrected evaluation inputs as the head comparison. Two-seed means.\nTotals sum intact trajectories; only fully observed windows. States/DC and US shown separately. Diagnostic support differs from frozen Hub support.',fontsize=8)
fig.tight_layout(rect=[0,.13,1,.93]);fig.savefig(p/'cumulative-comparison.png',dpi=180);fig.savefig(p/'cumulative-comparison.pdf')
