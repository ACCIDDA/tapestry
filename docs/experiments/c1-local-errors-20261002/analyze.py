"""Analyze completed C1 folds; reads scores only, without fitting or scoring models."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tapestry.model.scenario import Scenario

out=Path(__file__).parent
w=pd.read_csv(out/'paired-seed-scores.csv')
s=pd.read_csv(out/'summary.csv')
a=pd.read_csv(out/'target-season-scores.csv')
b=pd.read_csv(out/'finalized-training-target-scores.csv')
label=lambda s: 'Local log half' if s.reporting_strength==.5 else {'local_log':'Local log','calendar_log':'Calendar log','local_additive':'Local additive'}[s.reporting_method]
a['treatment']=[label(Scenario.from_string(c)) for c in a.config_id]
b['treatment']='Finalized training'
fields=['wis_ratio','model_coverage_90','model_underprediction_ratio','model_overprediction_ratio','model_dispersion_ratio']
t=pd.concat([a,b]).groupby(['treatment','geography','season','target'])[fields].mean().reset_index()
t.to_csv(out/'target-diagnostics.csv',index=False)
forward=t.query('geography == "all" and season == "2025-2026"')
x=forward[forward.treatment.eq('Local log half')].merge(forward[forward.treatment.eq('Finalized training')],on=['geography','season','target'],suffixes=('_half','_final'))
x['target_weight']=np.where(x.target.str.contains('hosp'),1.,.5)
x['weighted_score_increase']=(x.wis_ratio_half-x.wis_ratio_final)*x.target_weight/x.target_weight.sum()
x['share_of_total_increase_pct']=100*x.weighted_score_increase/x.weighted_score_increase.sum()
x.to_csv(out/'half-strength-forward-attribution.csv',index=False)
print('Share of forward loss from COVID targets:',x.loc[x.target.str.contains('covid'),'share_of_total_increase_pct'].sum())
print(x[['target','wis_ratio_final','wis_ratio_half','weighted_score_increase','share_of_total_increase_pct']].to_string(index=False))

order=['Local seasonal log','Local seasonal log, half strength','Calendar log','Local seasonal additive']
fig,axes=plt.subplots(1,2,figsize=(13,6),sharey=True)
for ax,season,history,donors in zip(axes,['2024-2025','2025-2026'],['2022–23, 2023–24, 2025–26','2022–23, 2023–24, 2024–25'],['2025–26','2024–25']):
    for i,seed in enumerate([42,43,44]):
        q=w[(w.geography=='all')&(w.season==season)&(w.seed==seed)].set_index('training_treatment').reindex(order)
        ax.scatter(np.arange(4)+(i-1)*.12,q.vs_finalized_training_pct,label=f'Seed {seed}',s=48)
    ax.axhline(0,color='black',lw=1)
    ax.set_xticks(range(4),['Local log','Local log\nhalf strength','Calendar log','Local additive'])
    ax.set_title(f'Evaluate {season}\nTrain C1 on {history}\nDonor errors: {donors}',fontsize=10)
    ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('WIS change vs C1 trained on finalized inputs (%)\nNegative favors error training')
axes[1].legend()
fig.suptitle('C1 retrained with historical input errors: seed-by-seed effects\nFinalized future labels; actual nowcast-corrected evaluation inputs; native availability retained')
fig.tight_layout();fig.savefig(out/'paired-seed-changes.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(figsize=(10,6))
order_targets=['wk inc flu hosp','wk inc covid hosp','wk inc rsv hosp','wk inc flu prop ed visits','wk inc covid prop ed visits','wk inc rsv prop ed visits']
q=x.set_index('target').reindex(order_targets)
for offset,field,label_,color in [(-.18,'wis_ratio_final','C1 trained on finalized inputs','#88949d'),(.18,'wis_ratio_half','C1 trained with half-strength local log errors','#29847a')]:
    ax.bar(np.arange(6)+offset,q[field],width=.35,label=label_,color=color)
ax.axhline(1,color='gray',ls='--',lw=.8)
ax.set_xticks(range(6),['Flu\nadmissions','COVID\nadmissions','RSV\nadmissions','Flu ED','COVID ED','RSV ED'])
ax.set_ylabel('WIS / matched Hub WIS (lower is better)')
ax.legend(fontsize=9)
ax.set_title('C1 evaluated on actual nowcast-corrected reports in 2025–26\nTrain on 2022–23, 2023–24, 2024–25; finalized future labels\nError treatment uses 2024–25 historical residuals; three-seed means',fontsize=11)
fig.tight_layout();fig.savefig(out/'forward-target-comparison.png',dpi=180);plt.close(fig)

summary=w[w.geography.eq('all')].groupby(['training_treatment','season']).agg(
    mean_change=('vs_finalized_training_pct','mean'),
    best_seed_change=('vs_finalized_training_pct','min'),
    worst_seed_change=('vs_finalized_training_pct','max'),
    seeds_better=('vs_finalized_training_pct',lambda v:int((v<0).sum())))
summary.to_csv(out/'seed-consistency.csv')
print(summary.to_string())
