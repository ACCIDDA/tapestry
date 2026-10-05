"""Matched forecast/reconstruction tradeoffs across strength and joint uncertainty."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tapestry.model.scenario import Scenario
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output
f=pd.read_csv(out/'forecast-seed-scores.csv')
f=f[f.geography.eq('all')].copy()
scenarios=f.config_id.map(Scenario.from_string)
f['strength']=[s.replay_strength if s.replay_nowcaster=='context_residual' else 0 for s in scenarios]
f['uncertainty']=[s.replay_uncertainty for s in scenarios]
f=f[[s.replay_inputs=='nowcast' for s in scenarios]].copy()
keys=['season','stratum','seed']
base=f[f.strength.eq(0)&f.uncertainty.eq(0)][keys+['combined']].rename(columns={'combined':'control'})
f=f.merge(base,on=keys,validate='many_to_one')
f['wis_change_pct']=100*(f.combined/f.control-1)
f.to_csv(out/'tradeoff-seeds.csv',index=False)
r=f.groupby(['season','stratum','strength','uncertainty']).agg(wis_change_pct=('wis_change_pct','mean'),
    min_seed=('wis_change_pct','min'),max_seed=('wis_change_pct','max'),wis=('combined','mean')).reset_index()
r.to_csv(out/'tradeoff.csv',index=False)
print(r[r.season.eq('2025-2026')&~r.stratum.eq('all')].to_string(index=False))
fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
for ax,stratum,title in zip(axes,['complete_history_12','complete_and_uninterrupted'],['Complete histories','Also eight uninterrupted reports']):
    g=r[r.season.eq('2025-2026')&r.stratum.eq(stratum)]
    for u,color in zip(sorted(g.uncertainty.unique()),['#737d89','#466183','#16877f','#844785']):
        q=g[g.uncertainty.eq(u)].sort_values('strength')
        ax.plot(q.strength,q.wis_change_pct,'o-',color=color,label=f'Uncertainty ×{u:g}')
    ax.axhline(0,color='#222',lw=1);ax.set_xticks(sorted(g.strength.unique()))
    ax.set_title(title);ax.set_xlabel('Fraction of age-dependent admissions correction');ax.grid(alpha=.15)
axes[0].set_ylabel('C1 WIS change versus seasonal point nowcast (%)\nNegative is better');axes[1].legend()
fig.suptitle('Reconstructed-history tradeoff · 2025–26 · three paired C1 seeds\nForecaster weights, input encoding and scored cells held fixed')
fig.tight_layout();fig.savefig(out/'forecast-tradeoff.png',dpi=180);plt.close(fig)
