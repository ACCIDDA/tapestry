"""C1 missingness ablation against full augmentation and matched finalized training."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chromantis.model.scenario import Scenario

root=Path('data/experiments/b2-reporting-values-only-20261002')
out=Path(__file__).parent
ranked=max(root.glob('ranking-*'),key=lambda p:p.stat().st_mtime)
settings=json.loads((root/'experiment.json').read_text())
reference=Path('data/experiments/b2-reporting-control-replay-20261001')
original=json.loads((reference/'experiment.json').read_text())
for key in ('dataset_sha256','frozen_manifest_sha256','population_sha256','eval_members'):
    if settings[key]!=original[key]:raise ValueError('Comparison settings differ: '+key)
reference_rank=max(reference.glob('ranking-*'),key=lambda p:p.stat().st_mtime)
reference_run=json.loads((reference_rank/'manifest.json').read_text())['runs'][0]
keys_support=['target','season','location','horizon','n']
support=pd.read_csv(Path(reference_run['path'])/'totals.csv')[keys_support].sort_values(keys_support).reset_index(drop=True)
for run in json.loads((ranked/'manifest.json').read_text())['runs']:
    actual=pd.read_csv(Path(run['path'])/'totals.csv')[keys_support].sort_values(keys_support).reset_index(drop=True)
    if not actual.equals(support):raise ValueError('Different frozen support')
frames=[]
for name in ('season_composite_scores.csv','run_scores.csv'):
    d=pd.read_csv(ranked/name)
    if 'season' not in d:d['season']='combined'
    d['arm']=[Scenario.from_string(s).reporting_augmentation for s in d.config_id]
    frames.append(d)
d=pd.concat(frames,ignore_index=True).rename(columns={'combined':'values_only'})
keys=['arm','seed','geography','season']
old=pd.read_csv(out/'matched-control-seeds.csv').query('model == "C1"')
w=old.merge(d[keys+['values_only']],on=keys,validate='one_to_one')
if len(w)!=54 or w[['control','augmented','values_only']].isna().any().any():raise ValueError('Incomplete C1 comparison')
w['values_only_vs_control_pct']=100*(w.values_only/w.control-1)
w['values_only_vs_full_pct']=100*(w.values_only/w.augmented-1)
w.to_csv(out/'values-only-paired-seeds.csv',index=False)
cols=['control','augmented','values_only','values_only_vs_control_pct','values_only_vs_full_pct']
s=w.groupby(['arm','geography','season'])[cols].agg(['mean','std'])
s.columns=['_'.join(c) for c in s.columns];s=s.reset_index()
s.to_csv(out/'values-only-summary.csv',index=False)
t=pd.read_csv(ranked/'season_scores.csv')
t['arm']=[Scenario.from_string(c).reporting_augmentation for c in t.config_id]
t.to_csv(out/'values-only-target-scores.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(12,5),sharey=True)
for ax,season in zip(axes,['2025-2026','2024-2025']):
    q=s[s.geography.eq('all')&s.season.eq(season)].set_index('arm').reindex(['vintage','nowcast'])
    for i,(field,label,color) in enumerate([('control','Finalized training','#8b959c'),('augmented','Errors + missingness','#c98138'),('values_only','Numerical errors only','#12847e')]):
        ax.bar(np.arange(2)+(i-1)*.25,q[field+'_mean'],width=.23,yerr=q[field+'_std'],capsize=2,label=label,color=color)
    ax.set_xticks(range(2),['Raw vintages','Corrected vintages']);ax.axhline(1,color='grey',ls='--',lw=.8)
    ax.set_title(season+(' · forward' if season=='2025-2026' else ' · retrospective'))
axes[0].set_ylabel('WIS / matched Hub WIS');axes[1].legend(fontsize=8)
fig.suptitle('C1: does transporting old reporting availability cause the loss?\nThree-seed means ± SD; same final labels and real evaluation vintages')
fig.tight_layout();fig.savefig(out/'values-only-comparison.png',dpi=180);plt.close(fig)
rows=['| Season | Input treatment | Finalized training | Errors + missingness | Errors only | Errors only vs control |',
      '| --- | --- | ---: | ---: | ---: | ---: |']
for r in s[s.geography.eq('all')].itertuples():
    rows.append(f'| {r.season} | {r.arm} | {r.control_mean:.4f} | {r.augmented_mean:.4f} | {r.values_only_mean:.4f} | {r.values_only_vs_control_pct_mean:+.1f}% |')
text='\n'.join(rows)+'\n\n![C1 numerical-error-only comparison](values-only-comparison.png)\n\n[Paired seed changes](values-only-paired-seeds.csv) · [All geographies](values-only-summary.csv) · [Target scores](values-only-target-scores.csv).\n'
p=out/'index.md';p.write_text(p.read_text().split('\n<!-- values-only:start -->')[0]+'\n<!-- values-only:start -->\n## Numerical-error-only follow-up\n\n'+text)
print(s[s.geography.eq('all')].to_string(index=False))
