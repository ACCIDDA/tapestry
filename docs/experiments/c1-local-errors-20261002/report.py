"""Compare C1 training treatments on identical real nowcast inputs and support."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chromantis.model.scenario import Scenario

root = Path('data/experiments/c1-local-errors-20261002')
out = Path(__file__).parent
rank = max(root.glob('ranking-*'), key=lambda p: p.stat().st_mtime)
settings = json.loads((root/'experiment.json').read_text())
source = json.loads((root/'augmentation-source.json').read_text())
previous = Path('data/experiments/b2-reporting-values-only-20261002')
reference = Path('data/experiments/b2-reporting-control-replay-20261001')
for folder in (previous, reference):
    old = json.loads((folder/'experiment.json').read_text())
    for key in ('dataset_sha256','frozen_manifest_sha256','population_sha256','eval_members'):
        if settings[key] != old[key]: raise ValueError('Comparison mismatch: '+key)
oldrank = max(previous.glob('ranking-*'), key=lambda p:p.stat().st_mtime)
keys = ['target','season','location','horizon','n']
first = json.loads((oldrank/'manifest.json').read_text())['runs'][0]
support = pd.read_csv(Path(first['path'])/'totals.csv')[keys].sort_values(keys).reset_index(drop=True)
for run in json.loads((rank/'manifest.json').read_text())['runs']:
    actual = pd.read_csv(Path(run['path'])/'totals.csv')[keys].sort_values(keys).reset_index(drop=True)
    if not support.equals(actual): raise ValueError('Frozen evaluation support differs')
labels = {run: ('Local seasonal log, half strength' if rec['strength']==.5 else
                {'local_log':'Local seasonal log', 'calendar_log':'Calendar log',
                 'local_additive':'Local seasonal additive'}[rec['method']])
          for run,rec in source['labels'].items()}
frames=[]
for name in ('season_composite_scores.csv','run_scores.csv'):
    d=pd.read_csv(rank/name)
    if 'season' not in d: d['season']='combined'
    d['training_treatment']=[labels[Scenario.from_string(c).run_id] for c in d.config_id]
    frames.append(d)
d=pd.concat(frames,ignore_index=True)
# Completed matched scores are copied from the original report, never refitted.
old=pd.read_csv(out/'reference-seeds.csv').query('arm == "nowcast"')
keys=['seed','geography','season']
w=d.merge(old[keys+['control','values_only']],on=keys,validate='many_to_one')
if len(w)!=108: raise ValueError(f'Expected 4 x 3 seeds x 3 geographies x 3 season summaries, got {len(w)}')
w['vs_finalized_training_pct']=100*(w.combined/w.control-1)
w['vs_previous_error_training_pct']=100*(w.combined/w.values_only-1)
w['model']='C1, pathogen MLP with distance spatial sharing, no finality channel'
w['training_seasons']=w.season.map({'2024-2025':'2022-23, 2023-24, 2025-26',
    '2025-2026':'2022-23, 2023-24, 2024-25','combined':'separate leave-one-season-out fits; see per-season rows'})
w['prediction_labels']='finalized future outcomes'
w['evaluation_inputs']='actual provisional reports corrected by causal adaptive-chain nowcaster'
w.to_csv(out/'paired-seed-scores.csv',index=False)
cols=['combined','control','values_only','vs_finalized_training_pct','vs_previous_error_training_pct']
s=w.groupby(['training_treatment','geography','season'])[cols].agg(['mean','std'])
s.columns=['_'.join(c) for c in s.columns];s=s.reset_index()
s.to_csv(out/'summary.csv',index=False)
pd.read_csv(rank/'season_scores.csv').to_csv(out/'target-season-scores.csv',index=False)
fig,axes=plt.subplots(1,2,figsize=(14,7),sharey=True)
order=['Local seasonal log','Local seasonal log, half strength','Calendar log','Local seasonal additive']
for ax,season,training in zip(axes,['2024-2025','2025-2026'],['2022–24 + 2025–26','2022–25']):
    q=s[s.geography.eq('all')&s.season.eq(season)].set_index('training_treatment').reindex(order)
    mean=[q.control_mean.iloc[0],q.values_only_mean.iloc[0],*q.combined_mean]
    sd=[q.control_std.iloc[0],q.values_only_std.iloc[0],*q.combined_std]
    ax.bar(np.arange(6),mean,yerr=sd,capsize=2,color=['#88949d','#c38756']+['#29847a']*4)
    ax.set_xticks(range(6),['Finalized\ntraining','Previous\nerror method','Local log','Local log\nhalf strength','Calendar\nlog','Local\nadditive'],rotation=30,ha='right')
    ax.axhline(1,color='gray',ls='--',lw=.8)
    ax.set_title(f'Evaluate {season}; train {training}\n'+('Retrospective fold' if season=='2024-2025' else 'Forward fold'))
axes[0].set_ylabel('Weighted WIS / matched Hub WIS (lower is better)')
fig.suptitle('C1 retrained with finalized future labels; actual nowcast-corrected evaluation inputs\nError treatments modify historical training inputs only; three-seed means ± SD')
fig.tight_layout();fig.savefig(out/'forecast-comparison.png',dpi=180);plt.close(fig)
rows=['| Training-input error treatment | Evaluation season | C1 relative WIS | Finalized-training C1 | Previous error-training C1 | Change vs finalized training |',
      '| --- | --- | ---: | ---: | ---: | ---: |']
for r in s[s.geography.eq('all')].itertuples():
    rows.append(f'| {r.training_treatment} | {r.season} | {r.combined_mean:.4f} | {r.control_mean:.4f} | {r.values_only_mean:.4f} | {r.vs_finalized_training_pct_mean:+.1f}% |')
p=out/'index.md'
p.write_text(p.read_text().split('\n<!-- results -->')[0]+'\n<!-- results -->\n## Results\n\n'+
    '\n'.join(rows)+'\n\nLower WIS is better; 1 equals the matched Hub ensemble. Changes average paired seed ratios. '+
    'All rows use C1 and finalized future prediction labels. For evaluation in 2024–25, train on 2022–23, 2023–24, and 2025–26; '+
    'for evaluation in 2025–26, train on 2022–23, 2023–24, and 2024–25. Evaluation inputs are actual nowcast-corrected reports.\n\n'+
    '![C1 training treatments](forecast-comparison.png)\n\n[Paired seeds](paired-seed-scores.csv) · [Summary](summary.csv) · [Target scores](target-season-scores.csv).\n')
print(s[s.geography.eq('all')].to_string(index=False))
