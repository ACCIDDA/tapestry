"""Summarize completed augmentation fits using the shared manager's WIS scores."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tapestry.model.scenario import Scenario

out = Path(__file__).parent
root = Path('data/experiments/b2-reporting-augmentation-20261001')
ranked = max(root.glob('ranking-*'), key=lambda p: p.stat().st_mtime)
labels = json.loads((root/'augmentation-source.json').read_text())['labels']
records = []
for name in ('season_composite_scores.csv', 'run_scores.csv'):
    d = pd.read_csv(ranked/name)
    if 'season' not in d:
        d['season'] = 'combined'
    tags = [labels[Scenario.from_string(c).run_id] for c in d.config_id]
    d['model'] = [t['label'] for t in tags]
    d['arm'] = [t['arm'] for t in tags]
    records.append(d)
d = pd.concat(records, ignore_index=True)
w = d.pivot(index=['model','seed','geography','season'],columns='arm',values='combined').reset_index()
if len(w) != 135 or w[['vintage','nowcast']].isna().any().any():
    raise ValueError('Expected five models, three seeds, three geographies and three season summaries')
w['nowcast_vs_vintage_pct'] = 100*(w.nowcast/w.vintage-1)
w.to_csv(out/'paired-seed-scores.csv',index=False)
summary = w.groupby(['model','geography','season'])[['vintage','nowcast','nowcast_vs_vintage_pct']].agg(['mean','std'])
summary.columns = ['_'.join(c) for c in summary.columns]
summary = summary.reset_index()
summary.to_csv(out/'augmentation-comparison.csv',index=False)
target = pd.read_csv(ranked/'season_scores.csv')
tags = [labels[Scenario.from_string(c).run_id] for c in target.config_id]
target['model'], target['arm'] = [t['label'] for t in tags], [t['arm'] for t in tags]
target.to_csv(out/'target-season-scores.csv',index=False)
models = [f'C{i}' for i in range(1,6)]
fig, axes = plt.subplots(1,3,figsize=(15,5),sharey=True)
for ax, season in zip(axes,['2025-2026','2024-2025','combined']):
    subset = summary[summary.geography.eq('all') & summary.season.eq(season)].set_index('model').reindex(models)
    for i,(arm,label,color) in enumerate([('vintage','Raw-error training → raw vintages','#bd753d'),
                                        ('nowcast','Residual training → corrected vintages','#168880')]):
        ax.bar(np.arange(5)+(i-.5)*.36,subset[arm+'_mean'],width=.34,
               yerr=subset[arm+'_std'],capsize=2,label=label,color=color)
    ax.axhline(1,color='grey',ls='--'); ax.set_xticks(range(5),models)
    ax.set_title(season+(' · forward' if season=='2025-2026' else ' · retrospective' if season=='2024-2025' else ''))
axes[0].set_ylabel('WIS / matched Hub WIS (lower is better)')
axes[1].legend(fontsize=8)
fig.suptitle('B2 retrained on stochastic reporting errors · real held-out vintages\nThree-seed means ± seed SD; not independent-season confidence intervals')
fig.tight_layout(); fig.savefig(out/'augmentation-comparison.png',dpi=180); plt.close(fig)
rows = ['| Model | Season | Raw-error training | Residual-error training | Residual vs raw |',
        '| --- | --- | ---: | ---: | ---: |']
for row in summary[summary.geography.eq('all')].itertuples():
    rows.append(f'| {row.model} | {row.season} | {row.vintage_mean:.4f} | {row.nowcast_mean:.4f} | {row.nowcast_vs_vintage_pct_mean:+.1f}% |')
content = '\n'.join(rows)+'\n\n![Real-vintage forecast scores](augmentation-comparison.png)\n\n[Paired seed scores](paired-seed-scores.csv) · [All geographies](augmentation-comparison.csv) · [Target scores](target-season-scores.csv).\n'
# Prior corrected-encoding replay is an available fixed-checkpoint baseline for C1–C3.
old = Path('data/experiments/b2-flag-controls-20261001')
if list(old.glob('ranking-*')):
    r = max(old.glob('ranking-*'),key=lambda p:p.stat().st_mtime)
    old_labels = json.loads((old/'replay-source.json').read_text())['labels']
    from dataclasses import replace
    baseline=[]
    for name in ('season_composite_scores.csv','run_scores.csv'):
        b=pd.read_csv(r/name)
        if 'season' not in b: b['season']='combined'
        scenarios=[Scenario.from_string(c) for c in b.config_id]
        b['model']=[old_labels[replace(s,replay_from='',replay_inputs='none',replay_flags='native').run_id] for s in scenarios]
        b['arm']=[s.replay_inputs for s in scenarios]
        b=b[b.arm.isin(['vintage','nowcast'])]
        baseline.append(b)
    paired=d.merge(pd.concat(baseline),on=['model','seed','geography','season','arm'],suffixes=('_augmented','_fixed'))
    paired['change_pct']=100*(paired.combined_augmented/paired.combined_fixed-1)
    paired.to_csv(out/'versus-fixed-checkpoints.csv',index=False)
    comparison=paired.groupby(['model','geography','season','arm'])[['combined_augmented','combined_fixed','change_pct']].mean().reset_index()
    comparison.to_csv(out/'versus-fixed-summary.csv',index=False)
    content+='\n[C1–C3 versus earlier fixed-checkpoint replay](versus-fixed-summary.csv). This comparison changes training and removes the finality channel; it does not isolate augmentation alone.\n'
p=out/'index.md'
p.write_text(p.read_text().split('\n<!-- results:start -->')[0]+'\n<!-- results:start -->\n## Completed forecast scores\n\n'+content)
print(summary[summary.geography.eq('all')].to_string(index=False))
