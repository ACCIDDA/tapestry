"""Paired augmentation-versus-finalized-training comparison, same no-finality architecture."""
from pathlib import Path
from dataclasses import replace
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tapestry.model.scenario import Scenario
from tapestry.evaluation.totals import season_scores

out = Path(__file__).parent
roots = dict(augmented=Path('data/experiments/b2-reporting-augmentation-20261001'),
             control=Path('data/experiments/b2-reporting-control-replay-20261001'))
settings = [json.loads((p/'experiment.json').read_text()) for p in roots.values()]
for key in ('dataset_sha256','frozen_manifest_sha256','population_sha256','eval_members'):
    if settings[0][key] != settings[1][key]:
        raise ValueError('Experiments have different '+key)
records, horizon_records = [], []
reference_support = None
for treatment, root in roots.items():
    ranking = max(root.glob('ranking-*'), key=lambda p: p.stat().st_mtime)
    labels = json.loads((root/('augmentation-source.json' if treatment=='augmented' else 'replay-source.json')).read_text())['labels']
    def identity(config):
        s = Scenario.from_string(config)
        if treatment == 'augmented':
            tag = labels[s.run_id]
            return tag['label'], tag['arm']
        return labels[replace(s,replay_from='',replay_inputs='none',replay_flags='native').run_id],s.replay_inputs
    for filename in ('season_composite_scores.csv','run_scores.csv'):
        d = pd.read_csv(ranking/filename)
        if 'season' not in d: d['season']='combined'
        tags=[identity(c) for c in d.config_id]
        d['model'],d['arm'],d['treatment']=[t[0] for t in tags],[t[1] for t in tags],treatment
        records.append(d)
    for run in json.loads((ranking/'manifest.json').read_text())['runs']:
        totals = pd.read_csv(Path(run['path'])/'totals.csv')
        keys=['target','season','location','horizon','n']
        support=totals[keys].sort_values(keys).reset_index(drop=True)
        if reference_support is None: reference_support=support
        elif not support.equals(reference_support): raise ValueError('Comparison has different frozen support')
        totals['config_id'],totals['seed']=run['config_id'],run['seed']
        model,arm=identity(run['config_id'])
        for horizon, part in totals.groupby('horizon'):
            h=season_scores(part)
            h['model'],h['arm'],h['treatment'],h['horizon']=model,arm,treatment,horizon
            horizon_records.append(h)
d=pd.concat(records,ignore_index=True)
w=d.pivot(index=['model','arm','seed','geography','season'],columns='treatment',values='combined').reset_index()
if len(w)!=270 or w[['augmented','control']].isna().any().any():
    raise ValueError('Matched five-model, two-arm, three-seed comparison is incomplete')
w['change_pct']=100*(w.augmented/w.control-1)
w.to_csv(out/'matched-control-seeds.csv',index=False)
s=w.groupby(['model','arm','geography','season'])[['augmented','control','change_pct']].agg(['mean','std'])
s.columns=['_'.join(c) for c in s.columns]
s=s.reset_index();s.to_csv(out/'matched-control-summary.csv',index=False)
h=pd.concat(horizon_records,ignore_index=True)
h.to_csv(out/'horizon-calibration-seeds.csv',index=False)
metrics=['wis_ratio','model_coverage_50','model_coverage_80','model_coverage_90','model_coverage_95',
         'model_underprediction_ratio','model_overprediction_ratio','model_dispersion_ratio']
h.groupby(['model','arm','treatment','geography','season','target','horizon'])[metrics].mean().to_csv(out/'horizon-calibration.csv')
fig,axes=plt.subplots(2,2,figsize=(13,8),sharey=True)
models=[f'C{i}' for i in range(1,6)]
for row,season in enumerate(['2025-2026','2024-2025']):
    for col,arm in enumerate(['vintage','nowcast']):
        ax=axes[row,col]
        q=s[s.geography.eq('all')&s.season.eq(season)&s.arm.eq(arm)].set_index('model').reindex(models)
        for i,(field,label,color) in enumerate([('control','Finalized training','#85929c'),('augmented','Reporting-error training','#17877e')]):
            ax.bar(np.arange(5)+(i-.5)*.36,q[field+'_mean'],width=.34,yerr=q[field+'_std'],capsize=2,label=label,color=color)
        ax.set_xticks(range(5),models);ax.axhline(1,color='grey',ls='--',lw=.8)
        ax.set_title(season+(' · forward' if row==0 else ' · retrospective')+' · '+arm)
        ax.set_ylabel('WIS / matched Hub WIS')
axes[0,0].legend()
fig.suptitle('Matched architecture: does reporting-error training help on real vintages?\nBoth treatments omit finality; mean ± SD across three seeds')
fig.tight_layout();fig.savefig(out/'matched-controls.png',dpi=180);plt.close(fig)
rows=['| Model | Input treatment | Season | Finalized training | Error-augmented training | Change |',
      '| --- | --- | --- | ---: | ---: | ---: |']
for r in s[s.geography.eq('all')].itertuples():
    rows.append(f'| {r.model} | {r.arm} | {r.season} | {r.control_mean:.4f} | {r.augmented_mean:.4f} | {r.change_pct_mean:+.1f}% |')
text='\n'.join(rows)+'\n\n![Matched controls](matched-controls.png)\n\n[Paired seeds](matched-control-seeds.csv) · [All geographies](matched-control-summary.csv) · [Horizon, coverage and WIS-component diagnostics](horizon-calibration.csv).\n\nNegative changes favor augmentation. Both treatments use the same no-finality architecture, seed, training history, mask recipe and frozen evaluation support. Reporting augmentation also changes the synthetic validation input treatment; this comparison measures the complete training intervention. Three seeds do not provide independent-season replication.\n'
p=out/'index.md'
p.write_text(p.read_text().split('\n<!-- controls:start -->')[0]+'\n<!-- controls:start -->\n## Matched finalized-training controls\n\n'+text)
print(s[s.geography.eq('all')].to_string(index=False))
