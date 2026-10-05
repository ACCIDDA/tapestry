"""Paired flag-encoding diagnostics, using standard manager WIS aggregation."""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tapestry.model.scenario import Scenario

out=Path(__file__).parent
roots=[Path('data/experiments')/s for s in ('b2-input-replay-20261001','b2-flag-controls-20261001')]
labels=json.loads((roots[0]/'replay-source.json').read_text())['labels']
records=[];all_runs=[]
for root in roots:
    ranked=max(root.glob('ranking-*'),key=lambda p:p.stat().st_mtime)
    all_runs.extend(json.loads((ranked/'manifest.json').read_text())['runs'])
    for filename in ('season_composite_scores.csv','run_scores.csv'):
        d=pd.read_csv(ranked/filename)
        if 'season' not in d:d['season']='combined'
        scenarios=d.config_id.map(Scenario.from_string)
        d['model']=[labels[replace(s,replay_from='',replay_inputs='none',replay_flags='native').run_id] for s in scenarios]
        d['condition']=[s.replay_inputs+'_'+('available' if s.replay_inputs=='finalized' else 'off') if s.replay_flags=='native'
                        else s.replay_inputs+'_'+s.replay_flags for s in scenarios]
        records.append(d)
d=pd.concat(records,ignore_index=True)
keys=['model','seed','geography','season']
w=d.pivot(index=keys,columns='condition',values='combined').reset_index()
conditions=['finalized_available','finalized_off','vintage_off','vintage_available','nowcast_off','nowcast_available']
if w[conditions].isna().any().any() or len(w)!=81:raise ValueError('Incomplete paired flag controls')
for name in conditions[1:]:w[name+'_loss_pct']=100*(w[name]/w.finalized_available-1)
w['corrected_nowcast_vs_vintage_pct']=100*(w.nowcast_available/w.vintage_available-1)
w.to_csv(out/'paired-scores.csv',index=False)
cols=[*conditions,*[c+'_loss_pct' for c in conditions[1:]],'corrected_nowcast_vs_vintage_pct']
s=w.groupby(['model','geography','season'])[cols].mean().reset_index();s.to_csv(out/'encoding-comparison.csv',index=False)
fig,axes=plt.subplots(1,3,figsize=(14,5),sharey=True)
models=['C1','C2','C3']
for ax,season in zip(axes,['2024-2025','2025-2026','combined']):
    g=s[s.geography.eq('all')&s.season.eq(season)].set_index('model').reindex(models)
    arms=[('finalized_available','Final · training encoding','#657789'),('nowcast_off','Nowcast · flag off','#cf8966'),('nowcast_available','Nowcast · training encoding','#07877b')]
    for j,(field,label,color) in enumerate(arms):
        ax.bar(np.arange(3)+(j-1)*.25,g[field],width=.23,color=color,label=label)
    ax.axhline(1,color='#555',ls='--',lw=.8);ax.set_title(season);ax.set_xticks(range(3),models);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].set_ylabel('WIS / matched Hub WIS (lower is better)');axes[-1].legend(fontsize=8)
fig.suptitle('B2: isolate the finality-channel shift from corrected numerical inputs\nMeans across three matched seeds; fixed checkpoints',fontsize=12)
fig.tight_layout();fig.savefig(out/'encoding-comparison.png',dpi=180);plt.close(fig)
rows=['| Model | Season | Final | Final, flag off | Vintage, training encoding | Nowcast, training encoding | Nowcast loss vs final |',
      '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for row in s[s.geography.eq('all')].itertuples():
    rows.append(f'| {row.model} | {row.season} | {row.finalized_available:.4f} | {row.finalized_off:.4f} | {row.vintage_available:.4f} | {row.nowcast_available:.4f} | {row.nowcast_available_loss_pct:+.1f}% |')
p=out/'index.md';text=p.read_text().split('\n<!-- results:start -->')[0]
p.write_text(text+'\n<!-- results:start -->\n## Results\n\n'+'\n'.join(rows)+'\n\n![Encoding comparison](encoding-comparison.png)\n\n[All six conditions](encoding-comparison.csv) · [Paired seed scores](paired-scores.csv). Loss percentages are means of paired seed ratios.\n')
print(s[s.geography.eq('all')].to_string(index=False),flush=True)

# Score the new controls on the same causal continuity cohorts as the initial replay.
from tapestry.dataset.build import load
from tapestry.data.geography import STATE_FIPS
from tapestry.evaluation.hubs import CHANNEL,KEY
from tapestry.evaluation.totals import forecast_cells,cells_totals,season_scores,season_composites
from tapestry.experiment.finalization import reporting_support
settings=json.loads((roots[1]/'experiment.json').read_text());panel=load(settings['dataset'])
old=pd.read_csv('docs/experiments/b2-input-replay-20261001/stable-season-scores.csv')
old['condition']=np.where(old.arm.eq('finalized'),'finalized_available',old.arm+'_off')
stable=[old];support=None;first_keys=None
for run in [r for r in all_runs if 'b2-flag-controls-' in str(r['path'])]:
    cells=forecast_cells(run['path'],settings['frozen']);cell_keys=cells[['target','season',*KEY]].reset_index(drop=True)
    if support is None:
        first_keys=cell_keys
        support=pd.DataFrame(dict(signal=cells.target.map({name:str(panel['target_names'][i]) for name,i in CHANNEL.items()}),
            location=cells.location.map(STATE_FIPS|{'US':'US'}),
            issuance=(pd.to_datetime(cells.reference_date)-pd.Timedelta(days=3)).dt.strftime('%Y-%m-%d'),
            boundary=(pd.to_datetime(cells.reference_date)-pd.Timedelta(days=7)).dt.strftime('%Y-%m-%d')))
        support=reporting_support(panel,support)
    elif not cell_keys.equals(first_keys):raise ValueError('Controls have different frozen support')
    scenario=Scenario.from_string(run['config_id'])
    model=labels[replace(scenario,replay_from='',replay_inputs='none',replay_flags='native').run_id]
    for stratum,keep in [('complete_history_12',support.complete_history_12),('complete_and_uninterrupted',support.complete_history_12&support.uninterrupted_8)]:
        t=cells_totals(cells[keep.to_numpy()]).assign(config_id=run['config_id'],seed=run['seed'])
        scores=season_composites(season_scores(t))
        scores['model'],scores['condition'],scores['stratum']=model,scenario.replay_inputs+'_'+scenario.replay_flags,stratum
        stable.append(scores)
    print(f'Continuity controls: {model}, {scenario.replay_inputs}, seed {run["seed"]}',flush=True)
stable=pd.concat(stable,ignore_index=True)
stable.to_csv(out/'stable-seed-scores.csv',index=False)
sw=stable.pivot(index=keys+['stratum'],columns='condition',values='combined').reset_index()
if sw[conditions].isna().any().any():raise ValueError('Incomplete continuity pairs')
for name in conditions[1:]:sw[name+'_loss_pct']=100*(sw[name]/sw.finalized_available-1)
sw['corrected_nowcast_vs_vintage_pct']=100*(sw.nowcast_available/sw.vintage_available-1)
ss=sw.groupby(['model','geography','season','stratum'])[cols].mean().reset_index()
ss.to_csv(out/'stable-encoding-comparison.csv',index=False)
with p.open('a') as f:f.write('\n[Complete-history and uninterrupted-reporting comparisons](stable-encoding-comparison.csv) · [Seed-level continuity scores](stable-seed-scores.csv). These cohorts concern the forecast target/location; other inputs may have gaps.\n')
print(ss[ss.geography.eq('all')].to_string(index=False))
