"""Paired input-arm comparisons from the shared manager's unchanged WIS scores."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tapestry.model.scenario import Scenario

root=Path('data/experiments/b2-input-replay-20261001')
out=Path(__file__).parent
rankings=sorted(root.glob('ranking-*'),key=lambda p:p.stat().st_mtime)
if not rankings:raise SystemExit('Run manager rank first')
r=rankings[-1]
labels=json.loads((root/'replay-source.json').read_text())['labels']
def identity(config):
    s=Scenario.from_string(config)
    return labels[replace(s,replay_from='',replay_inputs='none').run_id],s.replay_inputs

def tagged(path):
    d=pd.read_csv(path)
    tags=d.config_id.map(identity)
    d['model']=[t[0] for t in tags];d['arm']=[t[1] for t in tags]
    return d
s=tagged(r/'season_composite_scores.csv')
a=tagged(r/'run_scores.csv').assign(season='combined')
d=pd.concat([s,a],ignore_index=True)
wide=d.pivot(index=['model','seed','geography','season'],columns='arm',values='combined').reset_index()
if len(wide)!=81 or wide[['finalized','nowcast','vintage']].isna().any().any():
    raise ValueError('Expected all three models/seeds/arms/geographies and both seasons plus combined')
for arm in ['nowcast','vintage']:
    wide[f'{arm}_loss_pct']=100*(wide[arm]/wide.finalized-1)
wide['nowcast_vs_vintage_pct']=100*(wide.nowcast/wide.vintage-1)
wide.to_csv(out/'paired-seed-scores.csv',index=False)
cols=['finalized','nowcast','vintage','nowcast_loss_pct','vintage_loss_pct','nowcast_vs_vintage_pct']
summary=wide.groupby(['model','geography','season'])[cols].agg(['mean','std'])
summary.columns=['_'.join(c) for c in summary.columns]
summary=summary.reset_index();summary.to_csv(out/'input-comparison.csv',index=False)
t=tagged(r/'season_scores.csv');t.to_csv(out/'target-season-scores.csv',index=False)
fig,axes=plt.subplots(1,3,figsize=(13,4.8),sharey=True)
colors=['#6c7885','#07877b','#bf7546'];models=['C1','C2','C3']
for ax,season in zip(axes,['2024-2025','2025-2026','combined']):
    g=summary[summary.geography.eq('all')&summary.season.eq(season)].set_index('model').reindex(models)
    for j,(arm,label,color) in enumerate(zip(['finalized','nowcast','vintage'],['Finalized','Nowcast','Vintage'],colors)):
        x=np.arange(3)+(j-1)*.25
        ax.bar(x,g[f'{arm}_mean'],width=.23,color=color,label=label,yerr=g[f'{arm}_std'],capsize=2)
    ax.axhline(1,color='#444',lw=.8,ls='--');ax.set_xticks(range(3),models);ax.set_title(season);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
axes[0].set_ylabel('WIS / matched Hub WIS (lower is better)');axes[-1].legend(fontsize=9)
fig.suptitle('Fixed B2 checkpoints: finalized, nowcast-corrected, and vintage histories\nThree-seed means; bars show seed SD, not confidence intervals',fontsize=12)
fig.tight_layout();fig.savefig(out/'input-comparison.png',dpi=180);plt.close(fig)
rows=['| Model | Season | Final | Nowcast | Vintage | Nowcast loss vs final | Vintage loss vs final |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for row in summary[summary.geography.eq('all')].itertuples():
    rows.append(f'| {row.model} | {row.season} | {row.finalized_mean:.4f} | {row.nowcast_mean:.4f} | {row.vintage_mean:.4f} | {row.nowcast_loss_pct_mean:+.1f}% | {row.vintage_loss_pct_mean:+.1f}% |')
results='\n## Results\n\n'+ '\n'.join(rows)+'\n\nLoss percentages are averaged paired seed ratios; scores are seed means.\n\n![Three-arm CV comparison](input-comparison.png)\n\n[Paired seed scores](paired-seed-scores.csv) · [Geography summaries](input-comparison.csv) · [Target and season scores](target-season-scores.csv).\n'
p=out/'index.md';content=p.read_text().split('\n<!-- results:start -->')[0]
p.write_text(content+'\n<!-- results:start -->\n'+results+'\n<!-- results:end -->\n')
print(summary[summary.geography.eq('all')].to_string(index=False))

# Reuse the standard cell scorer and weighting for matched reporting-continuity subsets.
from tapestry.dataset.build import load
from tapestry.data.geography import STATE_FIPS
from tapestry.evaluation.hubs import CHANNEL, KEY
from tapestry.evaluation.totals import forecast_cells,cells_totals,season_scores,season_composites
from tapestry.experiment.finalization import reporting_support
panel=load(json.loads((root/'experiment.json').read_text())['dataset'])
runs=json.loads((r/'manifest.json').read_text())['runs']
stratified=[];first_keys=None;support=None
for run in runs:
    cells=forecast_cells(run['path'],json.loads((root/'experiment.json').read_text())['frozen'])
    keys=cells[['target','season',*KEY]].reset_index(drop=True)
    if first_keys is None:
        first_keys=keys
        support=pd.DataFrame(dict(signal=cells.target.map({name:str(panel['target_names'][i]) for name,i in CHANNEL.items()}),
            location=cells.location.map(STATE_FIPS|{'US':'US'}),
            issuance=(pd.to_datetime(cells.reference_date)-pd.Timedelta(days=3)).dt.strftime('%Y-%m-%d'),
            boundary=(pd.to_datetime(cells.reference_date)-pd.Timedelta(days=7)).dt.strftime('%Y-%m-%d')))
        if support.location.isna().any():raise ValueError('Unmapped Hub location')
        support=reporting_support(panel,support)
    elif not keys.equals(first_keys):raise ValueError('Forecast support differs across replay arms')
    model,arm=identity(run['config_id'])
    for stratum,keep in [('complete_history_12',support.complete_history_12),
                         ('complete_and_uninterrupted',support.complete_history_12&support.uninterrupted_8)]:
        totals=cells_totals(cells[keep.to_numpy()]).assign(config_id=run['config_id'],seed=run['seed'])
        scores=season_composites(season_scores(totals))
        scores['model'],scores['arm'],scores['stratum']=model,arm,stratum
        stratified.append(scores)
    print(f'Continuity scores: {model}, {arm}, seed {run["seed"]}',flush=True)
stable=pd.concat(stratified,ignore_index=True)
stable.to_csv(out/'stable-season-scores.csv',index=False)
sw=stable.pivot(index=['model','seed','geography','season','stratum'],columns='arm',values='combined').reset_index()
for arm in ['nowcast','vintage']:sw[f'{arm}_loss_pct']=100*(sw[arm]/sw.finalized-1)
sw['nowcast_vs_vintage_pct']=100*(sw.nowcast/sw.vintage-1)
ss=sw.groupby(['model','geography','season','stratum'])[cols].mean().reset_index()
ss.to_csv(out/'stable-input-comparison.csv',index=False)
coverage=first_keys.copy();coverage['complete_history_12']=support.complete_history_12.to_numpy();coverage['complete_and_uninterrupted']=(support.complete_history_12&support.uninterrupted_8).to_numpy()
coverage.groupby(['season','target']).agg(tasks=('location','size'),complete_history_12=('complete_history_12','sum'),complete_and_uninterrupted=('complete_and_uninterrupted','sum')).to_csv(out/'stable-coverage.csv')
rows=['| Model | Season | Final | Nowcast | Vintage | Nowcast loss vs final | Vintage loss vs final |','| --- | --- | ---: | ---: | ---: | ---: | ---: |']
for row in ss[ss.geography.eq('all')&ss.stratum.eq('complete_history_12')].itertuples():
    rows.append(f'| {row.model} | {row.season} | {row.finalized:.4f} | {row.nowcast:.4f} | {row.vintage:.4f} | {row.nowcast_loss_pct:+.1f}% | {row.vintage_loss_pct:+.1f}% |')
with p.open('a') as stream:
    stream.write('\n## Complete reporting histories\n\n'+ '\n'.join(rows)+'\n\nThese subsets require the forecast target/location to have all 12 context weeks visible at issuance. The stricter subset additionally requires eight uninterrupted newest-week reports. Other target and covariate histories may still contain gaps. The same support and standard scorer weights apply in every arm; coverage can differ between seasons.\n\n[Continuity comparisons](stable-input-comparison.csv) · [Coverage](stable-coverage.csv) · [Seed-level continuity scores](stable-season-scores.csv).\n')
print(ss[ss.geography.eq('all')].to_string(index=False))
