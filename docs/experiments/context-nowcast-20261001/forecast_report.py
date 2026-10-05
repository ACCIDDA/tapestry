"""Matched C1 forecast scores under the documented source-availability hypothesis."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tapestry.dataset.build import load
from tapestry.data.geography import STATE_FIPS
from tapestry.model.scenario import Scenario
from tapestry.evaluation.hubs import CHANNEL,KEY
from tapestry.evaluation.totals import forecast_cells,cells_totals,season_scores,season_composites
from tapestry.experiment.finalization import reporting_support

p=argparse.ArgumentParser();p.add_argument('-e','--experiment',default='context-replay-v1-20261001')
p.add_argument('--output',type=Path,default=Path(__file__).parent);a=p.parse_args()
root=Path('data/experiments')/a.experiment;out=a.output;out.mkdir(parents=True,exist_ok=True)
rank=max(root.glob('ranking-*'),key=lambda p:p.stat().st_mtime)
runs=json.loads((rank/'manifest.json').read_text())['runs']
settings=json.loads((root/'experiment.json').read_text());panel=load(settings['dataset'])
reference=Scenario.from_string(json.loads((root/'replay-source.json').read_text())['nowcaster_reference']['scenario'])
summary=[];targets=[];horizons=[];coverage=[];distributions=[];first_keys=None
for run in runs:
    s=Scenario.from_string(run['config_id'])
    arm=('Final input' if s.replay_inputs=='finalized' else 'Raw reports' if s.replay_inputs=='vintage' else
         'Seasonal control' if s.replay_nowcaster=='selected' else 'Context residual')
    if s.replay_nowcaster=='context_residual' and s.replay_strength!=reference.finalization_strength:
        arm+=f' ×{s.replay_strength:g}'
    if s.replay_uncertainty:
        arm+=f' + uncertainty {s.replay_uncertainty:g}'
        # History draws use the same seed for every fixed C1 checkpoint; report once.
        if run['seed']==42:
            for f in Path(run['path']).glob('eval_*/trajectory-distributions.csv.gz'):
                d=pd.read_csv(f);d['season']=f.parent.name.removeprefix('eval_')
                d['geography']=np.where(d.location.eq('US'),'US','states')
                d['arm']=arm;distributions.append(d)
    cells=forecast_cells(run['path'],settings['frozen'])
    cell_keys=cells[['target','season',*KEY]].reset_index(drop=True)
    if first_keys is None:
        first_keys=cell_keys
        support=reporting_support(panel,pd.DataFrame(dict(
            signal=cells.target.map({name:str(panel['target_names'][i]) for name,i in CHANNEL.items()}),
            location=cells.location.map(STATE_FIPS|{'US':'US'}),
            issuance=(pd.to_datetime(cells.reference_date)-pd.Timedelta(days=3)).dt.strftime('%Y-%m-%d'),
            boundary=(pd.to_datetime(cells.reference_date)-pd.Timedelta(days=7)).dt.strftime('%Y-%m-%d'))))
    elif not cell_keys.equals(first_keys):raise ValueError('Forecast arms differ in frozen task support')
    strata={'all':np.ones(len(cells),bool),'complete_history_12':support.complete_history_12.to_numpy(),
        'complete_and_uninterrupted':(support.complete_history_12&support.uninterrupted_8).to_numpy()}
    for stratum,keep in strata.items():
        part=cells[keep]
        totals=cells_totals(part).assign(config_id=run['config_id'],seed=run['seed'])
        scores=season_scores(totals)
        targets.append(scores.assign(arm=arm,stratum=stratum))
        summary.append(season_composites(scores).assign(arm=arm,stratum=stratum))
        coverage.append(part.groupby(['season','target']).size().rename('cells').reset_index().assign(arm=arm,stratum=stratum,seed=run['seed']))
        for horizon,g in part.groupby('horizon'):
            totals=cells_totals(g).assign(config_id=run['config_id'],seed=run['seed'])
            horizons.append(season_scores(totals).assign(arm=arm,stratum=stratum,horizon=horizon))
    print(f'Scored C1 {arm}, seed {run["seed"]}',flush=True)
summary=pd.concat(summary,ignore_index=True);summary.to_csv(out/'forecast-seed-scores.csv',index=False)
pd.concat(targets,ignore_index=True).to_csv(out/'forecast-target-scores.csv.gz',index=False)
pd.concat(horizons,ignore_index=True).to_csv(out/'forecast-horizon-target-scores.csv.gz',index=False)
pd.concat(coverage,ignore_index=True).to_csv(out/'forecast-coverage.csv',index=False)
keys=['stratum','season','geography','seed']
w=summary.pivot(index=keys,columns='arm',values='combined').reset_index()
if w.isna().any().any():raise ValueError('Incomplete paired forecast arms')
w['change_vs_seasonal_pct']=100*(w['Context residual']/w['Seasonal control']-1)
w['change_vs_raw_pct']=100*(w['Context residual']/w['Raw reports']-1)
w['loss_vs_final_pct']=100*(w['Context residual']/w['Final input']-1)
for name in w.columns:
    if ' + uncertainty ' in name:
        w[name+' vs seasonal pct']=100*(w[name]/w['Seasonal control']-1)
w.to_csv(out/'forecast-paired.csv',index=False)
s=w.groupby(keys[:-1],as_index=False).mean(numeric_only=True).drop(columns='seed')
s.to_csv(out/'forecast-summary.csv',index=False)
print(s[s.geography.eq('all')].to_string(index=False))
arms=['Final input','Raw reports','Seasonal control','Context residual']+[n for n in w.columns if (n.startswith('Context residual ×') or ' + uncertainty ' in n) and ' pct' not in n]
fig,axes=plt.subplots(1,3,figsize=(max(14,len(arms)*2.4),5.5),sharey=True)
for ax,stratum,title in zip(axes,['all','complete_history_12','complete_and_uninterrupted'],
    ['All matched forecast tasks','Complete 12-week history','Also eight uninterrupted reports']):
    g=s[s.season.eq('2025-2026')&s.geography.eq('all')&s.stratum.eq(stratum)].iloc[0]
    vals=[g[v] for v in arms]
    ax.bar(range(len(arms)),vals,color=['#9ba5af','#c99970','#5c91ae','#16877f','#466183','#734886'])
    ax.set_xticks(range(len(arms)),[n.replace('Context residual + uncertainty ','Residual + U×').replace('Seasonal control + uncertainty ','Seasonal + U×') for n in arms],rotation=35,ha='right');ax.set_title(title,fontsize=10)
    ax.axhline(1,color='#444',ls='--',lw=1);ax.grid(axis='y',alpha=.15)
    for i,v in enumerate(vals):ax.text(i,v+.008,f'{v:.4f}',ha='center',fontsize=9)
    ax.set_ylim(0,max(1.1,max(vals)*1.15))
axes[0].set_ylabel('WIS / matched Hub WIS (lower is better)')
fig.suptitle('C1 forecast replay · 2025–26 · three paired seeds\nDocumented availability schedule; identical input encoding')
fig.tight_layout();fig.savefig(out/'forecast-comparison.png',dpi=180);plt.close(fig)
if distributions:
    d=pd.concat(distributions,ignore_index=True);rows=[]
    metrics=[c for c in d.columns if c.endswith(('_crps','_point_error','_coverage80'))]
    for stratum,keep in [('complete_history_12',d.complete_history_12),
        ('complete_and_uninterrupted',d.complete_history_12&d.uninterrupted_8)]:
        g=d[keep].groupby(['arm','season','geography','signal','location'])[metrics].mean()
        rows.append(g.groupby(['arm','season','geography','signal']).mean().reset_index().assign(stratum=stratum))
    by_signal=pd.concat(rows);by_signal.to_csv(out/'distribution-signal-scores.csv',index=False)
    scores=by_signal.groupby(['arm','season','geography','stratum'])[metrics].mean().reset_index()
    scores['trajectory_crps_skill_pct']=100*(1-scores.trajectory_crps/scores.trajectory_point_error)
    scores.to_csv(out/'distribution-scores.csv',index=False)
    print(scores.to_string(index=False))
