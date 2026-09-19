"""Rescore saved 2025–26 B1 predictions on forward tasks; rank and plot, no fitting."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from tapestry.data.geography import STATE_FIPS
from tapestry.model_data.wednesday import WednesdayDataset
from tapestry.models.manager import read_jobs, seed_state
from tapestry.evaluation.forward import TARGETS, task_frame, markdown, geographic_mean
from tapestry.evaluation.hubs import export, KEY, QCOLS, slug
from tapestry.evaluation.scoring import match_forecasts
from tapestry.evaluation.totals import (case_cells, cells_totals, frozen_cases, season_scores,
    run_scores, configuration_ranking, TARGET_WEIGHTS)

ROOT = Path('data/experiments')
OUT = Path('docs/results/Forward-2025')
FROZEN = Path('data/evaluation/forward_2025')
SEASON = '2025-2026'
SEEDS = (42,43,44)
FORWARD = {'direct':'Forward Direct B', 'joint':'Forward Joint gated', 'separate':'Forward Separate'}
# Fixed named comparators: closest recipe, previous control, analogous gate,
# and the old jointly trained two-stage formulation. No ranking-based search.
LEGACY = [
    ('B1-screen-256','target_mlp__direct_finalflag__mask0','B1 B no-mask'),
    ('B1-revisions-20260917','target_B_gap_only','B1 B gap-only'),
    ('B1-revisions-20260917','target_gated_nw0.2_rev0','B1 gated 0.20'),
    ('B1-revisions-20260917','target_two_nw0.2_rev0','B1 joint two-stage'),
]
ORDER = [FORWARD[c] for c in ('separate','direct','joint')] + [x[2] for x in LEGACY]
SHORT = {t:t.replace('wk inc ','').replace('prop ed visits','ED') for t in TARGETS.values()}


def completed(exp, name, seed, output):
    root = ROOT/exp
    job = next(j for j in read_jobs(root) if j['name'] == name)
    attempt, _, done = seed_state(root, job['scenario'], seed)
    if not done:
        raise ValueError(f'Incomplete comparator: {exp}/{name}/{seed}')
    return attempt/output


def heatmap(table, title, filename, center=1, vmin=.65, vmax=1.8, fmt='.2f', figsize=None):
    fig,ax=plt.subplots(figsize=figsize or (11,max(3,len(table)*.55+1.8)))
    im=ax.imshow(table.to_numpy(float), cmap='RdYlGn_r', norm=TwoSlopeNorm(vcenter=center,vmin=vmin,vmax=vmax),aspect='auto')
    ax.set_xticks(range(len(table.columns)), table.columns,rotation=25,ha='right')
    ax.set_yticks(range(len(table)),table.index)
    for i in range(len(table)):
        for j in range(len(table.columns)):
            v=table.iloc[i,j]
            ax.text(j,i,format(v,fmt) if np.isfinite(v) else 'NA',ha='center',va='center',fontsize=9)
    ax.set_title(title)
    fig.colorbar(im,ax=ax,label='Values annotated exactly; color range clipped')
    fig.tight_layout();fig.savefig(OUT/filename,dpi=170);plt.close(fig)


def fan_plots(frames, ds, indices):
    codes={'US':'US','NC':'37','CA':'06'}
    # Fixed illustrative locations, seed and regularly spaced issuances; no best-seed/date selection.
    references=ds.arrays['target_dates'][indices,2][::4]
    colors=['#087e8b','#3875b7','#a54670','#b47c22','#66713a','#725ca0','#b65135']
    for c,target in TARGETS.items():
        truth=task_frame(ds,indices,c)
        truth=truth[truth.valid].drop_duplicates(['target_end_date','location'])
        fig,axes=plt.subplots(len(ORDER),3,figsize=(17,2.2*len(ORDER)),sharex=True,sharey='col',squeeze=False)
        multiplier=100 if c>=3 else 1
        for row,label in enumerate(ORDER):
            frame=frames[(label,42,target)]
            for col,(postal,location) in enumerate(codes.items()):
                ax=axes[row,col]
                t=truth[truth.location==location].sort_values('target_end_date')
                ax.plot(pd.to_datetime(t.target_end_date),t.observed*multiplier,color='black',lw=1.2,label='Pinned reference')
                for reference in references:
                    p=frame[(frame.reference_date==reference)&(frame.location==location)].sort_values('horizon')
                    if p.empty: continue
                    dates=pd.to_datetime(p.target_end_date)
                    ax.fill_between(dates,p['q0.025']*multiplier,p['q0.975']*multiplier,color=colors[row],alpha=.16)
                    ax.fill_between(dates,p['q0.25']*multiplier,p['q0.75']*multiplier,color=colors[row],alpha=.32)
                    ax.plot(dates,p['q0.5']*multiplier,color=colors[row],lw=1)
                ax.set_ylim(bottom=0)
                ax.grid(alpha=.16)
                if row==0: ax.set_title(postal)
                if col==0: ax.set_ylabel(label+'\n'+('ED visits (%)' if c>=3 else 'Admissions'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
        fig.suptitle(f'{SHORT[target]} · 2025–26 · seed 42 · median and 50%/95% fans every fourth issuance\nBlack: September 16 reference. B1 rows used retrospective finalized inputs.',fontsize=13)
        fig.tight_layout(rect=(0,0,1,.965));fig.savefig(OUT/f'fans-{slug(target)}.png',dpi=150);plt.close(fig)


def overview_fans(frames, ds, indices):
    target=TARGETS[0]
    truth=task_frame(ds,indices,0)
    truth=truth[truth.valid & truth.location.eq('US')].drop_duplicates('target_end_date').sort_values('target_end_date')
    fig,axes=plt.subplots(2,2,figsize=(13,8),sharex=True,sharey=True)
    labels=['Forward Separate','Forward Direct B','Forward Joint gated','B1 B no-mask']
    for ax,label,color in zip(axes.flat,labels,['#087e8b','#3875b7','#a54670','#b47c22']):
        ax.plot(pd.to_datetime(truth.target_end_date),truth.observed,color='black',lw=1.4,label='Pinned reference')
        frame=frames[label,42,target]
        for reference in ds.arrays['target_dates'][indices,2][::4]:
            g=frame[(frame.reference_date==reference)&frame.location.eq('US')].sort_values('horizon')
            dates=pd.to_datetime(g.target_end_date)
            ax.fill_between(dates,g['q0.025'],g['q0.975'],color=color,alpha=.16)
            ax.fill_between(dates,g['q0.25'],g['q0.75'],color=color,alpha=.35)
            ax.plot(dates,g['q0.5'],color=color,lw=1)
        ax.set_title(label+(' (retrospective inputs)' if label.startswith('B1') else ''))
        ax.set_ylabel('Weekly US influenza admissions');ax.set_ylim(bottom=0);ax.grid(alpha=.15)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2));ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    fig.suptitle('2025–26 · seed 42 · median and 50%/95% forecast fans every fourth issuance')
    fig.tight_layout();fig.savefig(OUT/'fans-us-flu-overview.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-fans',action='store_true',help='Reuse previously generated fixed-seed fans')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    ds=WednesdayDataset.load('data/processed/forward_2025.npz')
    frames, all_cells, provenance = {}, [], []
    cases=frozen_cases(FROZEN)
    runs=[]
    for candidate,label in FORWARD.items():
        for seed in SEEDS:
            path=completed('Forward-2025',candidate,seed,'forward')
            with np.load(path/'forecasts.npz') as data:
                q,indices=data['quantiles'],data['indices']
            per_target={}
            for c,target in TARGETS.items():
                frame=task_frame(ds,indices,c)
                frame[QCOLS]=q[:,:,2:,c].reshape(len(QCOLS),-1).T
                per_target[target]=frame[frame.valid].drop(columns='valid')
            runs.append((label,seed,path,per_target,'forward'))
    for exp,name,label in LEGACY:
        for seed in SEEDS:
            path=completed(exp,name,seed,'b1')
            exported=export(path)
            runs.append((label,seed,path,{target:exported[(SEASON,target)] for target in TARGETS.values()},'retrospective B1'))
    for label,seed,path,per_target,protocol in runs:
        print(f'Scoring saved predictions: {label}, seed {seed}',flush=True)
        manifest=json.loads((path/'manifest.json').read_text())
        configuration=manifest.get('configuration',{})
        fold=manifest.get('folds',{}).get(SEASON,[])
        provenance.append(dict(label=label,seed=seed,path=str(path),protocol=protocol,
            manifest_sha256=hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest(),
            epochs_cap=configuration.get('epochs',manifest.get('epochs')), mask_rate=configuration.get('mask_rate',0),
            refit_epochs=[r['epochs'] for r in fold if r.get('phase')=='refit'],
            eval_members=manifest['eval_members'], configuration=configuration))
        for case in cases:
            target=case['target']; frame=per_target[target]
            units=pd.read_parquet(FROZEN/case['directory']/'units.parquet')
            ensemble=pd.read_parquet(FROZEN/case['directory']/'quantiles.parquet')
            matched=match_forecasts(frame,units,target)
            if seed==42: frames[label,seed,target]=frame.copy()
            cells=case_cells(matched,match_forecasts(ensemble,units,target),case)
            cells=cells.assign(config_id=label,seed=seed)
            all_cells.append(cells)
    cells=pd.concat(all_cells,ignore_index=True)
    totals=pd.concat([cells_totals(g).assign(config_id=label,seed=seed) for (label,seed),g in cells.groupby(['config_id','seed'])],ignore_index=True)
    totals.to_csv(OUT/'comparison-totals.csv',index=False)
    seasons=season_scores(totals); scores=run_scores(seasons)
    ranking=configuration_ranking(scores)
    # Coverage has the same target/geography weights as the WIS composite.
    for level in (50,80,90,95):
        coverage=run_scores(seasons.assign(wis_ratio=seasons[f'model_coverage_{level}']))
        cov=coverage[coverage.geography=='all'].groupby('config_id').combined.mean()
        ranking[f'coverage_{level}']=ranking.config_id.map(cov)
    ranking.to_csv(OUT/'comparison-ranking.csv',index=False)
    scores.to_csv(OUT/'comparison-seeds.csv',index=False)
    seasons.to_csv(OUT/'comparison-targets.csv',index=False)
    # Differences paired by seed; do not interpret as independent season replication.
    wide=scores[scores.geography=='all'].pivot(index='seed',columns='config_id',values='combined')
    paired=pd.DataFrame({'separate_minus_direct':wide['Forward Separate']-wide['Forward Direct B'],
        'separate_minus_joint':wide['Forward Separate']-wide['Forward Joint gated'],
        'joint_minus_direct':wide['Forward Joint gated']-wide['Forward Direct B']})
    paired.to_csv(OUT/'paired-forward-seeds.csv')
    target_table=seasons[seasons.geography=='all'].groupby(['config_id','target']).wis_ratio.mean().unstack().reindex(ORDER)
    target_table=target_table.reindex(columns=list(TARGETS.values())).rename(columns=SHORT)
    target_table.insert(0,'Combined',ranking.set_index('config_id').combined_mean)
    heatmap(target_table,'2025–26 matched tasks · relative WIS (lower is better; ensemble = 1)\nForward rows: cutoff-pinned; B1 rows: retrospective information', 'heatmap-comparison.png')
    # Geography contrasts preserve the shared location-relative denominators.
    loc=totals.groupby(['config_id','seed','target','location'])[['model_wis','ensemble_wis']].sum()
    loc['ratio']=loc.model_wis/loc.ensemble_wis
    loc=loc.reset_index()
    geography=loc.groupby(['config_id','target','location']).ratio.mean().unstack('config_id')
    change=100*(geography['Forward Separate']/geography['Forward Direct B']-1)
    change=change.unstack('target').reindex(columns=list(TARGETS.values())).rename(columns=SHORT)
    change.index=[STATE_FIPS.get(str(x),str(x)) for x in change.index]
    change=change.sort_index()
    heatmap(change,'Separate versus Direct B · change in location-relative WIS (%)\nNegative favors separate; mean seed scores, identical tasks', 'heatmap-geography.png',center=0,vmin=-50,vmax=50,fmt='+.0f',figsize=(10,17))
    # Horizon composites use the same scientific aggregation, one horizon at a time.
    horizons=[]
    for h,g in totals.groupby('horizon'):
        hr=run_scores(season_scores(g));hr=hr[hr.geography=='all']
        horizons.append(hr.assign(horizon=h))
    horizons=pd.concat(horizons,ignore_index=True)
    horizons.to_csv(OUT/'comparison-horizons.csv',index=False)
    ht=horizons.groupby(['config_id','horizon']).combined.mean().unstack().reindex(ORDER)
    ht.columns=[f'Horizon {h}' for h in ht.columns]
    heatmap(ht,'Matched target-weighted relative WIS by forecast horizon','heatmap-horizons.png',vmin=.6,vmax=2.5)
    # Recent diagnostics keep the two ages and two tasks separate.
    recent=pd.read_csv(OUT/'recent-summary.csv')
    rm=['scaled_ae','scaled_wis','baseline_scaled_ae','covered_95']
    summary=[]
    for keys,g in recent.groupby(['candidate','seed','kind','age_days']):
        w=g.target.map(TARGET_WEIGHTS)
        summary.append(dict(zip(['candidate','seed','kind','age_days'],keys),
            **{m:(g[m]*w).sum(min_count=1)/w.sum() for m in rm}))
    recent_summary=pd.DataFrame(summary).groupby(['candidate','kind','age_days'])[rm].mean().reset_index()
    recent_summary.to_csv(OUT/'recent-scientific-summary.csv',index=False)
    rt=recent.groupby(['candidate','kind','age_days','target']).scaled_ae.mean().unstack('target').reindex(columns=list(TARGETS.values())).rename(columns=SHORT)
    # A sequential heatmap is more appropriate for nonnegative scaled errors.
    fig,ax=plt.subplots(figsize=(11,6));im=ax.imshow(rt,cmap='YlOrRd',aspect='auto',vmin=0,vmax=.12)
    ax.set_xticks(range(6),rt.columns,rotation=20,ha='right');ax.set_yticks(range(len(rt)),[' / '.join(map(str,k))+'d' for k in rt.index])
    for i in range(len(rt)):
        for j in range(6):ax.text(j,i,f'{rt.iloc[i,j]:.3f}',ha='center',va='center',fontsize=8)
    ax.set_title('Recent estimates · MAE / training Q95 · lower is better\nVisible revisions and missing reconstruction remain separate')
    fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(OUT/'heatmap-recent.png',dpi=170);plt.close(fig)
    if not args.reuse_fans:
        fan_plots(frames,ds,indices)
    overview_fans(frames,ds,indices)
    components=[]
    for metric in ['wis','underprediction','overprediction','dispersion']:
        component=season_scores(totals.assign(model_wis=totals['model_'+metric]))
        components.append(component[component.geography=='all'].groupby(['config_id','target']).wis_ratio.mean().rename(metric))
    pd.concat(components,axis=1).to_csv(OUT/'comparison-wis-components.csv')
    support=cells[cells.config_id=='Forward Direct B'].query('seed == 42').groupby('target').agg(tasks=('horizon','size'),first=('reference_date','min'),last=('reference_date','max'))
    support.to_csv(OUT/'comparison-support.csv')
    # Count supplied finals on precisely the same B1 input cells used for these tasks.
    legacy_path=Path('data/processed/build_b1_wednesday_calendar.npz')
    legacy_hash=hashlib.sha256(legacy_path.read_bytes()).hexdigest()
    for _,_,path,_,protocol in runs:
        if protocol == 'retrospective B1' and json.loads((path/'manifest.json').read_text())['dataset_sha256'] != legacy_hash:
            raise ValueError('Legacy dataset differs from fitted input provenance')
    legacy_ds=WednesdayDataset.load(legacy_path)
    la=legacy_ds.arrays
    issue_map={str(d):i for i,d in enumerate(la['target_dates'][:,2])}
    location_map={loc:i for i,loc in enumerate(legacy_ds.locations)}
    fips_to_postal=dict(STATE_FIPS)|{'US':'US'}
    availability=[]
    for c,target in TARGETS.items():
        units=cells[(cells.config_id=='Forward Direct B')&(cells.seed==42)&(cells.target==target)].drop_duplicates(['reference_date','location'])
        for h,age in enumerate((11,4)):
            flags=[la['X_final'][issue_map[r.reference_date],-2+h,c,location_map[fips_to_postal[r.location]]] for r in units.itertuples()]
            availability.append(dict(target=target,age_days=age,inputs=len(flags),supplied_final_fraction=float(np.mean(flags))))
    pd.DataFrame(availability).to_csv(OUT/'b1-final-input-fractions.csv',index=False)
    meta=dict(seeds=list(SEEDS),fan_seed=42,fan_locations=['US','NC','CA'],fan_stride=4,
        comparator_selection='Fixed named recipes, not selected by current season ranking',
        scoring='Same frozen tasks, September 16 reference truth, ensemble and scientific weights',
        limitations='B1 retrospective inputs and latest reference labels, different selection/budgets/masking/draw counts; no causal cross-protocol comparison',
        runs=provenance,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'analysis-provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(ranking[['config_id','combined_mean','combined_sd','coverage_95']].to_string(index=False))
    print('Paired forward differences:\n'+paired.to_string())
    print('Recent scientific summary:\n'+recent_summary.to_string(index=False))
    print('B1 supplied-final fractions:\n'+pd.DataFrame(availability).to_string(index=False))
    # Interpretation is added after numerical review, without inspecting images.
    table=ranking[['config_id','combined_mean','combined_sd','states_dc_combined_mean','US_combined_mean','coverage_95']].copy()
    table.columns=['Candidate','Relative WIS','Seed SD','States/DC','US','95% coverage']
    (OUT/'analysis-tables.md').write_text(markdown(table)+'\n\n'+markdown(recent_summary)+'\n')


if __name__=='__main__':
    main()
