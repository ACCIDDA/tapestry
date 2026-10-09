"""Analyze completed weekend forecasts; no training or forecast scoring."""
from pathlib import Path
from dataclasses import replace
import re
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from chromantis.model.scenario import Scenario

root=Path(__file__).parent
out=root/'analysis'
design=pd.read_csv(root/'design.csv').rename(columns={'scenario':'config_id'})
rows=[]
for r in design.itertuples():
    s=Scenario.from_string(r.config_id)
    actual=replace(s,joint_weight=1.) if s.weekend_family=='joint' else s
    label=re.sub(r'; reconstruction weight [0-9.]+','; pooled eight-horizon loss',r.treatment) if s.weekend_family=='joint' else r.treatment
    rows.append(dict(config_id=r.config_id,effective_id=actual.run_id,priority=int(s.weekend_family=='joint' and s.joint_weight!=1),
        actual_treatment=label,model=f'{s.fit_partition} {s.encoder}; {s.spatial} spatial; covariates: {s.covariate_set or "none"}',
        reconstruction_labels='Finalized t-3..t0' if s.weekend_family=='joint' else 'None',
        evaluation_inputs='Regression-corrected Wednesday reports' if s.weekend_family=='two_stage' else 'Raw Wednesday reports'))
meta=design.merge(pd.DataFrame(rows),on='config_id',validate='one_to_one')
raw=pd.read_csv(out/'ranking/season_composite_scores.csv').merge(meta,on='config_id',validate='many_to_one')
# We know the old weight field was unused. Prefer its default spelling; if it
# did not finish, use the equivalent nondefault spelling. Never choose by score.
raw=raw.sort_values('priority').drop_duplicates(['effective_id','seed','season','geography'])
base=raw[raw.actual_treatment.eq('Unchanged finalized training inputs')][['architecture','seed','season','geography','combined']].rename(columns={'combined':'baseline'})
raw=raw.merge(base,on=['architecture','seed','season','geography'],how='left',validate='many_to_one')
raw['paired_change_pct']=100*(raw.combined/raw.baseline-1)
raw['train_seasons']=raw.season.map({'2024-2025':'2022-23, 2023-24, 2025-26','2025-2026':'2022-23, 2023-24, 2024-25'})
raw['error_season']=raw.season.map({'2024-2025':'2025-26','2025-2026':'2024-25'})
raw['forecast_labels']='Finalized t+1..t+4'
raw['availability']='B2 source schedule, final proxies for missing archives; no random masking'
raw.to_csv(out/'all-available-seed-scores.csv',index=False)
keys=['effective_id','architecture','source_b2_label','family','actual_treatment','model','reconstruction_labels','evaluation_inputs','season','train_seasons','error_season','geography']
summary=raw.groupby(keys).agg(seeds=('seed','nunique'),wis=('combined','mean'),sd=('combined','std'),paired_change_pct=('paired_change_pct','mean'),baseline=('baseline','mean')).reset_index()
summary.to_csv(out/'all-available-configurations.csv',index=False)
full=summary[summary.seeds.eq(3)].copy()
full.to_csv(out/'three-seed-season-results.csv',index=False)
allgeo=full[full.geography.eq('all')]
combined=allgeo.groupby(['effective_id','architecture','source_b2_label','family','actual_treatment','model']).agg(combined=('wis','mean'),baseline=('baseline','mean'),paired_change_pct=('paired_change_pct','mean'),seasons=('season','nunique')).reset_index()
combined=combined[combined.seasons.eq(2)].sort_values('combined')
combined.to_csv(out/'three-seed-ranking.csv',index=False)
print('COMPLETE_EFFECTIVE_CONFIGURATIONS',len(combined))
print('BEST_BY_FAMILY')
print(combined.groupby('family',sort=False).head(1).to_string(index=False))
print('BEST_FINALIZED_TRAINING')
print(combined[combined.actual_treatment.eq('Unchanged finalized training inputs')].head(1).to_string(index=False))
print('TOP10')
print(combined.head(10).to_string(index=False))

# Fixed architecture means compare every treatment on the same six backbones.
treatment=allgeo.groupby(['actual_treatment','family','season']).agg(mean_paired_change_pct=('paired_change_pct','mean'),architectures=('architecture','nunique'),architectures_better=('paired_change_pct',lambda x:int((x<0).sum())),mean_wis=('wis','mean'),mean_baseline=('baseline','mean')).reset_index()
treatment.to_csv(out/'treatment-across-six-architectures.csv',index=False)
print('TREATMENT_PAIRED_CHANGES')
print(treatment.pivot(index='actual_treatment',columns='season',values='mean_paired_change_pct').round(2).to_string())

order=['Unchanged finalized training inputs']+list(treatment[treatment.season.eq('2025-2026')].sort_values('mean_paired_change_pct').actual_treatment)
order=list(dict.fromkeys(order))
fig,axes=plt.subplots(1,2,figsize=(17,12),sharey=True)
for ax,season,training,donor in zip(axes,['2024-2025','2025-2026'],['2022–23, 2023–24, 2025–26','2022–23, 2023–24, 2024–25'],['2025–26','2024–25']):
    q=allgeo[allgeo.season.eq(season)]
    for i,label in enumerate(order):
        v=q[q.actual_treatment.eq(label)].paired_change_pct
        ax.scatter(v,np.full(len(v),i),alpha=.55,s=20,color='#668caa')
        ax.scatter(v.mean(),i,color='#202c35',marker='D',s=25)
    ax.axvline(0,color='black',lw=.8)
    ax.set_title(f'Evaluate {season}\nTrain {training}; errors from {donor}',fontsize=10)
    ax.set_xlabel('Paired WIS change vs same architecture trained on finalized inputs (%)\nNegative is better; each dot = three-seed mean for one architecture',fontsize=9)
    ax.grid(axis='x',alpha=.2)
axes[0].set_yticks(range(len(order)),order,fontsize=8);axes[0].invert_yaxis()
fig.suptitle('Completed B2 models: finalized future labels; B2 availability schedule with final proxies\nDirect/joint: raw evaluation reports. Two-stage: regression-corrected evaluation reports. Joint loss used one pooled normalization.',fontsize=11)
fig.tight_layout();fig.savefig(out/'treatment-paired-changes.png',dpi=160);plt.close(fig)

chosen=combined.groupby('family',sort=False).head(1).effective_id.tolist()
clean=combined[combined.actual_treatment.eq('Unchanged finalized training inputs')].head(1).effective_id.tolist()
chosen=list(dict.fromkeys(clean+chosen))
fig,axes=plt.subplots(1,2,figsize=(14,7),sharey=True)
labels=[]
for eid in chosen:
    r=combined[combined.effective_id.eq(eid)].iloc[0]
    labels.append(f'A{r.architecture} ({r.source_b2_label}): {r.actual_treatment}\n{r.model}')
for ax,season,training,donor in zip(axes,['2024-2025','2025-2026'],['2022–23, 2023–24, 2025–26','2022–23, 2023–24, 2024–25'],['2025–26','2024–25']):
    for i,eid in enumerate(chosen):
        v=raw[(raw.effective_id==eid)&(raw.season==season)&(raw.geography=='all')]
        ax.scatter(v.combined,np.full(len(v),i),s=40,color='#3a8a89')
        ax.scatter(v.combined.mean(),i,marker='D',color='#26393c',s=40)
    ax.axvline(1,color='gray',ls='--')
    ax.set_title(f'Evaluate {season}\nTrain {training}; errors from {donor}',fontsize=10)
    ax.set_xlabel('WIS / matched Hub WIS; lower is better')
    ax.grid(axis='x',alpha=.2)
axes[0].set_yticks(range(len(chosen)),labels,fontsize=8);axes[0].invert_yaxis()
fig.suptitle('Best completed three-seed configuration within each family (selected on two-season mean)\nFinalized labels; B2 availability and final proxies; separate nowcaster changes evaluation reports. Dots are seeds.',fontsize=10)
fig.tight_layout();fig.savefig(out/'best-completed-models.png',dpi=170);plt.close(fig)

# Target diagnostics for the best joint model and its matched clean baseline.
best=combined[combined.family.eq('joint')].iloc[0]
selected=raw[(raw.architecture==best.architecture)&((raw.effective_id==best.effective_id)|raw.actual_treatment.eq('Unchanged finalized training inputs'))]
keep=selected[['config_id','seed','effective_id','actual_treatment']].drop_duplicates()
target=pd.read_csv(out/'ranking/season_scores.csv').merge(keep,on=['config_id','seed'],validate='many_to_one')
target=target.groupby(['effective_id','actual_treatment','season','geography','target']).agg(wis=('wis_ratio','mean'),coverage90=('model_coverage_90','mean')).reset_index()
target.to_csv(out/'best-joint-target-diagnostics.csv',index=False)
print('BEST_FAMILY_SEASON_SCORES')
print(allgeo[allgeo.effective_id.isin(chosen)][['architecture','source_b2_label','family','actual_treatment','season','wis','baseline','paired_change_pct']].to_string(index=False))

# Forward-season examples selected from completed three-seed configurations.
examples=[(43,'synchronous_log strength 0.5','B2 pathogen MLP; all covariates; no spatial sharing\nTrain: synchronized half-strength revision windows'),
          (42,'Half clean episodes; local log errors strength 1.0','B2 pathogen MLP; ILINet; no spatial sharing\nTrain: half clean episodes, half full-strength local errors'),
          (46,'Nowcaster phase_local; ridge 10.0; correction 1.0','B2 pathogen MLP; Kinsa; neighboring-state sharing\nTrain/evaluate: phase-and-location nowcaster, then forecaster'),
          (46,'Joint recent reconstruction and forecast; local random; pooled eight-horizon loss','B2 pathogen MLP; Kinsa; neighboring-state sharing\nTrain: joint reconstruction/forecast, random revision strength')]
fig,ax=plt.subplots(figsize=(13,6))
for i,(architecture,treatment,label) in enumerate(examples):
    q=raw[(raw.architecture==architecture)&raw.actual_treatment.eq(treatment)&raw.season.eq('2025-2026')&raw.geography.eq('all')]
    ax.scatter(q.baseline,np.full(len(q),i-.12),color='#91979d',s=35,label='Same model trained on unchanged finalized inputs; raw evaluation reports' if i==0 else None)
    ax.scatter(q.combined,np.full(len(q),i+.12),color='#247f88',s=35,label='Specified treatment' if i==0 else None)
    ax.scatter(q.baseline.mean(),i-.12,color='#41464c',marker='D',s=40)
    ax.scatter(q.combined.mean(),i+.12,color='#10454b',marker='D',s=40)
ax.axvline(1,color='gray',ls='--',lw=1)
ax.set_yticks(range(len(examples)),[e[2] for e in examples],fontsize=9);ax.invert_yaxis()
ax.set_xlabel('WIS / matched Hub WIS (lower is better); circles = seeds, diamonds = three-seed means')
ax.set_title('Evaluate 2025–26; train on 2022–23, 2023–24, 2024–25; reporting errors from 2024–25\nFinalized future labels; joint model also learns finalized recent weeks\nB2 availability with final proxies. Raw evaluation reports, except separately nowcast-corrected pipeline.',fontsize=10)
ax.legend(loc='lower left',fontsize=8);ax.grid(axis='x',alpha=.2)
fig.tight_layout();fig.savefig(out/'forward-selected-comparisons.png',dpi=180);plt.close(fig)
