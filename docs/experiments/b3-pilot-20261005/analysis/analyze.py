"""Rebuild pilot comparison tables and figures from manager ranking exports."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
names = {x['scenario']: x['name'] for x in json.loads((ROOT.parent/'design.json').read_text())['rows']}
d = pd.read_csv(ROOT/'ranking/pilot-seed-scores.csv')
d['name'] = d.config_id.map(names)
assert d.name.notna().all()
d.to_csv(ROOT/'scores-labelled.csv', index=False)
a = d[d.season=='equal_season_mean']
s = a.groupby(['name','history','metric']).score.mean().unstack()
s.to_csv(ROOT/'scores-summary.csv')

pairs=[]
for backbone in ('pathogen_distance','target_neighbors_kinsa','series_mlp','series_mixer'):
    reference = backbone + ('_ili_none' if backbone.startswith('series_') else '_finalized')
    for name in sorted(n for n in d.name.unique() if n.startswith(backbone+'_') and n!=reference):
        left=d[(d.name==name)&(d.history=='raw')]
        right=d[(d.name==reference)&(d.history=='raw')]
        j=left.merge(right,on=['seed','season','metric'],suffixes=('_new','_base'))
        for (metric,season),g in j.groupby(['metric','season']):
            pairs.append(dict(reference=reference,treatment=name,metric=metric,season=season,
                              reference_score=g.score_base.mean(),treatment_score=g.score_new.mean(),
                              percent_change=100*(g.score_new.mean()/g.score_base.mean()-1),
                              seeds_improved=int((g.score_new<g.score_base).sum()),seeds=len(g)))
pd.DataFrame(pairs).to_csv(ROOT/'paired-training-comparisons.csv',index=False)

p=a.pivot(index=['name','seed','metric'],columns='history',values='score').reset_index()
for view in ['half','corrected']:
    p[view+'_percent_change']=100*(p[view]/p.raw-1)
p.to_csv(ROOT/'paired-evaluation-comparisons.csv',index=False)

short={
 'finalized':'Unchanged finalized training histories',
 'reported':'Archived training reports + final fills',
 'admission_errors':'Admissions errors in all training seasons',
 'early_errors_recent_reports':'Errors in old seasons; recent reports',
 'synchronous_half_episodes':'Synchronized errors in half the episodes',
 'joint_two_weeks':'Errors + reconstruct 2 weeks (weight .05)',
 'joint_four_weeks':'Errors + reconstruct 4 weeks (weight .25)',
 'synthetic_tree_pipeline':'Synthetic-tree-corrected training histories',
 'real_tree_pipeline':'Real-pair-tree-corrected training histories',
 'quantile_head':'Finalized histories; direct quantile output',
}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(14,7),sharey=True)
for ax,backbone,title in zip(axes,['pathogen_distance','target_neighbors_kinsa'],['Separate pathogen MLPs; distance exchange','Separate target MLPs; neighbors + Kinsa']):
    rows=[backbone+'_'+key for key in short]
    for view,color,offset in [('raw','#365caa',-.2),('half','#0f916e',0),('corrected','#c77729',.2)]:
        vals=[s.loc[(n,view),'six_target_native'] for n in rows]
        ax.scatter(vals,np.arange(len(rows))+offset,label={'raw':'Reported evaluation histories','half':'50/50 forecast mixture','corrected':'Admissions-corrected histories'}[view],color=color,s=36)
    ax.axvline(1,color='gray',lw=1,ls='--');ax.set_title(title);ax.set_xlabel('Equal-season ensemble-relative WIS (lower is better)');ax.grid(axis='x',alpha=.15)
axes[0].set_yticks(np.arange(len(short)),list(short.values()));axes[0].invert_yaxis()
axes[1].legend(fontsize=8,loc='lower right')
fig.suptitle('Training-input treatments and prediction-time admissions correction',fontsize=15)
fig.text(.02,.025,'Labels: finalized next 4 weeks of all six targets; joint models also reconstruct recent finalized histories.\nEvaluate 2025–26 after training 2022–25; retrospectively evaluate 2024–25 after training 2022–24 + 2025–26.\nMean of seeds 42/43; seasons equally weighted. Frozen support: six targets in 2025–26, flu/COVID admissions in 2024–25.',fontsize=9)
fig.tight_layout(rect=(0,.12,1,.95));fig.savefig(ROOT/'training-and-correction.png',dpi=180);plt.close(fig)

fig,axes=plt.subplots(1,3,figsize=(15,6),sharey=True)
for ax,metric,title in zip(axes,['six_target_native','flu_admissions_native','flu_admissions_log'],['Existing multi-target score','Flu admissions: native counts','Flu admissions: log(1 + count)']):
    for prefix,color in [('series_mlp','#365caa'),('series_mixer','#bd633b')]:
        suffixes=['ili_none','ili_pretrain','ili_joint','reported','damped_growth']
        vals=[s.loc[(prefix+'_'+v,'raw'),metric] for v in suffixes]
        ax.plot(vals,np.arange(5),marker='o',color=color,label={'series_mlp':'Shared per-series MLP','series_mixer':'Shared time mixer'}[prefix])
    ax.axvline(1,color='gray',ls='--',lw=1);ax.set_title(title);ax.set_xlabel('Relative WIS (lower is better)');ax.grid(axis='x',alpha=.15)
axes[0].set_yticks(range(5),['Finalized histories; no ILI','Finalized histories; ILI pretraining','Finalized histories; joint ILI learning','Archived training reports + fills','Finalized histories; damped growth anchor']);axes[0].invert_yaxis();axes[-1].legend(fontsize=9)
fig.suptitle('Shared forecaster and historical ILI development',fontsize=15)
fig.text(.02,.025,'Modern finalized labels: next 4 weeks, all six targets. ILI adds its own historical four-week forecasting task (2010–22).\nEvaluate reported histories in 2025–26 after training 2022–25, and 2024–25 after training 2022–24 + 2025–26.\nMeans over two seeds and equally weighted seasons. Multi-target retrospective support is flu/COVID admissions only.',fontsize=9)
fig.tight_layout(rect=(0,.13,1,.95));fig.savefig(ROOT/'shared-models-and-ili.png',dpi=180);plt.close(fig)

for metric in s.columns:
    print('\nTOP',metric);print(s.sort_values(metric).head(8).round(4).to_string())
print('\nFIXED FORECASTER CORRECTORS')
print(s.loc[[n for n in s.index.get_level_values(0).unique() if n.startswith('fixed_') or n=='pathogen_distance_finalized']].round(4).to_string())
