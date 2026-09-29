"""Matched descriptive comparisons from the manager's completed-run ranking."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tapestry.model.scenario import Scenario

parser=argparse.ArgumentParser()
parser.add_argument('ranking', type=Path)
args=parser.parse_args()
out=Path('docs/results/forecast-geography-v2')
out.mkdir(parents=True, exist_ok=True)
r=args.ranking
rank=pd.read_csv(r/'configuration_ranking.csv')
runs=pd.read_csv(r/'run_scores.csv')
seasons=pd.read_csv(r/'season_composite_scores.csv')
labels=[]
for config in rank.config_id:
    s=Scenario.from_string(config)
    bundle=('None' if not s.covariate_set else 'Kinsa' if s.covariate_set=='kinsa' else
            'Wastewater' if s.covariate_set=='ww_wval_like' else 'Claims' if s.covariate_set=='inpatient+outpatient' else
            'New flu' if set(s.covariate_set.split('+'))=={'ilinet','clinical_lab','flusurv'} else 'Mixed')
    labels.append(dict(config_id=config, spatial=s.spatial, bundle=bundle, representation=s.covariate_encoder,
                       name=f'{s.spatial} / {bundle} / {s.covariate_encoder}' if bundle!='None' else f'{s.spatial} / no covariates'))
labels=pd.DataFrame(labels)
rank=rank.merge(labels,on='config_id')
runs=runs.merge(labels,on='config_id')
seasons=seasons.merge(labels,on='config_id')
assert len(rank)==63 and len(runs[runs.geography=='all'])==189
assert (rank.seeds==3).all()
base=runs[runs.bundle=='None'][['spatial','seed','geography','combined']].rename(columns={'combined':'baseline'})
paired=runs.merge(base,on=['spatial','seed','geography'],validate='many_to_one')
paired['delta']=paired.combined-paired.baseline
paired['percent']=100*(paired.combined/paired.baseline-1)
paired['improved']=paired.delta<0
summary=paired.groupby(['config_id','geography'],as_index=False).agg(paired_delta=('delta','mean'),
    paired_percent=('percent','mean'),percent_sd=('percent','std'),seeds_improved=('improved','sum'))
summary=summary.merge(labels,on='config_id').merge(rank[['config_id','rank','combined_mean','combined_sd','states_dc_combined_mean','US_combined_mean']],on='config_id')
base_s=seasons[seasons.bundle=='None'][['spatial','seed','geography','season','combined']].rename(columns={'combined':'baseline'})
season_paired=seasons.merge(base_s,on=['spatial','seed','geography','season'],validate='many_to_one')
season_paired['delta']=season_paired.combined-season_paired.baseline
rank.to_csv(out/'named_ranking.csv',index=False)
paired.to_csv(out/'paired_seed_scores.csv',index=False)
summary.to_csv(out/'matched_comparisons.csv',index=False)
season_paired.to_csv(out/'paired_season_scores.csv',index=False)

# Each heatmap cell is a matched three-seed mean percentage change.
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
for region, suffix in [('all','overall'),('states_dc','states'),('US','us')]:
    fig, axes=plt.subplots(1,3,figsize=(13,5),sharey=True,layout='constrained')
    part=summary[(summary.geography==region)&(summary.bundle!='None')]
    limit=max(2,float(part.paired_percent.abs().max()))
    for ax, spatial in zip(axes,['none','pooled','attention']):
        table=part[part.spatial==spatial].pivot(index='bundle',columns='representation',values='paired_percent').reindex(index=['Claims','New flu','Mixed','Wastewater','Kinsa'],columns=['raw','smooth','summary','shared'])
        im=ax.imshow(table,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
        for y in range(5):
            for x in range(4):
                v=table.iloc[y,x]
                ax.text(x,y,f'{v:+.1f}%',ha='center',va='center',color='white' if abs(v)>.6*limit else 'black')
        ax.set_xticks(range(4),table.columns,rotation=30,ha='right')
        ax.set_yticks(range(5),table.index)
        baseline=rank[(rank.spatial==spatial)&(rank.bundle=='None')].iloc[0]
        col={'all':'combined_mean','states_dc':'states_dc_combined_mean','US':'US_combined_mean'}[region]
        ax.set_title(f'{spatial}\nNo-covariate WIS ratio {baseline[col]:.3f}')
    fig.colorbar(im,ax=axes,label='% change vs same-geography no-covariate control')
    fig.suptitle(f'{region}: covariate benefit by representation\nNegative (blue) is better; mean of three paired seed changes',fontsize=14)
    fig.savefig(out/f'covariate-effects-{suffix}.png',dpi=170)
    plt.close(fig)

# Seed dispersion for leaders and controls, with explicit labels.
selected=list(dict.fromkeys(list(rank.head(10).config_id)+list(rank[rank.bundle=='None'].config_id)))
table=rank.set_index('config_id').loc[selected].sort_values('combined_mean')
fig,ax=plt.subplots(figsize=(10,7),layout='constrained')
for y,(config,row) in enumerate(table.iterrows()):
    values=runs[(runs.config_id==config)&(runs.geography=='all')].sort_values('seed').combined
    ax.scatter(values,[y]*len(values),color='#3675a9',s=35,alpha=.7)
    ax.scatter([row.combined_mean],[y],color='black',marker='D',s=24)
ax.set_yticks(range(len(table)),table.name)
ax.invert_yaxis();ax.axvline(1,color='gray',linestyle='--')
ax.set_xlabel('Relative WIS (lower is better; 1 = Hub ensemble)')
ax.set_title('Top ten configurations and all geographic controls\nDots: seeds; black diamonds: means')
fig.savefig(out/'leaders-and-seeds.png',dpi=170);plt.close(fig)

print('BASELINES')
print(rank[rank.bundle=='None'][['name','combined_mean','combined_sd','states_dc_combined_mean','US_combined_mean']].to_string(index=False))
print('LEADERS')
print(summary[summary.geography=='all'].sort_values('rank')[['rank','name','combined_mean','combined_sd','paired_percent','seeds_improved']].head(20).to_string(index=False))
print('MATCHED MEANS')
print(summary[(summary.geography=='all')&(summary.bundle!='None')].groupby(['spatial','representation']).paired_percent.mean().unstack().round(2))
print('BUNDLE MEANS')
print(summary[(summary.geography=='all')&(summary.bundle!='None')].groupby(['spatial','bundle']).paired_percent.mean().unstack().round(2))
print('SEASONS: TOP AND CONTROLS')
print(seasons[(seasons.geography=='all')&seasons.config_id.isin(selected[:3]+list(rank[rank.bundle=='None'].config_id))].groupby(['name','season']).combined.mean().unstack().round(3))

# Supporting diagnostics use the scorer's same season-first target weighting.
from tapestry.evaluation.totals import season_composites
cells=pd.read_csv(r/'season_scores.csv')
coverage=season_composites(cells,value='model_coverage_95').groupby(['config_id','geography'],as_index=False).combined.mean().rename(columns={'combined':'coverage95'})
coverage=coverage.merge(labels,on='config_id')
coverage.to_csv(out/'coverage95.csv',index=False)
ensemble=season_composites(cells,value='ensemble_coverage_95')
manifest=json.loads((r/'manifest.json').read_text())
parameter_rows=[]
fingerprints=set()
for run in manifest['runs']:
    m=json.loads((Path(run['path'])/'manifest.json').read_text())
    for season,fold in m['fold_manifests'].items():
        fingerprints.add(fold['dataset_sha256'])
        parameter_rows.append(dict(config_id=run['config_id'],seed=run['seed'],season=season,parameters=fold['parameter_count']))
parameters=pd.DataFrame(parameter_rows).merge(labels,on='config_id')
parameters.to_csv(out/'parameter_counts.csv',index=False)
assert len(fingerprints)==1 and len(parameter_rows)==567
(out/'analysis_provenance.json').write_text(json.dumps(dict(ranking=str(r),runs=189,configurations=63,folds=567,
    panel_sha256=next(iter(fingerprints)),paired_percent='Mean across seeds of 100*(model/baseline-1), paired within geography.',
    ensemble_coverage95=float(ensemble[ensemble.geography=='all'].combined.mean()),
    assumptions=['Retrospective finalized values masked by Wednesday reporting availability.',
                 'Three seeds quantify optimization variability, not independent epidemiological replicates.',
                 'Selection and conclusions use these same development folds; no independent confirmation.',
                 'Coverage uses the same location/target/season weighting as the primary score.']),indent=2)+'\n')
leaders=list(rank.head(3).config_id)+list(rank[(rank.spatial=='none')&(rank.bundle=='None')].config_id)
season_table=seasons[(seasons.geography=='all')&seasons.config_id.isin(leaders)].groupby(['name','season']).combined.mean().unstack()
season_table.to_csv(out/'leading_seasons.csv')
fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
for name,row in season_table.iterrows():
    ax.plot(range(3),row.values,marker='o',label=name,linestyle='--' if 'no covariates' in name else '-')
ax.set_xticks(range(3),season_table.columns);ax.axhline(1,color='gray',linewidth=1)
ax.set_ylabel('Relative WIS (three-seed mean)');ax.set_title('Leading models by held-out season\nTarget support differs across seasons')
ax.legend(fontsize=9);fig.savefig(out/'leading-seasons.png',dpi=170);plt.close(fig)
