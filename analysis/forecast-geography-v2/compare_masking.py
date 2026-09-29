"""Paired artificial-masking ablation; appendable evidence for the existing report."""
import argparse,json
from dataclasses import replace
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tapestry.model.scenario import Scenario
from tapestry.evaluation.totals import season_composites
p=argparse.ArgumentParser();p.add_argument('ranking',type=Path);args=p.parse_args()
old=Path('data/experiments/forecast-geography-v2/ranking-e53fd92c4f32')
new=args.ranking
out=Path('docs/results/forecast-geography-v2/no-mask');out.mkdir(parents=True,exist_ok=True)
labels=pd.read_csv(out.parent/'named_ranking.csv',keep_default_na=False)[['config_id','name','spatial','bundle','representation']]
a=pd.read_csv(old/'run_scores.csv');b=pd.read_csv(new/'run_scores.csv')
b['masked_config']=b.config_id.map(lambda x:replace(Scenario.from_string(x),mask_rate=.5).scenario_string)
assert b.config_id.nunique()==32 and len(b[b.geography=='all'])==96
pairs=b.merge(a.rename(columns={'config_id':'masked_config'}),on=['masked_config','seed','geography'],suffixes=('_no_mask','_masked'),validate='one_to_one')
pairs['delta']=pairs.combined_no_mask-pairs.combined_masked
pairs['percent']=100*(pairs.combined_no_mask/pairs.combined_masked-1)
pairs['improved']=pairs.delta<0
pairs=pairs.merge(labels.rename(columns={'config_id':'masked_config'}),on='masked_config',validate='many_to_one')
summary=pairs.groupby(['config_id','masked_config','name','spatial','bundle','representation','geography'],as_index=False).agg(
    masked=('combined_masked','mean'),no_mask=('combined_no_mask','mean'),no_mask_sd=('combined_no_mask','std'),
    delta=('delta','mean'),paired_percent=('percent','mean'),seeds_improved=('improved','sum'))
summary.to_csv(out/'matched_configurations.csv',index=False);pairs.to_csv(out/'paired_seeds.csv',index=False)
# Matching must be at the seed/season/region level, with the same non-masking scenario.
sa=pd.read_csv(old/'season_composite_scores.csv');sb=pd.read_csv(new/'season_composite_scores.csv')
sb['masked_config']=sb.config_id.map(lambda x:replace(Scenario.from_string(x),mask_rate=.5).scenario_string)
sp=sb.merge(sa.rename(columns={'config_id':'masked_config'}),on=['masked_config','seed','geography','season'],suffixes=('_no_mask','_masked'),validate='one_to_one')
sp['delta']=sp.combined_no_mask-sp.combined_masked
sp=sp.merge(labels.rename(columns={'config_id':'masked_config'}),on='masked_config')
sp.to_csv(out/'paired_seasons.csv',index=False)
# Both saved rankings must use identical settings and exactly matching task support.
manifests=[json.loads((root/'manifest.json').read_text()) for root in (old,new)]
for key in ['quantile_levels','target_weights','score_version','us_weight','admissions_weight','ed_weight']:
    assert manifests[0][key]==manifests[1][key],key
lookup={(run['config_id'],run['seed']):run for run in manifests[0]['runs']}
keys=['target','season','location','horizon','n'];fingerprints=set()
for run in manifests[1]['runs']:
    config=replace(Scenario.from_string(run['config_id']),mask_rate=.5).scenario_string
    counterpart=lookup[(config,run['seed'])]
    tables=[pd.read_csv(Path(x['path'])/'totals.csv')[keys].sort_values(keys).reset_index(drop=True) for x in (run,counterpart)]
    pd.testing.assert_frame_equal(*tables)
    for x in (run,counterpart):
        settings=json.loads((Path(x['path'])/'run.json').read_text())['settings']
        fingerprints.add(tuple(settings[k] for k in ['dataset_sha256','population_sha256','frozen_manifest_sha256','eval_members']))
assert len(fingerprints)==1
coverage=[]
for condition,root in [('masked',old),('no_mask',new)]:
    c=season_composites(pd.read_csv(root/'season_scores.csv'),value='model_coverage_95')
    c=c.groupby(['config_id','geography'],as_index=False).combined.mean().rename(columns={'combined':'coverage95'})
    c['condition']=condition;coverage.append(c)
pd.concat(coverage).to_csv(out/'coverage95.csv',index=False)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
all_=summary[summary.geography=='all'].sort_values('no_mask')
fig,ax=plt.subplots(figsize=(11,12),layout='constrained')
for y,row in enumerate(all_.itertuples()):
    ax.plot([row.masked,row.no_mask],[y,y],color='#277c65' if row.no_mask<row.masked else '#bb554c',linewidth=2)
    ax.scatter(row.masked,y,color='#777777',s=25)
    ax.scatter(row.no_mask,y,color='#2176ae',s=28)
ax.set_yticks(range(len(all_)),all_.name);ax.invert_yaxis();ax.axvline(1,color='gray',linestyle='--')
ax.scatter([],[],color='#777777',label='50% episode masking');ax.scatter([],[],color='#2176ae',label='No artificial masking')
ax.legend(loc='lower right');ax.set_xlabel('Three-seed mean relative WIS (lower is better)')
ax.set_title('Removing artificial masking: the top 32 original formulations\nGreen line: removal helps; red line: removal hurts')
fig.savefig(out/'matched-masking.png',dpi=160);plt.close(fig)
meta=dict(configurations=32,seeds=3,runs=96,rankings=[str(old),str(new)],
    improved_configurations=int((all_.delta<0).sum()),improved_seeds=int(pairs[pairs.geography=='all'].improved.sum()),
    average_paired_percent=float(all_.paired_percent.mean()),median_paired_percent=float(all_.paired_percent.median()),
    support='Every matched seed has identical frozen target/season/location/horizon/n rows and pinned inputs.',
    caveat='Selected top half of the original masked ranking; no independent confirmation and no inference to all 63 formulations.')
(out/'comparison_provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
print('LEADERS');print(all_[['name','masked','no_mask','no_mask_sd','paired_percent','seeds_improved']].head(15).to_string(index=False))
print('GEOGRAPHY');print(all_.groupby('spatial')[['masked','no_mask','paired_percent']].mean())
print('PREVIOUS TOP');print(all_[all_.name.isin(['none / Mixed / summary','none / New flu / smooth','none / Kinsa / summary','none / no covariates','attention / no covariates'])][['name','masked','no_mask','paired_percent','seeds_improved']].to_string(index=False))
