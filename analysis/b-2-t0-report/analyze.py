"""Analyze the completed sweep; uncertainty describes seed randomness, not new-season error."""
import json
from pathlib import Path
from dataclasses import asdict
import numpy as np
import pandas as pd
from tapestry.model.scenario import Scenario

OUT = Path('analysis/b-2-t0-report')
ranking = Path((OUT / 'ranking-path.txt').read_text().strip())
r = pd.read_csv(ranking / 'configuration_ranking.csv').sort_values('combined_mean')
s = pd.read_csv(ranking / 'season_composite_scores.csv')
s = s[s.geography.eq('all')]
u = pd.read_csv(ranking / 'run_scores.csv')
u = u[u.geography.eq('all')]
design = pd.DataFrame(json.load(open('analysis/b-2-t0/design.json'))['rows']).rename(columns={'scenario':'config_id'})
named = r.merge(design, on='config_id', validate='one_to_one')
named['label'] = ['C'+str(i) for i in range(1,len(named)+1)]
assert len(named)==309 and named.seeds.eq(3).all(), 'Analysis requires all configurations and three seeds.'
named.to_csv(OUT/'named_ranking.csv', index=False)
seed = u.pivot(index='config_id', columns='seed', values='combined')
season = s.groupby(['config_id','season']).combined.mean().unstack()
# A Student-t interval with 2 df: describes uncertainty across three fitting seeds only.
T = 4.302652729911275
b = pd.read_csv('docs/results/b1-to-b0-chain/season_composite_scores.csv')
b = b[b.config_id.eq('stage09') & b.geography.eq('all') & b.season.isin(['2024-2025','2025-2026'])]
bs = b.groupby('seed').combined.mean()
deltas = seed.subtract(bs, axis=1)
ci = pd.DataFrame({'mean_delta':deltas.mean(axis=1),'seed_sd':deltas.std(axis=1),'better_seeds':deltas.lt(0).sum(axis=1)})
ci['lo']=ci.mean_delta-T*ci.seed_sd/np.sqrt(3);ci['hi']=ci.mean_delta+T*ci.seed_sd/np.sqrt(3)
ci.reset_index().to_csv(OUT/'b0_seed_comparison.csv',index=False)

# Match on EVERY scenario field except the one intervention represented by each design factor.
fields={'backbone':['encoder','fit_partition'],'sources':['covariate_set'],'spatial':['spatial'],
        'mask_rate':['mask_rate'],'coordinates':['coordinates'],'signal_features':['signal_features'],
        'covariate_encoder':['covariate_encoder']}
base={'backbone':'pathogen-mlp','sources':'none','spatial':'none','mask_rate':0.0,
      'coordinates':False,'signal_features':'none','covariate_encoder':'summary'}
settings={c:asdict(Scenario.from_string(c)) for c in named.config_id}
pairrows=[]
for factor, omitted in fields.items():
    groups={}
    for row in named.to_dict('records'):
        conf={k:v for k,v in settings[row['config_id']].items() if k not in omitted}
        key=json.dumps(conf,sort_keys=True)
        groups.setdefault(key,{})[row[factor]]=row['config_id']
    for key, variants in groups.items():
        if base[factor] not in variants:continue
        ref=variants[base[factor]]
        for value,c in variants.items():
            if value==base[factor]:continue
            for sd in seed.columns:
                pairrows.append(dict(factor=factor,reference=str(base[factor]),value=str(value),
                  reference_config=ref,config_id=c,seed=int(sd),delta=float(seed.loc[c,sd]-seed.loc[ref,sd])))
pairs=pd.DataFrame(pairrows);pairs.to_csv(OUT/'matched_pairs.csv',index=False)
effects=[]
for (factor,ref,val),g in pairs.groupby(['factor','reference','value']):
    byseed=g.groupby('seed').delta.mean();ctx=g.groupby(['reference_config','config_id']).delta.mean()
    mean=byseed.mean();half=T*byseed.std()/np.sqrt(3)
    effects.append(dict(factor=factor,reference=ref,value=val,contexts=len(ctx),mean_delta=mean,
                        seed_ci_low=mean-half,seed_ci_high=mean+half,improved_contexts=int((ctx<0).sum()),
                        improved_seeds=int((byseed<0).sum())))
effects=pd.DataFrame(effects).sort_values(['factor','mean_delta']);effects.to_csv(OUT/'matched_effects.csv',index=False)

# Leave-one-source-out effects: all-minus-X versus all, other scenario fields fixed.
loo=[]
for row in named[named.sources.str.startswith('all-minus-')].to_dict('records'):
    c=row['config_id'];candidate={k:v for k,v in settings[c].items() if k!='covariate_set'}
    matches=[z for z in named[named.sources.eq('all')].config_id if {k:v for k,v in settings[z].items() if k!='covariate_set'}==candidate]
    for ref in matches:
        for sd in seed.columns:loo.append(dict(source_removed=row['sources'][10:],backbone=row['backbone'],seed=sd,delta=seed.loc[c,sd]-seed.loc[ref,sd]))
loo=pd.DataFrame(loo);loo.to_csv(OUT/'source_removal_pairs.csv',index=False)
print('COUNTS',len(named),int((named.combined_mean<1).sum()),(season<1).sum().to_dict(),flush=True)
print('BEST',named.head(5)[['label','backbone','sources','spatial','coordinates','mask_rate','combined_mean','combined_sd']].to_string(index=False))
print('WORST',named.tail(5)[['label','backbone','sources','spatial','mask_rate','combined_mean','combined_sd']].to_string(index=False))
print('B0 TOP',ci.loc[named.head(5).config_id].to_string())
print('EFFECTS',effects.to_string(index=False))
print('REMOVAL',loo.groupby('source_removed').delta.agg(['mean','min','max']).to_string())

# Compact supplementary heatmap, using the comparable source-attribution subset.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
subset=named[named.spatial.eq('none') & ~named.coordinates & named.mask_rate.eq(.2) & named.covariate_encoder.eq('summary') & named.signal_features.eq('none')]
hm=subset.pivot(index='sources',columns='backbone',values='combined_mean')
fig,ax=plt.subplots(figsize=(8,8));sns.heatmap(hm,annot=True,fmt='.3f',cmap='RdBu_r',center=1,ax=ax)
ax.set_title('Matched source × backbone comparison\nMean of 3 seeds; lower is better; hub ensemble = 1')
fig.tight_layout();fig.savefig(OUT/'source-backbone-heatmap.png',dpi=180);plt.close(fig)
