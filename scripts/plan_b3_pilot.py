"""Generate the small B3 pilot; only --plan registers it with the shared manager."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import shlex
import numpy as np
from tapestry.model.scenario import Scenario

p=argparse.ArgumentParser();p.add_argument('--experiment',default='b3-pilot-20261005');p.add_argument('--plan',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-inner',action='store_true');args=p.parse_args()
base=Scenario(pilot_method='finalized',input_mode='scheduled_final',evaluation_seasons='recent_two',
    epochs=120,patience=15,ed_transform='logit',input_normalization='b0',covariate_encoder='summary',
    fit_partition='pathogen',spatial='distance',reporting_missingness=False,reporting_strength=.5)
rows=[]
def add(name,s):rows.append(dict(name=name,scenario=s.scenario_string))
for label,anchor in [('pathogen_distance',base),('target_neighbors_kinsa',replace(base,fit_partition='target',spatial='neighbors',covariate_set='kinsa'))]:
    add(f'{label}_quantile_head',replace(anchor,decoder='quantile'))
    treatments=[('finalized',{}),('reported',dict(pilot_method='reported')),
      ('admission_errors',dict(pilot_method='errors')),
      ('early_errors_recent_reports',dict(pilot_method='errors',revision_scope='early_actual')),
      ('synchronous_half_episodes',dict(pilot_method='errors',reporting_method='synchronous_log',reporting_probability=.5)),
      ('joint_two_weeks',dict(pilot_method='joint',nowcast_weeks=2,joint_weight=.05)),
      ('joint_four_weeks',dict(pilot_method='joint',nowcast_weeks=4,joint_weight=.25)),
      ('synthetic_tree_pipeline',dict(pilot_method='corrected')),
      ('real_tree_pipeline',dict(pilot_method='two_stage',pilot_nowcaster='real_tree'))]
    for name,kw in treatments:add(f'{label}_{name}',replace(anchor,**kw))
for corrector in ('real_tree','real_mlp','pretrained_mlp'):
    add(f'fixed_forecaster_{corrector}',replace(base,pilot_nowcaster=corrector))
for encoder in ('series_mlp','series_mixer'):
    anchor=replace(base,encoder=encoder,fit_partition='all',spatial='none',width=32)
    for ili in ('none','pretrain','joint'):
        add(f'{encoder}_ili_{ili}',replace(anchor,ili_training=ili))
    add(f'{encoder}_reported',replace(anchor,pilot_method='reported'))
    add(f'{encoder}_damped_growth',replace(anchor,growth_anchor=True))
# Fixed-seed random recipes probe treatment interactions before the large random search.
rng=np.random.default_rng(20261005)
for i,method in enumerate(('finalized','reported','errors','joint')):
    add(f'random_recipe_{i}',replace(base,pilot_method=method,lookback=int(rng.choice([8,10,12])),
        width=int(rng.choice([32,64])),lr=float(rng.choice([.0005,.001])),weight_decay=float(rng.choice([0,.0001])),
        spatial=str(rng.choice(['none','distance','neighbors'])),joint_weight=.05))
if args.smoke:
    names=['pathogen_distance_finalized','pathogen_distance_reported','pathogen_distance_joint_two_weeks',
           'pathogen_distance_real_tree_pipeline','fixed_forecaster_pretrained_mlp','series_mlp_ili_joint','series_mixer_ili_pretrain']
    rows=[r for r in rows if r['name'] in names]
    rows=[dict(r,scenario=replace(Scenario.from_string(r['scenario']),epochs=2,patience=0).scenario_string) for r in rows]
if args.smoke_inner:
    names=['pathogen_distance_quantile_head','pathogen_distance_real_tree_pipeline','series_mlp_ili_pretrain','pathogen_distance_early_errors_recent_reports']
    rows=[dict(r,scenario=replace(Scenario.from_string(r['scenario']),epochs=3,patience=1).scenario_string) for r in rows if r['name'] in names]
assert len({r['scenario'] for r in rows})==len(rows)
folder=Path('docs/experiments')/args.experiment;folder.mkdir(parents=True,exist_ok=True)
(folder/'design.json').write_text(json.dumps(dict(configurations=len(rows),seeds=[42,43],rows=rows),indent=2)+'\n')
(folder/'scenarios.txt').write_text('\n'.join(r['scenario'] for r in rows)+'\n')
command=[sys.executable,'-m','tapestry.experiment.planner','plan','-e',args.experiment,'-s',*[r['scenario'] for r in rows],'--seeds','42','43','--device','cuda']
print(f'{len(rows)} configurations, seeds 42/43, both seasons')
print('Manager plan: '+shlex.join(command))
if args.plan:subprocess.run(command,check=True)
