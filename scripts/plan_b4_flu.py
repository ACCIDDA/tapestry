"""Flu admissions + ED: 150 random recipes × four matched pathogen-input choices."""
import argparse,json,subprocess,sys
from dataclasses import replace
from pathlib import Path
import numpy as np
from tapestry.model.scenario import Scenario
p=argparse.ArgumentParser();p.add_argument('--experiment',default='b4-flu-600-20261005');p.add_argument('--plan',action='store_true');p.add_argument('--smoke',action='store_true');args=p.parse_args()
rng=np.random.default_rng(20261006)
base=Scenario(pilot_method='finalized',forecast_targets='flu',loss_weights='flu_hosp_ed',input_mode='scheduled_final',evaluation_seasons='recent_two',epochs=180,patience=25,ed_transform='logit',input_normalization='b0',covariate_encoder='summary',fit_partition='target',spatial='neighbors',reporting_missingness=False,reporting_strength=.5)
methods=['corrected','corrected','corrected','two_stage','two_stage','early','early','reported','joint','finalized']
rows=[]
def draw(method,shared=False):
 kw=dict(lookback=int(rng.choice([8,12,16])),width=int(rng.choice([32,64,96])),lr=float(rng.choice([.0003,.0005,.001,.002])),weight_decay=float(rng.choice([0,.0001,.001])),count_transform=str(rng.choice(['fourth_root','sqrt','log1p'])),reporting_strength=float(rng.choice([.25,.5,.75,1.])),correction_weeks=int(rng.choice([2,4,8])),correction_strength=float(rng.choice([.5,.75,1.])),correction_penalty=float(rng.choice([3.,10.,30.])),pilot_method='errors' if method=='early' else method,revision_scope='early_actual' if method=='early' else 'all',pilot_nowcaster='real_tree' if method=='two_stage' else str(rng.choice(['synthetic_tree','synthetic_tree','real_tree'])),nowcast_weeks=int(rng.choice([2,4])),joint_weight=float(rng.choice([.025,.05,.1,.25])))
 if method=='corrected':kw['pilot_nowcaster']='synthetic_tree'
 if shared:kw.update(encoder=str(rng.choice(['series_mixer','series_mixer','series_mlp'])),fit_partition='all',spatial='none',covariate_set='',epochs=320,patience=40,decoder='quantile')
 else:kw.update(fit_partition=str(rng.choice(['target','target','target','pathogen'])),spatial=str(rng.choice(['neighbors','neighbors','distance','none'])),covariate_set=str(rng.choice(['kinsa','kinsa',''])),decoder=str(rng.choice(['legacy','legacy','quantile'])),epochs=int(rng.choice([160,240])),patience=25)
 return replace(base,**kw)
def add_group(s,block,label):
 for scope in ['flu','flu_covid','flu_rsv','all']:
  rows.append(dict(name=f'{block:03d}_{label}_{scope}',block=block,scenario=replace(s,pathogen_inputs=scope).scenario_string))
for i in range(110):add_group(draw(methods[i%10]),i,methods[i%10])
for j in range(10):
 s=draw(methods[j],True)
 for k,(mode,units) in enumerate([('none','own'),('pretrain','own'),('pretrain','flu_scaled'),('joint','flu_scaled')]):
  add_group(replace(s,ili_training=mode,ili_units=units,ili_steps=400,ili_weight=.05),110+4*j+k,f'{methods[j]}_{s.encoder}_{mode}_{units}')
assert len(rows)==600 and len({r['scenario'] for r in rows})==600
if args.smoke:
 names=[0,12,20,28,32,36,440,444,448,452]
 rows=[dict(rows[i],scenario=replace(Scenario.from_string(rows[i]['scenario']),epochs=3,patience=1,ili_steps=3).scenario_string) for i in names]
folder=Path('docs/experiments')/args.experiment;folder.mkdir(parents=True,exist_ok=True)
(folder/'design.json').write_text(json.dumps(dict(configurations=len(rows),seeds=[42,43],design_seed=20261006,rows=rows),indent=2)+'\n')
(folder/'scenarios.txt').write_text('\n'.join(r['scenario'] for r in rows)+'\n')
print(f'{len(rows)} configurations × 2 seeds = {len(rows)*2} runs')
if args.plan:subprocess.run([sys.executable,'-m','tapestry.experiment.planner','plan','-e',args.experiment,'-s',*[r['scenario'] for r in rows],'--seeds','42','43','--device','cuda'],check=True)
