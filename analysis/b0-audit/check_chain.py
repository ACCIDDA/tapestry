"""Material scientific checks for pinned chain stages, before launching fits."""
from pathlib import Path
import json,hashlib,sys
import numpy as np
from tapestry.dataset.build import load
from tapestry.dataset.cv import fold,SEASONS
from tapestry.model.scenario import Scenario
from tapestry.model.objective import LOSS_WEIGHTS,loss_cell_weights
from tapestry.experiment.planner import read_jobs,pinned_inputs

base=load('data/audits/b0/original-panel-unified.npz');records=[]
for stage in sys.argv[1:]:
 root=Path('data/experiments')/f'b1-b0-chain-{stage}'
 config=json.loads((root/'experiment.json').read_text());panel=load(config['dataset'])
 for name in ['dates','locations','targets']:np.testing.assert_equal(panel[name],base[name])
 assert config['eval_members']==2048 and config['device']=='cuda'
 assert all(config[k]==v for k,v in pinned_inputs(config).items())
 jobs=read_jobs(root);assert len(jobs)==1 and jobs[0]['seeds']==[42,43,44]
 scenario=Scenario.from_string(jobs[0]['scenario']);assert scenario.fit_partition=='pathogen' and not scenario.covariate_set
 assert scenario.input_mode==('finalized' if stage in ('07','08','09') else 'finalized_available')
 for held in SEASONS:
  f=fold(panel,scenario,held,inner=True);hidden=set(f.info['validation_weeks'])
  for e in f.train:
   from tapestry.dataset.cv import season
   assert all(season(d)!=held and d not in hidden for i,d in enumerate(e['context_dates']) if e['available'][i].any())
   assert all(season(d)!=held and d not in hidden for i,d in enumerate(e['target_dates']) if e['target_available'][i].any())
  if scenario.input_mode=='finalized':
   before=fold(base,scenario,held,inner=True)
   assert len(f.train)==len(before.train)
   for a,b in zip(f.train,before.train):
    for key in ['values','available','Y']:np.testing.assert_equal(a[key],b[key])
  for ch in [[0,3],[1,4],[2,5]]:
   mask=np.stack([e['target_available'] for e in f.train])
   cw=np.array(LOSS_WEIGHTS['objective']);cw[[i for i in range(6) if i not in ch]]=0
   weights=loss_cell_weights(f.train,cw)
   assert np.isclose(weights.sum(),1)
   assert not weights[:,:, [i for i in range(6) if i not in ch]].any()
  records.append(dict(stage=stage,held=held,fit_examples=len(f.train),validation_examples=len(f.validation),held_out_and_hidden_excluded=True,original_finalized_values_unchanged=True,full_component_weight_mass=1.0))
Path('docs/results/b1-to-b0-chain').mkdir(parents=True,exist_ok=True)
Path('docs/results/b1-to-b0-chain/preflight-'+ '-'.join(sys.argv[1:])+'.json').write_text(json.dumps(records,indent=2)+'\n')
print('Scientific preflight passed:',len(records),'stage/fold combinations')
