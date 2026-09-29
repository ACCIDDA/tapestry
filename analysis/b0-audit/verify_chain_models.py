"""Check restored fitting endpoint against original B0 before forecast regeneration."""
from pathlib import Path
import json
import torch,pandas as pd
from tapestry.experiment.planner import read_jobs,seed_state

root=Path('data/experiments/b1-b0-chain-09');job=read_jobs(root)[0];rows=[]
for seed in [42,43,44]:
 attempt,_,_=seed_state(root,job['scenario'],seed)
 if attempt is None:continue
 refroot=Path('data/experiments')/('b0-exact-reproduction-l40' if seed==44 else 'b0-training-code-reference')
 refjob=next(j for j in read_jobs(refroot) if 'fit_partition=pathogen' in j['scenario']);ref,_,_=seed_state(refroot,refjob['scenario'],seed)
 for season in ['2023-2024','2024-2025','2025-2026']:
  a=attempt/f'eval_{season}/model.pt';b=ref/f'legacy/eval_{season}/model.pt'
  if not a.exists():continue
  x=torch.load(a,map_location='cpu',weights_only=False);y=torch.load(b,map_location='cpu',weights_only=False)
  assert x['groups']==y['groups']
  for i,(p,q) in enumerate(zip(x['components'],y['components'])):
   assert p['state_dict'].keys()==q['state_dict'].keys()
   error=max(float((p['state_dict'][k]-q['state_dict'][k]).abs().max()) for k in p['state_dict'])
   rows.append(dict(seed=seed,season=season,component=i,max_parameter_difference=error,parameters_equal=error==0))
if rows:
 f=pd.DataFrame(rows);f.to_csv('docs/results/b1-to-b0-chain/endpoint_model_verification.csv',index=False);print(f.groupby(['seed','season']).max_parameter_difference.max().to_string())
