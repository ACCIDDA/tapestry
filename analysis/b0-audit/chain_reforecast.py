"""Manager adapter: regenerate held-out forecasts with fixed evaluation draws only."""
from pathlib import Path
import json,hashlib,os
from dataclasses import replace
from datetime import date,timedelta
import numpy as np,torch
from tapestry.experiment.planner import read_jobs,seed_state
from tapestry.dataset.build import load
from tapestry.dataset.cv import fold,SEASONS
from tapestry.model.scenario import Scenario
from tapestry.model.network import load_model
from tapestry.experiment.training import evaluate
from tapestry.evaluation.totals import score_run


def fit(args):
 torch.set_num_threads(int(os.environ.get('TAPESTRY_TORCH_THREADS','1')))
 experiment=Path(__file__).resolve().parents[3]
 protocol=json.loads((experiment/'reforecast-protocol.json').read_text())
 source_root=Path('data/experiments')/protocol['source_experiment']
 source,record,complete=seed_state(source_root,protocol['source_scenario'],args.seed)
 assert complete,('Source fit is not complete',source,record)
 scenario=Scenario.from_string(args.scenario);panel=load(args.dataset)
 output=Path(args.output);output.mkdir(parents=True,exist_ok=True);records=[]
 for held in SEASONS:
  folder=output/f'eval_{held}';folder.mkdir()
  model_path=source/f'eval_{held}/model.pt'
  checkpoint=torch.load(model_path,map_location='cpu',weights_only=False)
  model=load_model(checkpoint).to(args.device)
  # All stages use the original full-finalized evaluation calendar, including
  # unscored origins, so shared random draws stay aligned by forecast origin.
  full=fold(panel,replace(scenario,input_mode='finalized'),held)
  if scenario.input_mode=='finalized_available':
   t_index={str(d):i for i,d in enumerate(panel['dates'])}
   w_index={str(d):i for i,d in enumerate(panel['issuance_dates'])}
   for episode in full.score:
    issuance=(date.fromisoformat(episode['context_dates'][-1])+timedelta(days=4)).isoformat()
    w=w_index[issuance]
    visible=np.stack([np.isfinite(panel['asof_targets'][w,t_index[day]]).T if day in t_index else np.zeros_like(episode['available'][0]) for day in episode['context_dates']])
    episode['available']=episode['available'] & visible
    episode['known_final']=episode['available'].copy()
    episode['values']=np.where(episode['available'],episode['values'],0)
    episode['issuance']=issuance
  # Explicitly shared across every stage and identical to historical B0.
  torch.manual_seed(args.seed+1000)
  evaluate(model,full.score,args.eval_members,args.device,folder)
  (folder/'model.pt').symlink_to(model_path.resolve())
  metadata=dict(checkpoint['metadata'],evaluation_seed=args.seed+1000,evaluation_only=True,source_model=str(model_path),source_model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),evaluation_input_mode=scenario.input_mode)
  (folder/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
  records.append(dict(season=held,source_model=str(model_path),sha256=metadata['source_model_sha256'],epochs=[r.get('epochs') for r in metadata.get('records',[])]))
 (output/'manifest.json').write_text(json.dumps(dict(scenario=args.scenario,seed=args.seed,folds=list(SEASONS),protocol=protocol,source_run=str(source),source_models=records,eval_members=args.eval_members),indent=2)+'\n')
 score_run(output,args.frozen)
