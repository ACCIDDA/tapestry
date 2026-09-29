"""Locate the residual rewritten/B0 fitting discrepancy before terminal chain runs."""
import sys,json
from pathlib import Path
import numpy as np, torch
import tapestry
# Both implementations loaded independently under their original module names.
tapestry.__path__.append(str(Path('data/experiments/b0-exact-reproduction/legacy-code/src/tapestry').resolve()))
from tapestry.models import season_cv as old_cv
from tapestry.models import run as old_run
from tapestry.models import experiments as old_exp
from tapestry.models.bundles import supervision
from tapestry.models.b0 import B0
from tapestry.model_data import FinalizedDataset
from tapestry.dataset import cv
from tapestry.dataset.build import load
from tapestry.experiment.training import model_options,populations,unique_truth,to_tensors
from tapestry.model.network import Model,fair_crps_cells
from tapestry.model.objective import loss_cell_weights,loss_scales,LOSS_WEIGHTS
from tapestry.model.scenario import Scenario
from argparse import Namespace

torch.set_num_threads(1)
panel=load('data/audits/b0/original-panel-unified.npz');ds=FinalizedDataset.load('data/processed/build_b_finalized.npz')
scenario=Scenario(ed_transform='logit',epochs=300,patience=30,fit_partition='pathogen')
config=json.loads(Path('data/experiments/b0-exact-reproduction/legacy-configs.json').read_text())['pathogen']
args=Namespace(**config);args.device='cpu';args.seed=42;args.population_file='data/audits/b0/populations-from-checkpoint.csv'
records=[]
for held in cv.SEASONS:
 new=cv.fold(panel,scenario,held,inner=True)
 old,old_val,old_scales,_=old_cv.validation_split(ds,held,12,4)
 assert len(old)==len(new.train)
 for a,b in zip(old,new.train):
  np.testing.assert_equal(a['X'][:,:,0],b['values']);np.testing.assert_equal(a['X'][:,:,1].astype(bool),b['available']);np.testing.assert_equal(a['Y'],b['Y'])
 pop=populations('data/audits/b0/populations-from-checkpoint.csv',new.train[0]['locations'])
 options=model_options(new.train,scenario,pop);oo=old_exp.model_options(old,args)
 old_sc=np.asarray(old_scales);new_sc=np.asarray(loss_scales(unique_truth(new.train)))
 print('scales',held,np.max(abs(old_sc-new_sc)),flush=True)
 # Float/dict representation can differ but values must agree.
 diffs={k:(oo.get(k),options.get(k)) for k in set(oo)|set(options) if oo.get(k)!=options.get(k)}
 print('option differences',diffs.keys(),flush=True)
 torch.manual_seed(42);om=B0(12,(1,2,3,4),64,scale=old_scales,**oo)
 torch.manual_seed(42);nm=Model(horizons=scenario.horizons,scale=new_sc.tolist(),**options)
 assert om.state_dict().keys()==nm.state_dict().keys()
 params=max((a-b).abs().max().item() for a,b in zip(om.state_dict().values(),nm.state_dict().values()))
 values,available,final,y,mask,cal,cov=to_tensors(new.train,'cpu');ch=[0,3]
 old_sup=supervision(old,ch);wx=torch.tensor(old_run.loss_cell_weights(old_sup,LOSS_WEIGHTS['objective']))
 restricted=[]
 for e in new.train:
  yy=e['Y'].copy();yy[:,[1,2,4,5]]=0;restricted.append(dict(e,Y=yy))
 nw=torch.tensor(loss_cell_weights(restricted,LOSS_WEIGHTS['objective']))
 assert torch.equal(wx,nw)
 ids=torch.arange(min(8,len(old)));z=torch.randn(128,len(ids),16)
 x,oy,ocal=old_run.tensors(old_sup,om,'cpu')
 a=om(x[ids],ocal[ids],z=z,locations=old[0]['locations'])
 b=nm(values=values[ids],available=available[ids],known_final=final[ids],calendar=cal[ids],z=z,locations=new.train[0]['locations'])
 ofair=old_run.fair_crps_cells(a,oy[ids,:,:,0],oy[ids,:,:,1]);nfair=fair_crps_cells(b,y[ids],mask[ids])
 ol=(wx[ids]*ofair/om.scale).sum()*len(old)/len(ids)
 nl=(nw[ids][:,:,ch]*nfair[:,:,ch]/nm.scale[ch]).sum()*len(old)/len(ids)
 ol.backward();nl.backward()
 grad=max((a.grad-b.grad).abs().max().item() for a,b in zip(om.parameters(),nm.parameters()) if a.grad is not None)
 nm.zero_grad()
 bfull=nm(values=values[ids],available=available[ids],known_final=final[ids],calendar=cal[ids],z=z,locations=new.train[0]['locations'])
 fullmask=mask[ids].clone();fullmask[:,:,[1,2,4,5]]=False
 full_y=torch.stack((torch.where(fullmask,y[ids],0),fullmask.to(y.dtype)),dim=3)
 fullscore=fair_crps_cells(bfull,full_y[:,:,:,0,:],full_y[:,:,:,1,:])
 full_loss=(nw[ids]*fullscore/nm.scale).sum()*len(old)/len(ids)
 full_loss.backward()
 fullgrad=max((a.grad-b.grad).abs().max().item() for a,b in zip(om.parameters(),nm.parameters()) if a.grad is not None)
 per_parameter={name:float((p.grad-dict(nm.named_parameters())[name].grad).abs().max()) for name,p in om.named_parameters() if p.grad is not None}
 records.append(dict(held_out=held,per_parameter_gradient=per_parameter,full_loss=full_loss.item(),full_gradient_max=fullgrad,scale_max=float(abs(old_sc-new_sc).max()),option_differences=diffs,initial_parameter_max=params,output_max=(a-b).abs().max().item(),loss_old=ol.item(),loss_new_reweighted=nl.item(),gradient_max=grad))
 print({k:v for k,v in records[-1].items() if k not in ('option_differences','per_parameter_gradient')},flush=True)
 print({k:v for k,v in per_parameter.items() if v},flush=True)
Path('docs/results/b1-to-b0-chain').mkdir(parents=True,exist_ok=True)
Path('docs/results/b1-to-b0-chain/endpoint_diagnostic.json').write_text(json.dumps(records,indent=2)+'\n')
