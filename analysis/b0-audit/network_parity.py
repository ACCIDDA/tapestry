"""Check old/current network equivalence with identical fitted options, weights and draws."""
import importlib,sys,types,json
from pathlib import Path
import numpy as np
import torch
from tapestry.model.network import Model
from tapestry.dataset.episodes import calendar
package=types.ModuleType('historical_models');package.__path__=[str(Path('data/audits/b0/old-src/tapestry/models').resolve())]
sys.modules['historical_models']=package
B0=importlib.import_module('historical_models.b0').B0
m=json.loads(Path('data/audits/b0/b0-pathogen300/s42/manifest.json').read_text())
config=m['folds'][0]['components'][0]['experiment']
torch.set_num_threads(1)
torch.manual_seed(42);old=B0(**config)
torch.manual_seed(42);new=Model(**config)
assert old.state_dict().keys()==new.state_dict().keys(), (old.state_dict().keys(),new.state_dict().keys())
parameter_error=max(float((a-b).abs().max()) for a,b in zip(old.state_dict().values(),new.state_dict().values()))
new.load_state_dict(old.state_dict())
with np.load('data/processed/build_b_finalized.npz') as d:
 x=torch.tensor(d['panel'][50:62][None]);locations=d['locations'].tolist()
cal=torch.tensor(calendar(['2024-11-09']))
z=torch.randn(8,1,16)
with torch.no_grad():
 a=old(x,cal,z=z,locations=locations);b=new(x,cal,z=z,locations=locations)
error=float((a-b).abs().max());relative=float(((a-b).abs()/(1+a.abs())).max())
result=dict(initial_parameter_max_difference=parameter_error,output_max_abs_difference=error,output_max_relative_difference=relative,same_parameter_names=True)
print(json.dumps(result,indent=2))
Path('docs/results/b0-audit/network_parity.json').write_text(json.dumps(result,indent=2)+'\n')
