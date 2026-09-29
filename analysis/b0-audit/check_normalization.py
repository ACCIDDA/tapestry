"""Verify the intervention against B0's fitted-context normalization."""
import ast,importlib.util,json
from pathlib import Path
import numpy as np
from tapestry.dataset.build import load
from tapestry.dataset.cv import fold
from tapestry.model.scenario import Scenario
from tapestry.experiment.training import populations
from tapestry.model.network import transform_counts,transform_proportions

path=Path('data/experiments/b0-normalization-audit/code/src/tapestry/experiment/training.py')
spec=importlib.util.spec_from_file_location('tapestry.experiment.audit_training',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
src=Path('data/audits/b0/old-src/tapestry/models/experiments.py').read_text()
node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='input_scales')
code='\n'.join(src.splitlines()[node.lineno-1:node.end_lineno]).replace('from .b0 import transform_counts, transform_proportions','from tapestry.model.network import transform_counts, transform_proportions')
namespace={'np':np};exec(code,namespace)
panel=load('data/processed/panel.npz')
eps=fold(panel,Scenario(ed_transform='logit',patience=30),'2025-2026',inner=True).train
pop=populations('data/metadata/locations.csv',eps[0]['locations'])
legacy=[dict(e,X=np.stack((e['values'],e['available']),axis=2)) for e in eps]
a=namespace['input_scales'](legacy,'fourth_root','logit',pop)
b=module.model_options(eps,Scenario(ed_transform='logit'),pop)
for k in ('input_scale','input_offset'):np.testing.assert_allclose(a[k],b[k],rtol=0,atol=0)
# Masked numerical placeholders must not influence scaling.
changed=[]
for e in eps:
 v=e['values'].copy();v[~e['available']]=1e12;changed.append(dict(e,values=v))
c=module.model_options(changed,Scenario(ed_transform='logit'),pop)
for k in ('input_scale','input_offset'):np.testing.assert_allclose(b[k],c[k],rtol=0,atol=0)
summary=dict(finalized_matches_b0_exactly=True,masked_placeholders_ignored=True,
             scales_range=[float(np.min(b['input_scale'])),float(np.max(b['input_scale']))],
             ed_offsets_range=[float(np.min(np.array(b['input_offset'])[3:])),float(np.max(np.array(b['input_offset'])[3:]))])
Path('data/experiments/b0-normalization-audit/normalization-checks.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
