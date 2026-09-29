"""Preserve B0's exact finalized panel, attaching current Wednesday availability only."""
import hashlib,json
from pathlib import Path
import numpy as np
from tapestry.dataset.build import load,save
old_path=Path('data/processed/build_b_finalized.npz')
old=np.load(old_path);panel=load('data/processed/panel.npz');n=len(old['dates'])
assert np.array_equal(panel['dates'][:n].astype(str),old['dates'])
assert np.array_equal(panel['locations'],old['locations'])
for key in list(panel):
 if key.startswith('asof_'):
  panel[key]=panel[key][:n,:n]
 elif key in ('dates','issuance_dates','targets','covariates','covariates_national'):
  panel[key]=panel[key][:n]
values=np.where(old['panel'][:,:,1].astype(bool),old['panel'][:,:,0],np.nan)
panel['targets']=values.transpose(0,2,1)
panel['metadata']=json.dumps(dict(purpose='B0 exact finalized values and calendar with current Wednesday availability',
    original_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest(),
    availability_source='data/processed/panel.npz',
    availability_sha256=hashlib.sha256(Path('data/processed/panel.npz').read_bytes()).hexdigest()))
out=Path('data/audits/b0/original-panel-unified.npz');save(panel,out)
r=load(out)
np.testing.assert_equal(r['targets'].transpose(0,2,1),values)
print(out,hashlib.sha256(out.read_bytes()).hexdigest())
