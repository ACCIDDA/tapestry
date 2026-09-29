"""Finalize revised raw-CSV availability coverage before any stage00–03 attempt."""
from pathlib import Path
import json,hashlib
p=Path('data/audits/b0/original-panel-deadline2025.npz');digest=hashlib.sha256(p.read_bytes()).hexdigest()
for stage in ['00','01','02','03']:
 root=Path('data/experiments')/f'b1-b0-chain-{stage}'
 assert not list(root.glob('*/s*/attempt-*')),'Never change launched source/data pins'
 config=json.loads((root/'experiment.json').read_text());old=config['dataset_sha256']
 assert config['dataset']==str(p)
 config['dataset_sha256']=digest;(root/'experiment.json').write_text(json.dumps(config,indent=2)+'\n')
 (root/'prelaunch-data-correction.json').write_text(json.dumps(dict(old_sha256=old,new_sha256=digest,reason='Include historical NSSP latest.csv files, replaced by Parquet in November2025; no fit had started',common_earliest_deadline='Verified presence-mask-equivalent to individual Hub cutoffs'),indent=2)+'\n')
 print(stage,digest)
