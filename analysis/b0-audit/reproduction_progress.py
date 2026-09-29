"""Compact progress plus an early deterministic comparison to historical training."""
import json,sys
from pathlib import Path
import numpy as np
root=Path('data/experiments')/(sys.argv[1] if len(sys.argv)>1 else 'b0-exact-reproduction')
selected=json.loads(Path('data/audits/b0/selected.json').read_text())
for item in selected:
    label,seed=item['label'],item['seed']
    prefix='mlp-target-' if label=='b0-target100' else 'mlp-pathogen-'
    scenario=next(root.glob(prefix+'*'))
    attempts=sorted((scenario/f's{seed}').glob('attempt-*'))
    if not attempts:continue
    run=attempts[-1]
    records=[]
    for line in (run/'run.log').read_text().splitlines():
        try:row=json.loads(line)
        except ValueError:continue
        if 'epoch' in row:records.append(row)
    if not records:print(label,seed,'no epochs');continue
    first=[]
    for row in records:
        if first and row['epoch']<=first[-1]['epoch']:break
        first.append(row)
    original=json.loads((Path(item['path'])/'manifest.json').read_text())
    expected=original['folds'][0]['components'][0]['early_stopping']
    n=min(len(first),len(expected['loss']))
    delta=max(abs(first[i]['loss']-expected['loss'][i]) for i in range(n))
    val_delta=max(abs(first[i]['validation_loss']-expected['validation_loss'][i]) for i in range(n))
    folds=list((run/'legacy').glob('eval_*/forecasts.npz'))
    completed_manifest=json.loads((run/'legacy/manifest.json').read_text())
    print(label,seed,'folds',len(folds),'log_epochs',len(records),'last',records[-1]['epoch'],
          'first_component_epochs_checked',n,'loss_max_diff',delta,'validation_max_diff',val_delta,
          'completed_selected_epochs',[f['epochs'] for f in completed_manifest['folds']],flush=True)
