"""Copy selected historical forecasts without changing quantiles or original artifacts."""
import json
from pathlib import Path
import shutil
import numpy as np
root = Path('data/experiments/B0.1')
out = Path('data/audits/b0')
manifest = json.loads((root/'ranking-509b07b0d243/manifest.json').read_text())
names = {'B0.1/independent__encoder_mlp__fit_partition_target__epochs_100': 'b0-target100',
         'B0.1/independent__encoder_mlp__fit_partition_pathogen__epochs_300': 'b0-pathogen300'}
selected = []
for run in manifest['runs']:
    if run['name'] not in names:
        continue
    dest = out/names[run['name']]/f"s{run['seed']}"
    src = Path(run['path'])
    dest.mkdir(parents=True, exist_ok=True)
    for f in src.glob('*.json'):
        shutil.copy2(f, dest/f.name)
    shutil.copy2(src/'totals.csv', dest/'historical_totals.csv')
    for fold in src.glob('eval_*'):
        (dest/fold.name).mkdir(exist_ok=True)
        with np.load(fold/'forecasts.npz', allow_pickle=False) as data:
            np.savez_compressed(dest/fold.name/'forecasts.npz', **{k:data[k] for k in data.files if k != 'samples'})
        for f in fold.glob('*.json'):
            shutil.copy2(f, dest/fold.name/f.name)
    selected.append(dict(run, label=names[run['name']], extracted=str(dest)))
(out/'selected.json').write_text(json.dumps(selected, indent=2)+'\n')
print(json.dumps(selected, indent=2))
