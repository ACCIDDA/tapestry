"""Same-formulation/seed fans for the original mixed-summary leader."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from tapestry.dataset.build import load
from tapestry.evaluation.plots import fans
base=Path('docs/results/forecast-geography-v2/no-mask')
s=pd.read_csv(base/'matched_configurations.csv')
row=s[(s.name=='none / Mixed / summary')&(s.geography=='all')].iloc[0]
runs=[]
for folder,config,title in [('forecast-geography-v2/ranking-e53fd92c4f32',row.masked_config,'Mixed summaries: 50% masking'),
                           ('forecast-no-mask-top32-v3/ranking-981191c4dc56',row.config_id,'Mixed summaries: no masking')]:
    manifest=json.loads((Path('data/experiments')/folder/'manifest.json').read_text())
    run=next(r for r in manifest['runs'] if r['config_id']==config and r['seed']==42)
    runs.append(dict(run,config_id=title))
titles=[r['config_id'] for r in runs]
for path in fans(load('data/processed/panel.npz'),runs,titles,{t:f'M{i+1}' for i,t in enumerate(titles)},
                  'data/evaluation/b0_hub_comparison_q23',None,base):print(path)
