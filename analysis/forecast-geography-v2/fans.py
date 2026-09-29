"""Inline report fans for the top three configurations, using fixed seed 42."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from tapestry.dataset.build import load
from tapestry.evaluation.plots import fans

root=Path('data/experiments/forecast-geography-v2')
r=root/'ranking-e53fd92c4f32'
rank=pd.read_csv(r/'configuration_ranking.csv').sort_values('combined_mean')
manifest=json.loads((r/'manifest.json').read_text())
configs=rank.head(3).config_id.tolist()
titles=['Independent: mixed summaries','Independent: new flu smoothed','Independent: Kinsa summaries']
runs=[dict(run,config_id=titles[configs.index(run['config_id'])]) for run in manifest['runs'] if run['config_id'] in configs and run['seed']==42]
assert len(runs)==3
out=Path('docs/results/forecast-geography-v2/best-fans');out.mkdir(parents=True,exist_ok=True)
for path in fans(load('data/processed/panel.npz'),runs,titles,{c:f'M{i+1}' for i,c in enumerate(titles)},
                 'data/evaluation/b0_hub_comparison_q23',None,out):print(path)
(out/'selection.json').write_text(json.dumps(dict(seed=42,configurations=dict(zip(titles,configs)),
    selection='Top three three-seed mean scores; fans use seed 42, not the best seed.',
    intervals=[50,90],reference_spacing_weeks=4),indent=2)+'\n')
