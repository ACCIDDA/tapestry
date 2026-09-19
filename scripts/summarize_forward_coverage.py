"""Compare saved same-recipe, same-target coverage across seasons; no fitting."""
import json
from pathlib import Path

import pandas as pd

root = Path('docs/results/b1-overnight/ranking')
manifest = json.loads((root/'manifest.json').read_text())
names = {r['config_id']:r['name'] for r in manifest['runs']}
seasons = pd.read_csv(root/'season_scores.csv')
seasons['name'] = seasons.config_id.map(names)
selected = seasons[(seasons.name == 'target_mlp__direct_finalflag__mask0') &
                   (seasons.geography == 'all') & seasons.seed.isin([42,43,44]) &
                   seasons.target.isin(['wk inc flu hosp', 'wk inc covid hosp'])]
metrics = ['model_coverage_50','model_coverage_95','ensemble_coverage_95','wis_ratio']
summary = selected.groupby(['target','season'])[metrics].mean().reset_index()
summary.to_csv('docs/results/Forward-2025/coverage-across-seasons.csv',index=False)
print(summary.to_string(index=False))
