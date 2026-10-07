"""B4 runs vs Google_SAI-FluEns: pooled mean WIS, flu admissions, states/DC, 2025-26.

Google's pooled WIS on the identical 5,712 tasks comes from ../google-comparison-20261005/per-seed.csv
(frozen_all scope; native and log1p_sensitivity). Run from docs/experiments.
"""
import pandas as pd

g = pd.read_csv('google-comparison-20261005/per-seed.csv')
g = g[(g.target == 'wk inc flu hosp') & (g.scope == 'frozen_all')]
G = {'natural': g[g.scale == 'native'].pooled_google_wis.iloc[0],
     'log': g[g.scale == 'log1p_sensitivity'].pooled_google_wis.iloc[0]}
SOURCES = {'600-config sweep': 'b4-flu-600-20261005/final-analysis',
           'covariate refine': 'b4-flu-refine-20261006/analysis',
           'output heads': 'b4-flu-heads-20261006/analysis',
           'refineTop2 retrains': 'b4-refinetop2-20261006/analysis',
           'refineTop2 ensembles': 'b4-refinetop2-20261006/ensembles'}
rows = []
for name, path in SOURCES.items():
    d = pd.read_csv(f'{path}/pilot-raw-wis.csv')
    d = d[(d.target == 'wk inc flu hosp') & (d.season == '2025-2026') & (d.geography == 'states_dc')]
    assert set(d.tasks) == {5712}
    for (scale, history), x in d.groupby(['scale', 'history']):
        m = x.groupby('config_id').mean_wis.mean() / G[scale]
        r = x.mean_wis / G[scale]
        rows.append(dict(experiment=name, scale=scale, history=history, configurations=len(m),
                         configurations_beating_google=int((m < 1).sum()),
                         seed_runs_beating_google=f'{int((r < 1).sum())}/{len(r)}',
                         best_ratio=round(m.min(), 3), median_ratio=round(m.median(), 3)))
out = pd.DataFrame(rows)
out.to_csv('b4-google-check-20261006/summary.csv', index=False)
print(out.to_string(index=False))
