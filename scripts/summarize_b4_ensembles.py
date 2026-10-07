"""Readable tables for the B4.refineTop2 ensembles (item 2), from the pulled rank_pilot outputs."""
import pandas as pd
import sys
D = sys.argv[1] if len(sys.argv) > 1 else 'docs/experiments/b4-refinetop2-20261006/ensembles/'
r = pd.read_csv(D + 'pilot-rankings.csv')
r = r[r.season.isin(['equal_season_mean', 'available_season_mean'])]
r['name'] = r.config_id.str.extract(r'ensemble=([^,]+),rule=([^,]+)').agg('-'.join, axis=1)
t = r.pivot_table(index=['name', 'history'], columns='metric', values='mean')[['flu_native', 'flu_admissions_native', 'flu_admissions_log', 'flu_ed_native']]
x = pd.read_csv(D + 'distribution-scores.csv')
x['name'] = x.config_id.str.extract(r'ensemble=([^,]+),rule=([^,]+)').agg('-'.join, axis=1)
cols = ['weekly_coverage_50', 'weekly_coverage_90', 'weekly_coverage_95']
g = x[x.target == 'flu_admissions'].assign(us=x.location.eq('US')).groupby(['name', 'history', 'season', 'us'])[cols].mean().reset_index()
g[cols] = g[cols].mul(g.us.map({True: .2, False: .8}), axis=0)
c = g.groupby(['name', 'history', 'season'])[cols].sum().groupby(['name', 'history']).mean()
out = t.join(c).reset_index().sort_values(['history', 'flu_native'])
out.to_csv(D + 'summary.csv', index=False)
print(out[out.history == 'corrected'].drop(columns='history').round(4).to_string(index=False))
