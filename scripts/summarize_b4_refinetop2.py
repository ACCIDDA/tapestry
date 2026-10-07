"""Labeled tables for B4.refineTop2 items 3-5 from the pulled rank_pilot outputs."""
import json, pandas as pd
D = 'docs/experiments/b4-refinetop2-20261006/'
design = pd.DataFrame(json.load(open(D + 'design.json'))['rows']).rename(columns={'scenario': 'config_id'})
design['roles'] = design.roles.map(','.join)
r = pd.read_csv(D + 'analysis/pilot-rankings.csv')
assert (r['count'] == 2).all()
r = r[r.season.isin(['equal_season_mean', 'available_season_mean'])].merge(design, on='config_id', validate='many_to_one')
t = r.pivot_table(index=['recipe', 'treatment', 'sum_wis_weight', 'roles', 'history'], columns='metric', values='mean').reset_index()
x = pd.read_csv(D + 'analysis/distribution-scores.csv').merge(design, on='config_id')
cols = ['weekly_coverage_50', 'weekly_coverage_90', 'weekly_coverage_95']
x['us'] = x.location.eq('US')
g = x[x.target == 'flu_admissions'].groupby(['recipe', 'treatment', 'sum_wis_weight', 'history', 'seed', 'season', 'us'])[cols].mean().reset_index()
g[cols] = g[cols].mul(g.us.map({True: .2, False: .8}), axis=0)
c = g.groupby(['recipe', 'treatment', 'sum_wis_weight', 'history', 'seed', 'season'])[cols].sum().groupby(['recipe', 'treatment', 'sum_wis_weight', 'history']).mean().reset_index()
s = x[(x.target == 'flu_admissions') & x.four_week_sum_wis.notna()].groupby(['recipe', 'treatment', 'sum_wis_weight', 'history', 'season', 'us']).four_week_sum_wis.mean().unstack(['season', 'us'])
s.columns = [f'total_wis_{a}_{"US" if b else "states"}' for a, b in s.columns]
t = t.merge(c, on=['recipe', 'treatment', 'sum_wis_weight', 'history'], how='left').merge(s.reset_index(), on=['recipe', 'treatment', 'sum_wis_weight', 'history'], how='left')
t.to_csv(D + 'analysis/labeled-summary.csv', index=False)
show = ['recipe', 'treatment', 'sum_wis_weight', 'flu_native', 'flu_admissions_native', 'flu_admissions_log', 'flu_ed_native', 'weekly_coverage_90', 'weekly_coverage_95']
pd.set_option('display.width', 250)
for item, q in [('ITEM 4', "roles.str.contains('4_treatment')"), ('ITEM 5', "roles.str.contains('5_sum_wis')")]:
    for h in ['corrected', 'calibrated']:
        print(f'\n{item} {h}'); print(t[(t.history == h)].query(q)[show].sort_values(['recipe', 'flu_native']).round(4).to_string(index=False))
print('\nITEM 5 four-week total WIS, corrected'); print(t[(t.history == 'corrected') & t.roles.str.contains('5_sum')][['recipe', 'sum_wis_weight'] + [c for c in t if c.startswith('total_wis')]].round(1).to_string(index=False))
cal = pd.read_csv(D + 'analysis/calibration-factors.csv')
print('\nCalibration factors (median over runs, by target channel and horizon; 0=admissions, 3=ED):')
print(cal.groupby(['channel', 'horizon']).factor.describe()[['min', '50%', 'max']].round(2).to_string())

# Figure: combined native score per configuration, both seeds, corrected (filled) and calibrated (open).
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, numpy as np
s = pd.read_csv(D + 'analysis/pilot-seed-scores.csv')
s = s[(s.metric == 'flu_native') & (s.season == 'equal_season_mean')].merge(design, on='config_id')
s['label'] = np.where(s.roles.str.contains('4_treatment') & ~s.roles.str.contains('5_sum'), s.recipe + ' x ' + s.treatment,
                      np.where(s.roles.str.contains('5_sum') & ~s.roles.str.contains('4_treat'), s.recipe + ' total-WIS ' + s.sum_wis_weight.astype(str),
                               s.recipe + ' x ' + s.treatment + ' (= total-WIS 0)'))
order = s[s.history == 'corrected'].groupby('label').score.mean().sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(9, 8))
for i, lab in enumerate(order):
    for h, style in (('corrected', dict(color='#2a6fb0')), ('calibrated', dict(mfc='none', color='#c4581d'))):
        v = s[(s.label == lab) & (s.history == h)].score
        ax.plot(v, [i + (0.15 if h == 'calibrated' else 0)] * len(v), 'o', ms=6, **style, label=h if i == 0 else None)
ax.set_yticks(range(len(order)), order, fontsize=8); ax.axvline(1, color='#888', lw=1)
ax.set_xlabel('Combined native relative WIS (admissions + 0.5 ED; 1 = Hub ensemble; lower is better)')
ax.set_title('B4.refineTop2: each dot is one seed (42 or 43). Trained on 3 of 2022-26 seasons with finalized flu labels;\n'
             'evaluated on 2024-25 and 2025-26 Wednesday reports, newest 2 admission weeks corrected (blue) and then calibrated (orange)', fontsize=9)
ax.legend(loc='lower right'); ax.grid(axis='x', color='#ddd', lw=.5); ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout(); fig.savefig(D + 'analysis/configurations.png', dpi=130)
