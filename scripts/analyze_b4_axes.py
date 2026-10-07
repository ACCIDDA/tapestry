"""Item 1 of B4.refineTop2: which randomized settings of the B4 600-configuration flu sweep mattered.

Uses the saved two-seed mean scores of the 440 existing-MLP configurations (the 160 shared
encoders have a different design). Main-effects least squares on log score, one categorical
factor per randomized setting; interactions are ignored. No training, no rescoring."""
import numpy as np, pandas as pd
from dataclasses import asdict
from pathlib import Path
from tapestry.model.scenario import Scenario

SRC = Path('docs/experiments/b4-flu-600-20261005/final-analysis/pilot-rankings.csv')
OUT = Path('docs/experiments/b4-refinetop2-20261006/axis-analysis'); OUT.mkdir(parents=True, exist_ok=True)
AXES = ['pilot_method', 'pilot_nowcaster', 'pathogen_inputs', 'fit_partition', 'decoder', 'covariate_set', 'spatial',
        'width', 'lookback', 'lr', 'weight_decay', 'count_transform', 'epochs', 'reporting_strength',
        'correction_weeks', 'correction_strength', 'correction_penalty']
METRICS = {'flu_native': 'equal_season_mean', 'flu_admissions_log': 'equal_season_mean', 'flu_ed_native': 'available_season_mean'}

d = pd.read_csv(SRC)
d = d[(d['count'] == 2) & (d.history == 'corrected') & d.apply(lambda r: METRICS.get(r.metric) == r.season, axis=1)]
wide = d.pivot(index='config_id', columns='metric', values='mean').reset_index()
fields = pd.DataFrame([asdict(Scenario.from_string(c)) for c in wide.config_id])
t = pd.concat([wide, fields[AXES + ['encoder', 'revision_scope']]], axis=1)
t = t[t.encoder == 'mlp'].copy()
t.loc[t.revision_scope == 'early_actual', 'pilot_method'] = 'errors_early_actual'
t['decoder'] = t.decoder.replace({'legacy': 'samples', 'quantile': 'quantiles'})
t['covariate_set'] = t.covariate_set.replace({'': 'none'})
assert len(t) == 440, len(t)
t.to_csv(OUT / 'mlp-configurations.csv', index=False)

rows = []
for metric in METRICS:
    y = np.log(t[metric].to_numpy())
    X = [np.ones(len(t))]; names = [('intercept', '')]
    refs = {}
    for a in AXES:
        levels = sorted(t[a].astype(str).unique(), key=lambda v: (len(v), v))
        counts = t[a].astype(str).value_counts()
        refs[a] = counts.idxmax()  # most common level is the reference
        for v in levels:
            if v != refs[a]:
                X.append((t[a].astype(str) == v).to_numpy(float)); names.append((a, v))
    X = np.column_stack(X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    bread = np.linalg.pinv(X.T @ X)
    cov = bread @ (X.T * resid ** 2) @ X @ bread  # HC0 robust covariance
    se = np.sqrt(np.diag(cov))
    r2 = 1 - resid.var() / y.var()
    for (a, v), b, s in zip(names, beta, se):
        if a == 'intercept': continue
        sub = t[t[a].astype(str) == v]
        rows.append(dict(metric=metric, axis=a, level=v, reference=refs[a], configs=len(sub),
                         effect_pct=100 * (np.exp(b) - 1), low_pct=100 * (np.exp(b - 1.96 * s) - 1),
                         high_pct=100 * (np.exp(b + 1.96 * s) - 1), median_score=sub[metric].median(),
                         reference_median=t[t[a].astype(str) == refs[a]][metric].median(), r2=r2))
    for a in AXES:
        rows.append(dict(metric=metric, axis=a, level=refs[a], reference=refs[a], configs=int((t[a].astype(str) == refs[a]).sum()),
                         effect_pct=0., low_pct=np.nan, high_pct=np.nan,
                         median_score=t[t[a].astype(str) == refs[a]][metric].median(), reference_median=np.nan, r2=r2))
effects = pd.DataFrame(rows).sort_values(['metric', 'axis', 'level'])
effects.to_csv(OUT / 'axis-effects.csv', index=False)

# Top-20 composition: which levels the best 20 MLP configurations use.
top = t.nsmallest(20, 'flu_native')
top.to_csv(OUT / 'top20-native.csv', index=False)
pd.set_option('display.width', 220)
for metric in METRICS:
    e = effects[effects.metric == metric]
    print(f'\n== {metric}  (R2 {e.r2.iloc[0]:.2f}); effect = % change in score vs reference level, other settings held fixed; negative = better')
    print(e[['axis', 'level', 'reference', 'configs', 'effect_pct', 'low_pct', 'high_pct', 'median_score']].round(2).to_string(index=False))
print('\nTop-20 native composite: level counts')
for a in AXES:
    print(a, dict(top[a].astype(str).value_counts()))

# Figure: one row per non-reference level, effect with 95% robust interval, three metrics side by side.
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
labels = {'flu_native': 'Combined native score (admissions + 0.5 ED)', 'flu_admissions_log': 'Log admissions',
          'flu_ed_native': 'Forward-season ED (2025-26 only)'}
e = effects[effects.level != effects.reference]
order = e[e.metric == 'flu_native'][['axis', 'level', 'reference']].drop_duplicates().reset_index(drop=True)
fig, axes = plt.subplots(1, 3, figsize=(13, 11), sharey=True)
for ax, metric in zip(axes, METRICS):
    m = order.merge(e[e.metric == metric], on=['axis', 'level', 'reference'])
    y = np.arange(len(m))[::-1]
    ax.axvline(0, color='#888', lw=1)
    ax.hlines(y, m.low_pct, m.high_pct, color='#2a6fb0', lw=2)
    ax.plot(m.effect_pct, y, 'o', color='#2a6fb0', ms=5)
    ax.set_title(labels[metric], fontsize=10); ax.set_xlabel('% change in relative WIS vs reference level\n(negative = better)')
    ax.grid(axis='x', color='#ddd', lw=.5); ax.spines[['top', 'right']].set_visible(False)
axes[0].set_yticks(np.arange(len(order))[::-1], [f'{a}: {l}  (ref {r})' for a, l, r in order.itertuples(index=False)], fontsize=8)
fig.suptitle('B4 600-configuration flu sweep, 440 existing-MLP configurations: main effect of each randomized setting\n'
             'Models trained on 2022-25 seasons minus the evaluated season, finalized flu labels; evaluated on 2024-25 and 2025-26 '
             'Wednesday reports with 2 newest admission weeks corrected; two-seed means', fontsize=10)
fig.tight_layout(); fig.savefig(OUT / 'axis-effects.png', dpi=130)
