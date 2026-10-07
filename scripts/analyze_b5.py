"""B5 exploration analysis: option effects by regression, within-run stochastic nowcast, top configurations.

Input: the pulled rank_pilot outputs (pilot-seed-scores.csv) of b5-explore-20261006 and its design.json.
Only configurations with both seeds are used; scores are two-seed means of the corrected view (and of
the sampled-history view where it exists). Effects are % change in relative WIS from the anchor's own
level, estimated by least squares on log score with anchor fixed effects and all factors as categorical
main effects (HC0 robust intervals); interactions with the anchor are reported separately per anchor.
"""
import json, sys
from dataclasses import asdict
from pathlib import Path
import numpy as np, pandas as pd
from tapestry.model.scenario import Scenario

D = Path(sys.argv[1] if len(sys.argv) > 1 else 'docs/experiments/b5-explore-20261006')
OUT = D / 'analysis'; OUT.mkdir(exist_ok=True)
FACTORS = ['vintage_seasons', 'actual_share', 'reporting_strength', 'reporting_random_strength', 'reporting_method',
           'reporting_missingness', 'revision_signals', 'correction_realizations', 'uncorrected_share', 'nowcast_noise',
           'nowcast_noise_train', 'correction_penalty', 'correction_weeks', 'correction_strength', 'pilot_nowcaster',
           'growth_anchor', 'log_loss_weight', 'covariate_dropout', 'ili_training', 'sum_wis_weight', 'latent', 'batch_size',
           'lr_ratio', 'weight_decay', 'signal_features', 'spatial', 'lookback', 'width', 'noise', 'us_error', 'members', 'decoder']
METRICS = {'flu_native': 'equal_season_mean', 'flu_admissions_log': 'equal_season_mean', 'flu_admissions_native': 'equal_season_mean',
           'flu_ed_native': 'available_season_mean'}

design = pd.DataFrame(json.loads((D / 'design.json').read_text())['rows']).rename(columns={'scenario': 'config_id'})
anchor_lr = {r.anchor: Scenario.from_string(r.config_id).lr for r in design[design.kind == 'anchor'].itertuples()}
anchor_values = {r.anchor: asdict(Scenario.from_string(r.config_id)) for r in design[design.kind == 'anchor'].itertuples()}
s = pd.read_csv(D / 'analysis-raw' / 'pilot-seed-scores.csv')
s = s[s.apply(lambda r: METRICS.get(r.metric) == r.season, axis=1)]
two = s[s.history == 'corrected'].groupby('config_id').seed.nunique()
s = s[s.config_id.isin(two[two == 2].index)]
wide = s.pivot_table(index=['config_id', 'history'], columns='metric', values='score').reset_index()
fields = pd.DataFrame([asdict(Scenario.from_string(c)) for c in design.config_id]).assign(config_id=design.config_id.values)
t = design[['config_id', 'anchor', 'kind']].merge(fields, on='config_id')
t['lr_ratio'] = (t.lr / t.anchor.map(anchor_lr)).round(2)
corrected = wide[wide.history == 'corrected'].merge(t, on='config_id')
corrected.to_csv(OUT / 'configurations.csv', index=False)
print(f'{len(corrected)} configurations with both seeds (A {sum(corrected.anchor == "A")}, B {sum(corrected.anchor == "B")}, Bs {sum(corrected.anchor == "Bs")})')

pd.set_option('display.width', 250)
print('\nAnchors (exact B4 leaders refitted, two-seed means):')
print(corrected[corrected.kind == 'anchor'][['anchor', *METRICS]].round(4).to_string(index=False))


def effects(frame, metric, by_anchor):
    rows = []
    y = np.log(frame[metric].to_numpy())
    X, names = [np.ones(len(frame))], [('intercept', '', '')]
    if by_anchor:
        for a in ('B', 'Bs'):
            X.append((frame.anchor == a).to_numpy(float)); names.append(('anchor', a, 'A'))
    for f in FACTORS:
        values = frame[f].astype(str)
        if values.nunique() < 2:
            continue
        ref = values.value_counts().idxmax()
        for v in sorted(values.unique()):
            if v != ref:
                X.append((values == v).to_numpy(float)); names.append((f, v, ref))
    X = np.column_stack(X)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    bread = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(bread @ (X.T * resid ** 2) @ X @ bread))
    for (f, v, ref), b, e in zip(names, beta, se):
        if f in ('intercept',):
            continue
        n = int(((frame[f].astype(str) == v) if f != 'anchor' else (frame.anchor == v)).sum())
        rows.append(dict(metric=metric, factor=f, level=v, reference=ref, configs=n, effect_pct=100 * (np.exp(b) - 1),
                         low_pct=100 * (np.exp(b - 1.96 * e) - 1), high_pct=100 * (np.exp(b + 1.96 * e) - 1)))
    return pd.DataFrame(rows), 1 - resid.var() / y.var()


tables = []
for metric in ('flu_native', 'flu_admissions_log'):
    e, r2 = effects(corrected, metric, True)
    tables.append(e.assign(scope='pooled', r2=r2))
    for a in ('A', 'B', 'Bs'):
        sub = corrected[corrected.anchor == a]
        if len(sub) > 40:
            e, r2 = effects(sub, metric, False)
            tables.append(e.assign(scope=a, r2=r2))
eff = pd.concat(tables)
eff.to_csv(OUT / 'effects.csv', index=False)
for scope in ('pooled', 'A', 'B', 'Bs'):
    e = eff[(eff.scope == scope) & (eff.metric == 'flu_native')]
    if e.empty:
        continue
    print(f'\n== {scope}: combined native, % change vs reference (negative = better), R2 {e.r2.iloc[0]:.2f}')
    clear = e[(e.high_pct < 0) | (e.low_pct > 0)]
    print(clear[['factor', 'level', 'reference', 'configs', 'effect_pct', 'low_pct', 'high_pct']].round(1).sort_values('effect_pct').to_string(index=False))

# Stochastic nowcasting inside the same fitted model: sampled-history view vs point-corrected view.
pair = wide[wide.history.isin(['corrected', 'sampled'])].pivot_table(index='config_id', columns='history', values='flu_native').dropna()
pair = pair.join(t.set_index('config_id')[['anchor', 'nowcast_noise', 'nowcast_noise_train']])
pair['change_pct'] = 100 * (pair.sampled / pair.corrected - 1)
pair.to_csv(OUT / 'sampled-vs-corrected.csv')
if len(pair):
    print('\nSampled histories vs point correction, same fitted model (combined native, % change; negative = better):')
    print(pair.groupby(['anchor', 'nowcast_noise', 'nowcast_noise_train']).change_pct.describe()[['count', 'mean', '25%', '50%', '75%']].round(2).to_string())

print('\nBest 10 configurations per anchor (two-seed mean, corrected view):')
for a in ('A', 'B', 'Bs'):
    sub = corrected[corrected.anchor == a].nsmallest(10, 'flu_native')
    anchor = anchor_values[a]
    for r in sub.itertuples():
        changed = {f: getattr(r, f) for f in FACTORS if f in anchor and getattr(r, f) != anchor[f]}
        print(a, round(r.flu_native, 4), round(r.flu_admissions_log, 4), changed)
