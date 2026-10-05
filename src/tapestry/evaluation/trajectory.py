"""Matched recent-level and growth diagnostics for point nowcasters."""
import numpy as np
import pandas as pd


def trajectory_cells(frame):
    """One four-week trajectory per issuance/target/location, chronological ages 3..0.

    All methods share support. Q95 from training provides common native-unit
    normalization. Growth is the recent two-week mean minus the previous two.
    Log growth uses a training-scale 1% offset and is secondary, never MAPE.
    """
    keys = ['signal', 'issuance', 'location']
    methods = [m for m in ('prediction', 'persistence', 'baselinenowcast') if m in frame]
    f = frame[frame.age.between(0, 3)]
    table = f.pivot(index=keys, columns='age', values=['truth', 'scale', *methods])
    columns = pd.MultiIndex.from_product([['truth', 'scale', *methods], range(4)])
    table = table.reindex(columns=columns).dropna()
    base = table.index.to_frame(index=False)
    flags = frame[frame.age.eq(0)].set_index(keys)
    for flag in ('complete_history_12', 'uninterrupted_8'):
        if flag in flags:
            base[flag] = flags.loc[table.index, flag].to_numpy(bool)
    base['geography'] = np.where(base.location.eq('US'), 'US', 'states')
    truth = table.truth.to_numpy()
    scale = table.scale[0].to_numpy()
    if (scale <= 0).any():
        raise ValueError('Trajectory scoring requires positive training scales')
    def growth(a):
        return a[:, :2].mean(1)-a[:, 2:].mean(1)
    def loggrowth(a):
        return np.log((a[:, :2].mean(1)+.01*scale)/(a[:, 2:].mean(1)+.01*scale))/2
    records = []
    for method in methods:
        values = table[method].to_numpy()
        g = base.copy()
        g['method'] = method
        g['point'] = abs(values[:, 0]-truth[:, 0])/scale
        g['level'] = abs(values.mean(1)-truth.mean(1))/scale
        g['growth'] = abs(growth(values)-growth(truth))/scale
        g['log_growth'] = abs(loggrowth(values)-loggrowth(truth))
        g['trajectory'] = g[['point', 'level', 'growth']].mean(axis=1)
        records.append(g)
    return pd.concat(records, ignore_index=True)


def trajectory_metrics(cells):
    metrics = ['point', 'level', 'growth', 'log_growth', 'trajectory']
    strata = {'all': np.ones(len(cells), bool)}
    if 'complete_history_12' in cells:
        strata['complete_history_12'] = cells.complete_history_12
        strata['complete_and_uninterrupted'] = cells.complete_history_12 & cells.uninterrupted_8
    results = []
    keys = ['signal', 'geography', 'method']
    for stratum, keep in strata.items():
        g = cells[keep]
        scored = g.groupby(keys+['location'])[metrics].mean().groupby(keys).mean()
        coverage = g.groupby(keys).agg(cells=('issuance', 'size'), weeks=('issuance', 'nunique'), locations=('location', 'nunique'))
        results.append(scored.join(coverage).reset_index().assign(stratum=stratum))
    return pd.concat(results, ignore_index=True)
