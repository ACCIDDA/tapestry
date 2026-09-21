"""Vintage-correct Wednesday-origin evaluation of Delphi covariates against NHSN.

Same protocol as analysis/wval/index/vintage_test.py, generalised so that any
Delphi source with a report_time archive can be swapped in: inpatient claims,
outpatient claims, PopHIVE ED, and wastewater for comparison.

Protocol
  origin r   a Wednesday NHSN report_time; the newest NHSN week is t = r - 4 days
  baseline   log NHSN admissions at t and t-1 AS KNOWN AT r, plus their growth
  covariate  the source's value at reference week t-k and its two-week change,
             using only rows with report_time <= r
  target     final NHSN admissions at t+h
  k          per source, the smallest lag at which >=80% of observations have
             arrived by the Wednesday origin (measured, not assumed)
"""
import glob
import numpy as np
import pandas as pd

ROOT = '/Users/chadi/Research/Tapestry/'
NHSN = ROOT + 'data/raw/delphi_nhsn/snapshots/*/signal=confirmed_admissions_{p}_ew/geo_type=state/archive.csv.gz'
MIN_OBS = 25


# ---------------------------------------------------------------- NHSN side

def nhsn_archive(pathogen):
    f = glob.glob(NHSN.format(p=pathogen))[0]
    d = pd.read_csv(f, parse_dates=['report_time', 'reference_time'])
    return d[['report_time', 'geo_value', 'reference_time', 'value']].dropna(subset=['value'])


def as_of_panel(d):
    """Value of every reference week as known at each report_time, carried forward."""
    out = []
    for geo, g in d.sort_values('report_time').groupby('geo_value'):
        piv = (g.groupby(['report_time', 'reference_time'])['value'].last()
                .unstack('reference_time').sort_index().ffill())
        s = piv.stack().rename('value').reset_index()
        s['geo_value'] = geo
        out.append(s)
    return pd.concat(out, ignore_index=True)


def final_series(d):
    return (d.sort_values('report_time')
             .groupby(['geo_value', 'reference_time'], as_index=False)['value'].last()
             .rename(columns={'value': 'y_final'}))


# ------------------------------------------------------------ covariate side

def load_source_chunked(path, since='2024-01-01', weekly_only=True, chunk=2_000_000):
    """Same as load_source but streams, for the multi-GB claims archives."""
    keep = []
    for c in pd.read_csv(path, parse_dates=['report_time', 'reference_time'],
                         chunksize=chunk):
        c = c[c['reference_time'] >= since]
        if weekly_only:
            c = c[c['reference_time'].dt.weekday == 5]
        c = c.dropna(subset=['value'])
        if len(c):
            c['report_time'] = c['report_time'].dt.normalize()
            keep.append(c[['report_time', 'geo_value', 'reference_time', 'value']])
    return pd.concat(keep, ignore_index=True) if keep else pd.DataFrame()


def load_source(path, age_all=False, weekly_only=True):
    d = pd.read_csv(path, parse_dates=['report_time', 'reference_time'])
    if age_all and 'age_group' in d.columns:
        d = d[d['age_group'] == 'all']
    d['report_time'] = d['report_time'].dt.normalize()
    if weekly_only:                      # daily sources -> the week ending Saturday
        d = d[d['reference_time'].dt.weekday == 5]
    return (d[['report_time', 'geo_value', 'reference_time', 'value']]
            .dropna(subset=['value']))


def latency_days(d):
    first = d.groupby(['geo_value', 'reference_time'])['report_time'].min()
    return (first.index.get_level_values('reference_time'), first.values)


def choose_lag(d, threshold=0.80, max_lag=6):
    """Smallest k such that >=threshold of week-t-k observations are in hand on the
    Wednesday that is 4 days after week t ends."""
    first = d.groupby(['geo_value', 'reference_time'])['report_time'].min().reset_index()
    delay = (first['report_time'] - first['reference_time']).dt.days
    for k in range(0, max_lag + 1):
        # available at the origin iff delay <= 7k + 4 days
        if (delay <= 7 * k + 4).mean() >= threshold:
            return k, float((delay <= 7 * k + 4).mean()), int(delay.median())
    return max_lag, float((delay <= 7 * max_lag + 4).mean()), int(delay.median())


# ------------------------------------------------------------------ fitting

def r2(y, X, adjust=True):
    """R^2, adjusted for the number of predictors by default.

    Adjustment matters here: the covariate arms add two predictors to three, and
    the usable sample differs a lot between sources (PopHIVE ~37 origins per
    state against ~78 for claims). Raw R^2 would reward the short samples.
    """
    n, p = len(y), X.shape[1]
    X = np.column_stack([np.ones(n), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    ss = np.sum((y - y.mean()) ** 2)
    if ss <= 0:
        return np.nan
    r = 1 - np.sum((y - X @ beta) ** 2) / ss
    if adjust and n - p - 1 > 0:
        r = 1 - (1 - r) * (n - 1) / (n - p - 1)
    return r


def covariate_at_origin(cov, origins, lag):
    """For each (geo, NHSN origin): the covariate at weeks t-lag and t-lag-2 as
    known at that origin.

    The source's own report_times rarely coincide with NHSN's Wednesdays, so
    this is an as-of join: each origin sees the latest vintage published on or
    before it. A source that updates monthly is therefore correctly penalised
    for being stale between its updates.
    """
    panel = as_of_panel(cov)                       # (geo, report_time, reference_time)
    panel = panel.sort_values('report_time')
    o = origins.sort_values('report_time')
    frames = []
    for name, extra in [('c0', lag), ('c2', lag + 2)]:
        f = panel.copy()
        f['t'] = f['reference_time'] + pd.Timedelta(days=7 * extra)
        f = f[['geo_value', 't', 'report_time', 'value']].rename(columns={'value': name})
        # as-of: for each (geo, t) pair take the newest vintage <= origin
        f = f.sort_values('report_time')
        m = pd.merge_asof(
            o.rename(columns={'report_time': 'origin'}).assign(key=1).merge(
                f[['geo_value', 't']].drop_duplicates().assign(key=1), on='key'
            ).drop(columns='key').sort_values('origin'),
            f.rename(columns={'report_time': 'origin'}),
            on='origin', by=['geo_value', 't'], direction='backward')
        frames.append(m[['geo_value', 'origin', 't', name]])
    out = frames[0].merge(frames[1], on=['geo_value', 'origin', 't'], how='inner')
    return out.dropna(subset=['c0', 'c2'])


def evaluate(pathogen, cov, lag, label, log_cov=True):
    arch = nhsn_archive(pathogen)
    vint = as_of_panel(arch)
    fin = final_series(arch)

    vint['t'] = vint['report_time'] - pd.Timedelta(days=4)
    cur = vint[vint['reference_time'] == vint['t']][
        ['geo_value', 'report_time', 't', 'value']].rename(columns={'value': 'y_t'})
    prev = vint[vint['reference_time'] == vint['t'] - pd.Timedelta(days=7)][
        ['geo_value', 'report_time', 'value']].rename(columns={'value': 'y_tm1'})
    panel = cur.merge(prev, on=['geo_value', 'report_time'], how='inner')

    origins = panel[['report_time']].drop_duplicates()
    cv = covariate_at_origin(cov, origins, lag).rename(columns={'origin': 'report_time'})
    panel = panel.merge(cv, on=['geo_value', 'report_time', 't'], how='inner')

    rows = []
    for h in [1, 2, 3, 4]:
        tgt = fin.copy()
        tgt['t'] = tgt['reference_time'] - pd.Timedelta(days=7 * h)
        p = panel.merge(tgt[['geo_value', 't', 'y_final']], on=['geo_value', 't'], how='inner')
        base, withcov, nobs = [], [], []
        for geo, g in p.groupby('geo_value'):
            y = np.log1p(g['y_final'].to_numpy(float))
            a0, a1 = np.log1p(g['y_t'].to_numpy(float)), np.log1p(g['y_tm1'].to_numpy(float))
            c0 = g['c0'].to_numpy(float); c2 = g['c2'].to_numpy(float)
            if log_cov:
                c0, c2 = np.log1p(np.clip(c0, 0, None)), np.log1p(np.clip(c2, 0, None))
            A = np.column_stack([a0, a1, a0 - a1])
            C = np.column_stack([c0, c0 - c2])
            ok = np.isfinite(y) & np.isfinite(A).all(1) & np.isfinite(C).all(1)
            if ok.sum() < MIN_OBS:
                continue
            base.append(r2(y[ok], A[ok]))
            withcov.append(r2(y[ok], np.column_stack([A[ok], C[ok]])))
            nobs.append(int(ok.sum()))
        if not base:
            continue
        rows.append({'source': label, 'pathogen': pathogen, 'h': h, 'lag_wk': lag,
                     'states': len(base), 'obs': int(np.median(nobs)),
                     'ar': np.nanmedian(base), 'ar_cov': np.nanmedian(withcov),
                     'gain': np.nanmedian(withcov) - np.nanmedian(base)})
    return pd.DataFrame(rows)
