"""The two production wastewater indices from Delphi NWSS: `wval_like`, `pct_rank`.

Every index shares the same skeleton and differs in exactly one place:

  group g   = (sewershed, nwss_source, pcr_target, major_lab_method)
  x         = ln(signal value)
  score     = <per-index transform of x against g's own history>   <- DENOMINATOR
  site-week = mean of score over the week's samples, then median over groups at the site
  state-wk  = median over sites

Keys
  wval_like       exp((x - p10_g) / sd_g), unweighted median over sites
  pct_rank        empirical percentile of x within g's history, unweighted median

These are the only two indices `dataset.build`/`COVARIATE_GROUPS` promotes into
the production covariate panel (`ww_wval_like`/`ww_pct_rank`). Four other
candidates that were screened but not promoted — `robust_z`, `flowpop_wval`,
`conc_matched`, `wval_popw` — moved to `analysis/covariates/exploratory.py`
(docs/design/restructure-2026-unified.md §2).
"""
import numpy as np
import pandas as pd

MIN_WEEKS = 26
MIN_SITES = 3
FLOWPOP_COVERAGE = 0.80


def load_signal(path, site_tbl, lab_tbl):
    d = pd.read_csv(path, dtype={'geo_value': str, 'nwss_source': str,
                                 'sample_index': str, 'pcr_target': str})
    d['value'] = pd.to_numeric(d['value'], errors='coerce')
    d = d[d['value'] > 0]
    d = d.merge(lab_tbl, on=['geo_value', 'nwss_source', 'reference_time',
                             'sample_index', 'pcr_target'], how='left')
    d = d.merge(site_tbl, on='geo_value', how='inner')
    d['reference_time'] = pd.to_datetime(d['reference_time'])
    d['major_lab_method'] = d['major_lab_method'].fillna('')
    d['g'] = (d['geo_value'] + '|' + d['nwss_source'] + '|' +
              d['pcr_target'] + '|' + d['major_lab_method'])
    d['x'] = np.log(d['value'])
    d['week_end'] = d['reference_time'] + pd.to_timedelta(
        (5 - d['reference_time'].dt.weekday) % 7, unit='D')
    return d


def eligible(d):
    grp = d.groupby('g')
    nweeks = grp['week_end'].nunique()
    spread = grp['x'].std()
    keep = nweeks[nweeks >= MIN_WEEKS].index.intersection(spread[spread > 0].index)
    return d[d['g'].isin(keep)].copy()


def score_wval(d):
    grp = d.groupby('g')['x']
    return np.exp((d['x'] - d['g'].map(grp.quantile(0.10))) / d['g'].map(grp.std()))


def score_pct_rank(d):
    return d.groupby('g')['x'].rank(pct=True)


def weighted_median(values, weights):
    o = np.argsort(values)
    v, w = np.asarray(values)[o], np.asarray(weights, dtype=float)[o]
    if not np.isfinite(w).any() or np.nansum(w) <= 0:
        return np.median(v)
    w = np.nan_to_num(w)
    c = np.cumsum(w) / np.sum(w)
    return float(v[np.searchsorted(c, 0.5)])


def aggregate(d, score, popweight=False):
    t = d.assign(score=score).dropna(subset=['score'])
    gw = t.groupby(['state', 'geo_value', 'g', 'week_end'])['score'].mean().reset_index()
    sw = gw.groupby(['state', 'geo_value', 'week_end'])['score'].median().reset_index()
    sw = sw.merge(d[['geo_value', 'population_served']].drop_duplicates('geo_value'),
                  on='geo_value', how='left')

    def reduce(frame, key):
        if popweight:
            r = frame.groupby(key).apply(
                lambda z: weighted_median(z['score'].values, z['population_served'].values),
                include_groups=False).rename('index').reset_index()
            r['n_sites'] = frame.groupby(key).size().values
        else:
            r = frame.groupby(key).agg(index=('score', 'median'),
                                       n_sites=('score', 'size')).reset_index()
        return r

    st = reduce(sw, ['state', 'week_end'])
    st = st[st['n_sites'] >= MIN_SITES]
    nat = reduce(sw, ['week_end'])
    nat['state'] = 'US'
    return pd.concat([st, nat[['state', 'week_end', 'index', 'n_sites']]], ignore_index=True)


def build_all(pathogen='covid'):
    site = pd.read_csv('aux_site.csv', dtype={'geo_value': str, 'state': str})
    site['population_served'] = pd.to_numeric(site['population_served'], errors='coerce')
    lab = pd.read_csv('aux_lab.csv', dtype=str)[
        ['geo_value', 'nwss_source', 'reference_time', 'sample_index',
         'pcr_target', 'major_lab_method']]

    conc = eligible(load_signal(f'{pathogen}.csv', site, lab))
    out = {'wval_like': aggregate(conc, score_wval(conc)), 'pct_rank': aggregate(conc, score_pct_rank(conc))}
    res = pd.concat([v.assign(key=k) for k, v in out.items()], ignore_index=True)
    res['pathogen'] = pathogen
    return res


if __name__ == '__main__':
    r = build_all('covid')
    r.to_csv('indices_covid.csv', index=False)
    print(r.groupby('key')['index'].describe()[['count', 'min', '50%', 'max']].round(3))
