"""Does an index that keeps resolving below baseline help predict NHSN admissions?

Builds three state-week aggregations from the SAME per-site scores and compares
their lead correlation with NHSN admissions per 100k, overall and restricted to
the pre-onset weeks where the median index is pinned at its floor.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import indices as I

site = pd.read_csv('aux_site.csv', dtype={'geo_value': str, 'state': str})
site['population_served'] = pd.to_numeric(site['population_served'], errors='coerce')
lab = pd.read_csv('aux_lab.csv', dtype=str)[
    ['geo_value','nwss_source','reference_time','sample_index','pcr_target','major_lab_method']]
nhsn = pd.read_csv('nhsn.csv', parse_dates=['week_end'])


def aggregations(pathogen):
    d = I.eligible(I.load_signal(f'{pathogen}.csv', site, lab))
    d = d.assign(score=I.score_wval(d), rank=I.score_pct_rank(d))
    floor = d.groupby('g')['x'].transform('min')
    d['above'] = d['x'] > floor + 1e-12
    sw = d.groupby(['state','geo_value','week_end']).agg(
        score=('score','median'), rank=('rank','median'), above=('above','mean')).reset_index()
    st = sw.groupby(['state','week_end']).agg(
        wval_like=('score','median'),
        pct_rank=('rank','median'),
        frac_above=('above','mean'),
        log_mean=('score', lambda s: float(np.mean(np.log(s)))),
        n=('score','size')).reset_index()
    return st[st['n'] >= 3]


def lead_table(pathogen, target):
    st = aggregations(pathogen)
    m = st.merge(nhsn[['state','week_end',target]].rename(columns={target:'y'}),
                 on=['state','week_end'], how='inner').dropna(subset=['y'])
    pinned = np.abs(m['wval_like'] - 1) < 1e-9
    out = []
    for lag in range(0, 5):
        row = {'lag_weeks': lag}
        for col in ['wval_like','pct_rank','frac_above','log_mean']:
            rs = []
            rs_pin = []
            for s, g in m.groupby('state'):
                g = g.sort_values('week_end')
                x = g[col].to_numpy(float)
                y = g['y'].shift(-lag).to_numpy(float)
                ok = np.isfinite(x) & np.isfinite(y)
                if ok.sum() > 40:
                    rs.append(spearmanr(x[ok], y[ok]).statistic)
                p = pinned.loc[g.index].to_numpy()
                okp = ok & p
                if okp.sum() > 40 and np.ptp(x[okp]) > 0:
                    rs_pin.append(spearmanr(x[okp], y[okp]).statistic)
            row[col] = np.nanmedian(rs) if rs else np.nan
            row[col + '_pinned'] = np.nanmedian(rs_pin) if rs_pin else np.nan
        out.append(row)
    return pd.DataFrame(out), float(pinned.mean()), len(m)


if __name__ == '__main__':
    for pathogen, target in [('flu','flu'), ('covid','covid'), ('rsv','rsv')]:
        t, pin, n = lead_table(pathogen, target)
        print(f"\n=== {pathogen.upper()} vs NHSN {target} admissions/100k "
              f"({n} state-weeks, {pin:.0%} pinned at the floor) ===")
        print("Spearman with admissions at t+lag, median over states\n")
        print(t[['lag_weeks','wval_like','pct_rank','frac_above','log_mean']]
              .round(3).to_string(index=False))
        print("\nrestricted to weeks where wval_like is pinned at exactly 1.0:")
        cols = ['lag_weeks'] + [c for c in t.columns if c.endswith('_pinned')]
        print(t[cols].round(3).to_string(index=False))
