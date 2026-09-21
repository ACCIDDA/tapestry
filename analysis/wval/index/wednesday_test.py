"""Does wastewater add anything at a Wednesday forecast origin?

Timing, measured rather than assumed:
  NHSN admissions for the week ending Saturday appear 4 days later, on the
  Wednesday. So at a Wednesday origin the forecaster HAS week t complete.
  Delphi NWSS samples take a median of 11 days from collection to first
  appearance, and only ~8% arrive within 5 days. So at that same Wednesday,
  wastewater reference week t is ~8% reported, t-1 ~60%, t-2 ~90%.

Baseline uses NHSN's own history. Wastewater enters either at its realistically
available week (t-2) or, as an upper bound, at week t with no latency at all.
"""
import numpy as np, pandas as pd
import indices as I

MIN_OBS = 60

def wastewater(pathogen):
    site = pd.read_csv('aux_site.csv', dtype={'geo_value': str, 'state': str})
    site['population_served'] = pd.to_numeric(site['population_served'], errors='coerce')
    lab = pd.read_csv('aux_lab.csv', dtype=str)[
        ['geo_value','nwss_source','reference_time','sample_index','pcr_target','major_lab_method']]
    d = I.eligible(I.load_signal(f'{pathogen}.csv', site, lab))
    d = d.assign(score=I.score_wval(d))
    floor = d.groupby('g')['x'].transform('min')
    d['above'] = d['x'] > floor + 1e-12
    sw = d.groupby(['state','geo_value','week_end']).agg(
        score=('score','median'), above=('above','mean')).reset_index()
    st = sw.groupby(['state','week_end']).agg(
        log_mean=('score', lambda s: float(np.mean(np.log(s)))),
        frac_above=('above','mean'), n=('score','size')).reset_index()
    return st[st['n'] >= 3]


def r2(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss = np.sum((y - y.mean())**2)
    return 1 - np.sum(resid**2)/ss if ss > 0 else np.nan


def run(pathogen, target):
    ww = wastewater(pathogen)
    nhsn = pd.read_csv('nhsn.csv', parse_dates=['week_end'])[['state','week_end',target]]
    nhsn = nhsn.rename(columns={target: 'y'}).dropna(subset=['y'])
    m = nhsn.merge(ww, on=['state','week_end'], how='inner').sort_values(['state','week_end'])

    rows = []
    for h in [1, 2, 3, 4]:
        res = {'horizon_wk': h, 'n_states': 0}
        acc = {k: [] for k in ['ar', 'ar_ww_lag2', 'ar_ww_now', 'ww_only_lag2']}
        for s, g in m.groupby('state'):
            g = g.set_index('week_end').asfreq('W-SAT') if False else g.sort_values('week_end')
            g = g.reset_index(drop=True)
            ly = np.log1p(g['y'].to_numpy(float))
            lm = g['log_mean'].to_numpy(float)
            fa = g['frac_above'].to_numpy(float)
            wk = g['week_end'].to_numpy()
            # require a regular weekly grid so the shifts mean what they say
            step = np.diff(wk).astype('timedelta64[D]').astype(int)
            good = np.r_[True, step == 7]
            def sh(a, k):
                out = np.full_like(a, np.nan, dtype=float)
                if k > 0: out[k:] = a[:-k]
                elif k == 0: out = a.copy()
                return out
            y_t = ly
            feats = {
                'ar': np.column_stack([sh(ly,0), sh(ly,1), sh(ly,0)-sh(ly,1)]),
                'ww2': np.column_stack([sh(lm,2), sh(lm,2)-sh(lm,4), sh(fa,2), sh(fa,2)-sh(fa,4)]),
                'ww0': np.column_stack([sh(lm,0), sh(lm,0)-sh(lm,2), sh(fa,0), sh(fa,0)-sh(fa,2)]),
            }
            tgt = np.full(len(ly), np.nan); tgt[:-h] = ly[h:]
            base_ok = np.isfinite(tgt) & good & np.isfinite(feats['ar']).all(1)
            ok = base_ok & np.isfinite(feats['ww2']).all(1) & np.isfinite(feats['ww0']).all(1)
            if ok.sum() < MIN_OBS:
                continue
            y = tgt[ok]
            acc['ar'].append(r2(y, feats['ar'][ok]))
            acc['ar_ww_lag2'].append(r2(y, np.column_stack([feats['ar'][ok], feats['ww2'][ok]])))
            acc['ar_ww_now'].append(r2(y, np.column_stack([feats['ar'][ok], feats['ww0'][ok]])))
            acc['ww_only_lag2'].append(r2(y, feats['ww2'][ok]))
            res['n_states'] += 1
        for k, v in acc.items():
            res[k] = np.nanmedian(v) if v else np.nan
        res['gain_lag2'] = res['ar_ww_lag2'] - res['ar']
        res['gain_now'] = res['ar_ww_now'] - res['ar']
        rows.append(res)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    for pathogen, target in [('flu','flu'), ('covid','covid'), ('rsv','rsv')]:
        t = run(pathogen, target)
        print(f"\n=== {pathogen.upper()} -> NHSN {target} admissions/100k, in-sample R², "
              f"median over {int(t['n_states'].iloc[0])} states ===")
        print(t[['horizon_wk','ar','ar_ww_lag2','gain_lag2','ar_ww_now','gain_now','ww_only_lag2']]
              .round(3).to_string(index=False))
