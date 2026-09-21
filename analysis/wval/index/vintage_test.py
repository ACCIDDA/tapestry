"""Wednesday-origin test, rerun against real NHSN vintages.

The earlier run (wednesday_test.py) conditioned on FINAL admissions for the most
recent week, which no forecaster holds at the origin. Here the autoregressive
features are rebuilt from the Delphi NHSN archive as they actually stood at each
Wednesday report_time; only the target stays final.

Both arms are restricted to the same origins (the archive starts 2024-11-19), so
"final" and "vintage" differ in one thing only.
"""
import glob
import numpy as np, pandas as pd
import indices as I

ARCHIVE = 'data/raw/delphi_nhsn/snapshots/*/signal=confirmed_admissions_{p}_ew/geo_type=state/archive.csv.gz'
ROOT = '/Users/chadi/Research/Tapestry/'
SIGNAL = {'flu': 'flu', 'covid': 'covid', 'rsv': 'rsv'}
MIN_OBS = 40


def nhsn_archive(pathogen):
    f = glob.glob(ROOT + ARCHIVE.format(p=SIGNAL[pathogen]))[0]
    d = pd.read_csv(f, parse_dates=['report_time', 'reference_time'])
    return d[['report_time', 'geo_value', 'reference_time', 'value']].dropna(subset=['value'])


def vintage_panel(d):
    """For each origin (report_time), the value of every reference week as known then."""
    d = d.sort_values('report_time')
    # latest value per (origin, geo, ref) -- archive rows are already per-report
    latest = d.groupby(['report_time', 'geo_value', 'reference_time'], as_index=False)['value'].last()
    # carry forward: a week not restated in this vintage keeps its previous value
    out = []
    for (geo,), g in latest.groupby(['geo_value']):
        piv = g.pivot(index='report_time', columns='reference_time', values='value').sort_index()
        piv = piv.ffill()
        s = piv.stack().rename('value').reset_index()
        s['geo_value'] = geo
        out.append(s)
    return pd.concat(out, ignore_index=True)


def final_series(d):
    return (d.sort_values('report_time')
             .groupby(['geo_value', 'reference_time'], as_index=False)['value'].last()
             .rename(columns={'value': 'y_final'}))


def wastewater(pathogen):
    site = pd.read_csv('aux_site.csv', dtype={'geo_value': str, 'state': str})
    site['population_served'] = pd.to_numeric(site['population_served'], errors='coerce')
    lab = pd.read_csv('aux_lab.csv', dtype=str)[
        ['geo_value','nwss_source','reference_time','sample_index','pcr_target','major_lab_method']]
    d = I.eligible(I.load_signal(f'{pathogen}.csv', site, lab))
    d = d.assign(score=I.score_wval(d))
    floor = d.groupby('g')['x'].transform('min')
    d['above'] = d['x'] > floor + 1e-12
    sw = d.groupby(['state', 'geo_value', 'week_end']).agg(
        score=('score', 'median'), above=('above', 'mean')).reset_index()
    st = sw.groupby(['state', 'week_end']).agg(
        log_mean=('score', lambda s: float(np.mean(np.log(s)))),
        frac_above=('above', 'mean'), n=('score', 'size')).reset_index()
    return st[st['n'] >= 3].rename(columns={'state': 'geo_value', 'week_end': 'reference_time'})


def r2(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    ss = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - X @ beta) ** 2) / ss if ss > 0 else np.nan


def run(pathogen):
    arch = nhsn_archive(pathogen)
    vint = vintage_panel(arch)
    fin = final_series(arch)
    ww = wastewater(pathogen)

    # at origin r, the newest reference week is the Saturday 4 days earlier
    vint['t'] = vint['report_time'] - pd.Timedelta(days=4)
    cur = vint[vint['reference_time'] == vint['t']][['geo_value','report_time','t','value']] \
        .rename(columns={'value': 'y_t_vintage'})
    prev = vint[vint['reference_time'] == vint['t'] - pd.Timedelta(days=7)][
        ['geo_value','report_time','value']].rename(columns={'value': 'y_tm1_vintage'})
    panel = cur.merge(prev, on=['geo_value','report_time'], how='inner')

    panel = panel.merge(fin.rename(columns={'reference_time':'t','y_final':'y_t_final'}),
                        on=['geo_value','t'], how='left')
    panel = panel.merge(fin.rename(columns={'reference_time':'t','y_final':'y_tm1_final'})
                          .assign(t=lambda z: z['t'] + pd.Timedelta(days=7)),
                        on=['geo_value','t'], how='left')
    for lag, name in [(2,'ww2'), (4,'ww4')]:
        w = ww.copy()
        w['t'] = w['reference_time'] + pd.Timedelta(days=7*lag)
        panel = panel.merge(w[['geo_value','t','log_mean','frac_above']]
                            .rename(columns={'log_mean': f'lm_{name}', 'frac_above': f'fa_{name}'}),
                            on=['geo_value','t'], how='left')

    rows = []
    for h in [1, 2, 3, 4]:
        tgt = fin.copy()
        tgt['t'] = tgt['reference_time'] - pd.Timedelta(days=7*h)
        p = panel.merge(tgt[['geo_value','t','y_final']].rename(columns={'y_final':'y_target'}),
                        on=['geo_value','t'], how='inner')
        acc = {k: [] for k in ['ar_final','ar_vintage','ar_vintage_ww','ar_final_ww']}
        nst = 0
        nobs = []
        for geo, g in p.groupby('geo_value'):
            L = lambda c: np.log1p(g[c].to_numpy(float))
            y = L('y_target')
            fv, fp = L('y_t_final'), L('y_tm1_final')
            vv, vp = L('y_t_vintage'), L('y_tm1_vintage')
            W = np.column_stack([g['lm_ww2'], g['lm_ww2'] - g['lm_ww4'],
                                 g['fa_ww2'], g['fa_ww2'] - g['fa_ww4']]).astype(float)
            Af = np.column_stack([fv, fp, fv - fp])
            Av = np.column_stack([vv, vp, vv - vp])
            ok = np.isfinite(y) & np.isfinite(Af).all(1) & np.isfinite(Av).all(1) & np.isfinite(W).all(1)
            if ok.sum() < MIN_OBS:
                continue
            nst += 1
            nobs.append(int(ok.sum()))
            acc['ar_final'].append(r2(y[ok], Af[ok]))
            acc['ar_vintage'].append(r2(y[ok], Av[ok]))
            acc['ar_vintage_ww'].append(r2(y[ok], np.column_stack([Av[ok], W[ok]])))
            acc['ar_final_ww'].append(r2(y[ok], np.column_stack([Af[ok], W[ok]])))
        row = {'h': h, 'states': nst,
               'obs_per_state': int(np.median(nobs)) if nobs else 0}
        row.update({k: (np.nanmedian(v) if v else np.nan) for k, v in acc.items()})
        row['gain_vintage'] = row['ar_vintage_ww'] - row['ar_vintage']
        row['gain_final'] = row['ar_final_ww'] - row['ar_final']
        row['cost_of_vintage'] = row['ar_final'] - row['ar_vintage']
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    for p in ['flu', 'covid', 'rsv']:
        t = run(p)
        print(f"\n=== {p.upper()} — same origins (2024-11 on), in-sample R², median over states ===")
        print(t[['h','states','ar_final','ar_vintage','cost_of_vintage',
                 'ar_vintage_ww','gain_vintage','gain_final']].round(3).to_string(index=False))
