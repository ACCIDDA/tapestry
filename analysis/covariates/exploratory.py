"""Four screened-but-not-promoted wastewater index candidates.

Split out of `indices.py` (docs/design/restructure-2026-unified.md §2): only
`wval_like` and `pct_rank` are part of the production covariate panel
(`dataset.build.COVARIATE_GROUPS`). These four remain for reference/comparison
only and are not read by any production code path.

  robust_z        (x - median_g) / IQR_g, unweighted median over sites
  flowpop_wval    wval_like transform applied to flowpop_lin instead of avg_conc_lin
  conc_matched    wval_like on avg_conc_lin, restricted to the flowpop_wval panel
  wval_popw       wval_like scores, population-weighted median over sites
"""
import numpy as np
import pandas as pd

from .indices import aggregate, eligible, load_signal, score_wval, weighted_median

FLOWPOP_COVERAGE = 0.80


def score_robust_z(d):
    grp = d.groupby('g')['x']
    iqr = grp.quantile(0.75) - grp.quantile(0.25)
    iqr = iqr.where(iqr > 0)
    return (d['x'] - d['g'].map(grp.median())) / d['g'].map(iqr)


def build_all(pathogen='covid'):
    site = pd.read_csv('aux_site.csv', dtype={'geo_value': str, 'state': str})
    site['population_served'] = pd.to_numeric(site['population_served'], errors='coerce')
    lab = pd.read_csv('aux_lab.csv', dtype=str)[
        ['geo_value', 'nwss_source', 'reference_time', 'sample_index',
         'pcr_target', 'major_lab_method']]

    conc = eligible(load_signal(f'{pathogen}.csv', site, lab))
    fpop = eligible(load_signal(f'{pathogen}_flowpop_lin.csv', site, lab))

    fp_groups = set(fpop['g'])
    cov = conc.groupby('g').size()
    fcov = fpop.groupby('g').size()
    shared = [g for g in fp_groups
              if g in cov.index and fcov[g] / cov[g] >= FLOWPOP_COVERAGE]
    matched = conc[conc['g'].isin(shared)].copy()

    out = {
        'robust_z':     aggregate(conc, score_robust_z(conc)),
        'flowpop_wval': aggregate(fpop[fpop['g'].isin(shared)],
                                  score_wval(fpop[fpop['g'].isin(shared)])),
        'conc_matched': aggregate(matched, score_wval(matched)),
        'wval_popw':    aggregate(conc, score_wval(conc), popweight=True),
    }
    print(f"groups: conc {conc['g'].nunique()}, flowpop {fpop['g'].nunique()}, "
          f"matched panel {len(shared)}")
    res = pd.concat([v.assign(key=k) for k, v in out.items()], ignore_index=True)
    res['pathogen'] = pathogen
    return res


if __name__ == '__main__':
    r = build_all('covid')
    r.to_csv('indices_covid_exploratory.csv', index=False)
    print(r.groupby('key')['index'].describe()[['count', 'min', '50%', 'max']].round(3))
