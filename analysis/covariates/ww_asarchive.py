"""Wastewater state index expressed as a report_time archive, so it can go
through the identical protocol as the claims and PopHIVE sources."""
import numpy as np, pandas as pd
import indices as I


def wastewater_archive(pathogen):
    site = pd.read_csv('aux_site.csv', dtype={'geo_value': str, 'state': str})
    site['population_served'] = pd.to_numeric(site['population_served'], errors='coerce')
    lab = pd.read_csv('aux_lab.csv', dtype=str)[
        ['geo_value','nwss_source','reference_time','sample_index','pcr_target','major_lab_method']]
    d = I.eligible(I.load_signal(f'{pathogen}.csv', site, lab))
    d = d.assign(score=I.score_wval(d))
    # load_signal already carries report_time; it is the vintage in which this
    # sample's current value was set, which is the best available proxy for when
    # the sample became visible.
    # NOTE: the snapshot's report_time is the vintage in which a value was LAST
    # set, so for rows loaded in Delphi's 2026-06-26 bulk backfill it sits ~265
    # days after collection and is useless as a first-availability proxy. The
    # NWSS archive only begins 2026-02-25, so no observed as-of join is possible
    # for the period NHSN vintages cover. Availability is therefore MODELLED from
    # the measured sample-level latency (median 11 days, 90% within 19), by
    # stamping each state-week as visible 18 days after it ends.
    d['report_time'] = pd.NaT
    sw = d.groupby(['state','geo_value','week_end']).agg(
        score=('score','median')).reset_index()
    st = sw.groupby(['state','week_end']).agg(
        value=('score', lambda s: float(np.mean(np.log(s)))),
        n=('score','size')).reset_index()
    st = st[st['n'] >= 3].copy()
    st['report_time'] = pd.to_datetime(st['week_end']) + pd.Timedelta(days=18)
    return st.rename(columns={'state':'geo_value','week_end':'reference_time'})[
        ['report_time','geo_value','reference_time','value']]
