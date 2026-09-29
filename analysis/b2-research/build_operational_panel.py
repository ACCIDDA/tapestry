"""Backcast an explicit 2025-26 reporting regime for historical forecasts.

Actual deadline vintages win. Only a missing, never-reported-at-that-cutoff cell
can receive its finalized historical value as a proxy, after its source lag.
Known explicit nulls/withdrawals and structurally absent truth stay unavailable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tapestry.dataset.build import load, save
from tapestry.dataset.cv import season


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/processed/panel-b2-deadline.npz')
    parser.add_argument('--output', default='data/processed/panel-b2-operational.npz')
    parser.add_argument('--report-dir', default='docs/data/availability/provenance')
    args = parser.parse_args()
    panel = load(args.source)
    metadata = json.loads(str(panel['metadata']))
    native = {name: detail for source in metadata['deadline_build']['source_details'].values()
              for name, detail in source['first_release'].items()}
    lags = {}
    for name, detail in native.items():
        value = detail['first_release_median_days']
        if value is None:
            raise ValueError(f'No empirical operational first-release lag for {name}; specify a scientific assumption before building')
        lags[name] = int(np.floor(value))
    # Native daily Kinsa history after its append-only ingestion stabilized has
    # one-day publication delay. The week is complete after Saturday's value.
    lags['kinsa_ili'] = 1
    # The current FluView release calendar is Friday for the preceding Saturday.
    # This documented six-day lag avoids conflating archive ingestion with source publication.
    lags['ilinet_ili'] = lags['clinical_lab_flu_pct_positive'] = 6
    dates = panel['dates'].astype('datetime64[D]')
    issuances = panel['issuance_dates'].astype('datetime64[D]')
    context_end = issuances - np.timedelta64(4, 'D')
    cutoffs = pd.to_datetime(panel['forecast_cutoff_utc'], utc=True).tz_localize(None).to_numpy('datetime64[ns]')
    rows, nulls = [], []
    for array, truth_name, known_name, names_name in [
        ('asof_targets', 'targets', 'known_report_targets', 'target_names'),
        ('asof_covariates', 'covariates', 'known_report_covariates', 'covariate_names'),
        ('asof_covariates_national', 'covariates_national', 'known_report_covariates_national', 'covariate_national_names'),
    ]:
        observed = panel[array].copy()
        known = panel[known_name]
        truth = panel[truth_name]
        proxy = np.zeros(observed.shape, bool)
        for c, name_ in enumerate(panel[names_name]):
            name = str(name_)
            lag = 4 if truth_name == 'targets' else lags[name]
            # Date-only assumed releases are available at the end of their UTC
            # release day, exactly like the conservative historical source policy.
            assumed_release = (dates + np.timedelta64(lag + 1, 'D')).astype('datetime64[ns]') - np.timedelta64(1, 'ns')
            allowed = (assumed_release[None, :] <= cutoffs[:, None]) & (dates[None, :] <= context_end[:, None])
            x = observed[..., c]
            target = truth[..., c]
            if x.ndim == 3:
                allowed = allowed[:, :, None]
            fill = ~np.isfinite(x) & ~known[..., c] & np.isfinite(target)[None] & allowed
            if truth_name == 'targets' and name.startswith('nssp_'):
                # Missouri's missing ED reports were specifically established in
                # the 2025-26 audit; do not assume those observations will arrive.
                fill[:, :, panel['locations'].tolist().index('MO')] = False
            panel[array][..., c] = np.where(fill, target[None], x)
            proxy[..., c] = fill
            # Freshness is measured separately from per-cell release latency.
            freshness = []
            for w in range(len(issuances)):
                valid = np.isfinite(x[w])
                if valid.ndim == 2:
                    valid = valid.any(-1)
                t = np.flatnonzero(valid)
                if len(t) and season(str(context_end[w])) == '2025-2026':
                    freshness.append(int((context_end[w]-dates[t[-1]])/np.timedelta64(1,'D')))
                recent = (dates <= context_end[w]) & (dates >= context_end[w]-np.timedelta64(77,'D'))
                rows.append(dict(issuance=str(issuances[w]), season=season(str(context_end[w])), name=name, assumed_release_lag_days=lag, actual_context_cells=int(np.isfinite(x[w,recent]).sum()), proxy_context_cells=int(fill[w,recent].sum()), known_null_context_cells=int((known[w,recent,...,c] & ~np.isfinite(x[w,recent])).sum())))
            nulls.append(dict(name=name, lag_days=lag, proxy_cells=int(fill.sum()), actual_cells=int(np.isfinite(x).sum()), protected_null_cells=int((known[...,c] & ~np.isfinite(x)).sum()), latest_observation_origins_2025_2026=len(freshness), latest_observation_median_lag_from_context_days=None if not freshness else float(np.median(freshness)), native_first_release=native.get(name)))
        panel['proxy_' + truth_name] = proxy
        if not np.array_equal(panel[array][np.isfinite(observed)], observed[np.isfinite(observed)]):
            raise AssertionError(f'Actual vintage changed: {array}')
        if np.any(np.isfinite(panel[array]) & known & ~np.isfinite(observed)):
            raise AssertionError(f'Known null filled: {array}')
    policy = dict(source=args.source, source_sha256=sha256(args.source), builder_sha256=sha256(__file__),
        authorized_assumption='Backcast the expected 2025-26 source reporting regime; when archive evidence is absent, lean toward availability using finalized historical values after source-specific delay.',
        covariate_lags_days=lags, target_lag_days=4, sources=nulls,
        native_delay_estimator='Median first publication delay per event-week/location among 2025-26 first releases with delay 0-60 days; floor fractional median days. Reports >60 days are treated as archive backfills and excluded. Explicit overrides: daily Kinsa1day after May2026 stabilization; ILINet and clinical lab6days from CDC weekly release schedule.',
        proxy_value_policy='Actual deadline vintage if present. Otherwise frozen finalized value only where no report/null exists, truth is finite, assumed source release precedes actual deadline, and event week does not exceed original context end. Missouri ED missing cells are never proxied.',
        timestamp_policy='Assumed publication at end of UTC release day; actual holiday-aware Eastern cutoffs retained. This is conservative within the assumed release day.',
        interpretation='Operational availability hypothesis, not reconstructed historical real-time data. Proxy values include later revisions; NWSS proxies additionally use retrospective finalized index baselines. Results measure model/covariate value under this reporting hypothesis and cannot establish historical deployable performance.',
        unchanged='All finalized training/label truth arrays; all actually observed deadline input values; Wednesday origin identities and context boundaries.')
    metadata['operational_backcast'] = policy
    panel['metadata'] = json.dumps(metadata)
    save(panel, args.output)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report_dir/'operational-proxy-coverage.csv', index=False)
    policy.update(output=args.output, output_sha256=sha256(args.output))
    (report_dir/'operational-policy.json').write_text(json.dumps(policy, indent=2)+'\n')
    pd.DataFrame([{k:v for k,v in row.items() if k!='native_first_release'} for row in nulls]).to_csv(report_dir/'operational-source-lags.csv', index=False)
    print(json.dumps(policy, indent=2))


if __name__ == '__main__':
    main()
