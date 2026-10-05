"""Independent baselinenowcast point comparator; see docs/reference/baselinenowcast.md."""
import numpy as np


def apply_delay(report, age, cdf, integer=False):
    """Apply upstream's successive incremental expectations, preserving reports."""
    result = np.asarray(report, dtype=float).copy()
    for d in range(1, len(cdf)):
        use = age < d
        increment = cdf[d] - cdf[d-1]
        expectation = (result[use] + float(integer)*(1-cdf[d-1])) / cdf[d-1]
        result[use] += expectation * increment
    return np.maximum(0., result)


def delay_cdf(cumulative, integer=False):
    """Complete a regular triangle and estimate its delay CDF from column totals.

    Missing historical reports must be removed before calling, not filled as zero.
    Signed increments preserve downward revisions. None means insufficient support.
    """
    observed = np.isfinite(cumulative)
    if not len(cumulative) or not observed.all(1).any() or not observed[:, 0].all():
        return None
    if np.any(observed[:, 1:] & ~observed[:, :-1]) or np.any(np.diff(observed.astype(int), axis=0) > 0):
        return None
    increments = np.diff(cumulative, axis=1, prepend=0.)
    filled = increments.copy()
    for d in range(1, cumulative.shape[1]):
        missing = ~observed[:, d]
        if not missing.any():
            continue
        top = observed[:, d]
        exposure = cumulative[top, d-1].sum()
        denominator = max(exposure, 1.) if integer else exposure
        if denominator <= 0:
            return None
        factor = increments[top, d].sum() / denominator
        filled[missing, d] = factor * filled[missing, :d].sum(1)
    total = filled.sum()
    if not np.isfinite(total) or total <= 0:
        return None
    cdf = np.cumsum(filled.sum(0) / total)
    return cdf if np.isfinite(cdf).all() and (cdf > 0).all() else None


def predictions(panel, asof, rows, anchor, integer=False, max_delay=12, window=52):
    """Causal per-location baseline on every row, with explicit fallback status."""
    dates = panel['dates'].astype('datetime64[D]')
    issues = panel['issuance_dates'].astype('datetime64[D]')
    scheduled = dates + np.timedelta64(4 + 7*rows['lag'], 'D')
    report_dates = scheduled[:, None] + np.arange(max_delay+1) * np.timedelta64(7, 'D')
    indices = np.searchsorted(issues, report_dates)
    exact = (indices < len(issues)) & (issues[indices.clip(max=len(issues)-1)] == report_dates)
    triangle = asof[indices.clip(max=len(issues)-1), np.arange(len(dates))[:, None]].astype(float)
    triangle[~exact] = np.nan
    result = np.array(anchor, dtype=float, copy=True)
    status = np.full(len(result), 'missing_report', dtype='U32')
    support = np.zeros(len(result), dtype=int)
    row_issues = rows['issuance'].astype('datetime64[D]')
    for issue in np.unique(row_issues):
        ts = np.flatnonzero(scheduled <= issue)[-window:]
        expected = report_dates[ts] <= issue
        for loc in np.unique(rows['location'][row_issues == issue]):
            use = (row_issues == issue) & (rows['location'] == loc)
            reported = use & np.isfinite(rows['baseline_history'][:, -1])
            mature = reported & (rows['age'] >= max_delay)
            result[mature] = rows['baseline_history'][mature, -1]
            status[mature] = 'already_mature'
            active = reported & ~mature
            if not active.any():
                continue
            history = np.where(expected, triangle[ts, :, loc], np.nan)
            # Retain only rows with every scheduled vintage actually archived.
            complete_prefix = ((np.isfinite(history) & (history >= 0)) | ~expected).all(1)
            history = history[complete_prefix]
            support[active] = len(history)
            cdf = delay_cdf(history, integer=integer)
            status[active] = 'insufficient_triangle'
            if cdf is None:
                continue
            result[active] = apply_delay(rows['baseline_history'][active, -1],
                                         rows['age'][active], cdf, integer=integer)
            status[active] = 'estimated'
    return result, status, support
