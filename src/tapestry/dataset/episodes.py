"""Cut training/scoring episodes of any lookback from the one panel (`dataset.build`).

One builder, one switch (docs/architecture.md):

- `input_mode='finalized'`: one episode per Saturday origin t of the calendar, with
  context weeks t-lookback+1..t and target weeks t+1..t+4 from the truth panel.
  Origins run from the first calendar week to the last one whose four target weeks
  lie inside the calendar. Context weeks before the calendar are unavailable (padding).
  Every visible context cell is known-final.
- `input_mode='finalized_available'`: one episode per Wednesday; targets and
  covariates use final values only where the Wednesday snapshot has a report.
  This permits later revisions, but never fills an unpublished input. Labels
  are final truth. It is a retrospective forecaster training protocol.
- `input_mode='vintaged'`: one episode per Wednesday issuance, origin = its context
  end (the Saturday four days earlier). With `asof_weeks` = the scenario's field
  (default 2, the previous vintaged builder):
  * the last min(asof_weeks, lookback) context weeks take the target value visible
    at the issuance cutoff (known_final False) and are **unavailable where nothing
    was visible** (user decision 2026-09-22: a cell that was not published at the
    cutoff must be unavailable, not filled with later truth; this replaces the
    previous truth fallback and changes vintaged results against old B1);
  * older context weeks and all target weeks are truth, known-final where available;
  * covariates (state and national) are as of the issuance for every context week,
    NaN where nothing was visible (no truth fallback), as the previous builder did.
  `known_final` is not stored in the panel: it follows from position and visibility.

The `horizons` argument controls label offsets from the context end; forecasting
uses (1,2,3,4), nowcasting uses (1-R,...,0). Nowcast/pipeline CV callers set
asof_weeks=lookback so their entire input history is as-of.

An episode is kept only if some context cell and some target cell are available.
Covariates (`covariate_names`) are per-location `[lookback, K, 2, L]` (value, available);
a national name (Kinsa) is broadcast to every location with its original
availability. This is a shared national signal, not a state-level measurement.
"""
from datetime import date, timedelta
import numpy as np

from .build import context_end

HORIZONS = (1, 2, 3, 4)


def select_covariates(state, national, state_names, national_names, locations, names):
    """`[..., L, Ks]` state + `[..., Kn]` national arrays -> `[..., K, L]` values and availability."""
    state_names, national_names, locations = list(state_names), list(national_names), list(locations)
    values = np.zeros((*state.shape[:-2], len(names), len(locations)), np.float32)
    for k, name in enumerate(names):
        if name in state_names:
            values[..., k, :] = state[..., state_names.index(name)]
        elif name in national_names:
            values[..., k, :] = national[..., national_names.index(name), None]
        else:
            raise ValueError(f'Unknown covariate: {name}')
    available = ~np.isnan(values)
    return np.nan_to_num(values), available


def _pad(array, before, after, axis=0):
    shape = lambda n: (*array.shape[:axis], n, *array.shape[axis + 1:])
    return np.concatenate([np.full(shape(before), np.nan, array.dtype), array,
                           np.full(shape(after), np.nan, array.dtype)], axis=axis)


def _channel_first(panel):
    """`[.., L, C]` (the array-on-disk convention) -> `[.., C, L]` (Model's convention)."""
    return np.moveaxis(panel, -1, -2)


def episodes(panel, lookback, input_mode, covariate_names=(), asof_weeks=2, horizons=HORIZONS):
    """All usable episodes of a (possibly masked) panel; see the module docstring."""
    if input_mode not in ('finalized', 'finalized_available', 'scheduled_final', 'vintaged'):
        raise ValueError(f'Unknown input_mode: {input_mode}')
    vintaged = input_mode == 'vintaged'
    dated = input_mode not in ('finalized', 'scheduled_final')
    dates = [str(d) for d in panel['dates']]
    first, locations = date.fromisoformat(dates[0]), tuple(str(l) for l in panel['locations'])
    pad, tail = lookback - 1, max(0, max(horizons))
    targets = _channel_first(_pad(panel['targets'], pad, tail))  # padded index = calendar index + pad
    if dated:
        asof_targets = _channel_first(_pad(panel['asof_targets'], pad, tail, axis=1))
    if covariate_names:
        names = (panel['covariate_names'], panel['covariate_national_names'], locations, covariate_names)
        if dated:
            covariates = select_covariates(_pad(panel['asof_covariates'], pad, tail, axis=1),
                                           _pad(panel['asof_covariates_national'], pad, tail, axis=1), *names)
        else:
            covariates = select_covariates(_pad(panel['covariates'], pad, tail),
                                           _pad(panel['covariates_national'], pad, tail), *names)
        if input_mode == 'finalized_available':
            final_values, final_available = select_covariates(
                _pad(panel['covariates'], pad, tail),
                _pad(panel['covariates_national'], pad, tail), *names)
            available = covariates[1] & final_available[None]
            covariates = (np.where(available, final_values[None], 0), available)
    week = lambda t: (first + timedelta(weeks=t)).isoformat()
    if dated:
        origins = [((date.fromisoformat(context_end(d)) - first).days // 7, w)
                   for w, d in enumerate(panel['issuance_dates'])]
    else:
        origins = [(t, None) for t in range(len(dates) - tail)]
    recent = min(asof_weeks, lookback)
    result = []
    for t, w in origins:
        if t < 0:
            continue  # context ends before the calendar: nothing is available
        context = slice(t, t + lookback)  # padded indices of calendar weeks t-lookback+1..t
        future = [t + pad + h for h in horizons]
        values = targets[context].copy()
        known_final = ~np.isnan(values)
        if vintaged and recent:
            # Nothing visible at the cutoff stays NaN: unavailable, never truth-filled.
            values[-recent:] = asof_targets[w, context][-recent:]
            known_final[-recent:] = False
        if input_mode == 'finalized_available':
            values[np.isnan(asof_targets[w, context])] = np.nan
            known_final = ~np.isnan(values)
        # scheduled_final keeps all six target histories through T-0, including
        # the latest context week. Source-specific covariate lags apply below.
        available = ~np.isnan(values)
        target_values = targets[future]
        target_available = ~np.isnan(target_values)
        if not available.any() or not target_available.any():
            continue
        episode = dict(values=np.nan_to_num(values), available=available, known_final=known_final,
                       target_values=np.nan_to_num(target_values), target_available=target_available,
                       context_dates=tuple(week(t - pad + i) for i in range(lookback)),
                       target_dates=tuple(week(t + h) for h in horizons), locations=locations,
                       issuance=None if w is None else str(panel['issuance_dates'][w]))
        if w is not None and 'forecast_cutoff_utc' in panel:
            episode['forecast_cutoff_utc'] = str(panel['forecast_cutoff_utc'][w])
        episode['Y'] = np.stack((episode['target_values'], episode['target_available']), axis=2)
        if covariate_names:
            cov_values, cov_available = (covariates[0][context], covariates[1][context]) if w is None else \
                (covariates[0][w, context], covariates[1][w, context])
            if input_mode == 'scheduled_final':
                cov_values, cov_available = cov_values.copy(), cov_available.copy()
                for k, name in enumerate(covariate_names):
                    if name in ('ilinet_ili', 'clinical_lab_flu_pct_positive', 'flusurv_flu_rate'):
                        cov_available[-1, k] = False
                    if name in ('inpatient_flu', 'inpatient_covid') and 'VT' in locations:
                        cov_available[:, k, locations.index('VT')] = False
                cov_values = np.where(cov_available, cov_values, 0)
            episode['covariates'] = np.stack((cov_values, cov_available), axis=-2)
        result.append(episode)
    return result


def restrict_labels(episode, allowed_dates):
    """Copy of `episode` whose labels outside `allowed_dates` are unavailable (None if no label is left)."""
    keep = np.array([d in allowed_dates for d in episode['target_dates']])
    episode = dict(episode, target_available=episode['target_available'] & keep[:, None, None])
    episode['target_values'] = np.where(episode['target_available'], episode['target_values'], 0).astype(np.float32)
    episode['Y'] = np.stack((episode['target_values'], episode['target_available']), axis=2)
    return episode if episode['target_available'].any() else None


def calendar(days, dynamics=True):
    phase = np.array([date.fromisoformat(day).timetuple().tm_yday for day in days]) * (2 * np.pi / 365.25)
    columns = [np.sin(phase), np.cos(phase)]
    if dynamics:
        dates = [date.fromisoformat(day) for day in days]
        columns.append(np.array([(d - date(d.year if d.month >= 7 else d.year - 1, 12, 25)).days / (7 * 26) for d in dates]))
    return np.stack(columns, axis=-1).astype('float32')
