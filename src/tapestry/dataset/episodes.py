"""Cut training/scoring episodes of any lookback from the one panel (`dataset.build`).

One builder, one switch (docs/design/restructure-2026-unified.md §3):

- `input_mode='finalized'`: one episode per Saturday origin t of the calendar, with
  context weeks t-lookback+1..t and target weeks t+1..t+4 from the truth panel.
  Origins run from the first calendar week to the last one whose four target weeks
  lie inside the calendar. Context weeks before the calendar are unavailable (padding).
  Every visible context cell is known-final.
- `input_mode='vintaged'`: one episode per Wednesday issuance, origin = its context
  end (the Saturday four days earlier). The window is the truth panel with the
  as-of overlay applied, exactly what the previous vintaged builder materialized:
  * the last min(R, lookback) context weeks take the value visible at the issuance
    cutoff; where nothing was visible yet they fall back to truth, flagged known-final;
  * older context weeks and all target weeks are truth, known-final where available;
  * covariates are as of the issuance for every context week (requires lookback <= D).
  `known_final` is not stored in the panel: it is a function of overlay availability.

An episode is kept only if some context cell and some target cell are available.
Covariates (`covariate_names`) are per-location `[lookback, K, 2, L]` (value, available);
a national-only name is broadcast to the `US` column only.
"""
from datetime import date, timedelta
import json

import numpy as np

from .build import context_end

HORIZONS = (1, 2, 3, 4)


def select_covariates(state, national, state_names, national_names, locations, names):
    """`[..., L, Ks]` state + `[..., Kn]` national arrays -> `[..., K, L]` values and availability."""
    state_names, national_names, locations = list(state_names), list(national_names), list(locations)
    us = locations.index('US') if 'US' in locations else None
    values = np.zeros((*state.shape[:-2], len(names), len(locations)), np.float32)
    for k, name in enumerate(names):
        if name in state_names:
            values[..., k, :] = state[..., state_names.index(name)]
        elif name in national_names:
            values[..., k, :] = np.nan
            if us is not None:
                values[..., k, us] = national[..., national_names.index(name)]
        else:
            raise ValueError(f'Unknown covariate: {name}')
    available = ~np.isnan(values)
    return np.nan_to_num(values), available


def _pad(array, before, after):
    return np.concatenate([np.full((before, *array.shape[1:]), np.nan, array.dtype), array,
                           np.full((after, *array.shape[1:]), np.nan, array.dtype)])


def _channel_first(panel):
    """`[.., L, C]` (the array-on-disk convention) -> `[.., C, L]` (Model's convention)."""
    return np.moveaxis(panel, -1, -2)


def episodes(panel, lookback, input_mode, covariate_names=(), horizons=HORIZONS):
    """All usable episodes of a (possibly masked) panel; see the module docstring."""
    if input_mode not in ('finalized', 'vintaged'):
        raise ValueError(f'Unknown input_mode: {input_mode}')
    metadata = json.loads(str(panel['metadata']))
    depth_targets, depth_covariates = metadata['asof_target_weeks'], metadata['asof_covariate_weeks']
    if input_mode == 'vintaged' and covariate_names and lookback > depth_covariates:
        raise ValueError(f'Vintaged lookback {lookback} exceeds the as-of covariate depth {depth_covariates}')
    dates = [str(d) for d in panel['dates']]
    first, locations = date.fromisoformat(dates[0]), tuple(str(l) for l in panel['locations'])
    pad, tail = lookback - 1, max(horizons)
    targets = _channel_first(_pad(panel['targets'], pad, tail))  # padded index = calendar index + pad
    if covariate_names:
        names = (panel['covariate_names'], panel['covariate_national_names'], locations, covariate_names)
        truth_cov = select_covariates(_pad(panel['covariates'], pad, tail),
                                      _pad(panel['covariates_national'], pad, tail), *names)
        if input_mode == 'vintaged':
            asof_cov = select_covariates(panel['asof_covariates'], panel['asof_covariates_national'], *names)
    week = lambda t: (first + timedelta(weeks=t)).isoformat()
    if input_mode == 'finalized':
        origins = [(t, None) for t in range(len(dates) - tail)]
    else:
        origins = [((date.fromisoformat(context_end(d)) - first).days // 7, w)
                   for w, d in enumerate(panel['issuance_dates'])]
    result = []
    for t, w in origins:
        if t < 0:
            continue  # context ends before the calendar: nothing is available
        context = slice(t, t + lookback)  # padded indices of calendar weeks t-lookback+1..t
        future = [t + pad + h for h in horizons]
        values = targets[context].copy()
        known_final = ~np.isnan(values)
        if w is not None:
            recent = min(depth_targets, lookback)
            asof = _channel_first(panel['asof_targets'][w, depth_targets - recent:])
            seen = ~np.isnan(asof)
            values[-recent:] = np.where(seen, asof, values[-recent:])
            known_final[-recent:] = ~seen & ~np.isnan(targets[context][-recent:])
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
        episode['Y'] = np.stack((episode['target_values'], episode['target_available']), axis=2)
        if covariate_names:
            if w is None:
                cov_values, cov_available = truth_cov[0][context], truth_cov[1][context]
            else:
                cov_values = asof_cov[0][w, depth_covariates - lookback:]
                cov_available = asof_cov[1][w, depth_covariates - lookback:]
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
