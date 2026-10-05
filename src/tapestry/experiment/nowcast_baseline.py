"""Matched point-baseline scoring for standalone and pipeline target nowcasts."""
import json

import numpy as np

from tapestry.model.baselinenowcast import predictions
from tapestry.model.objective import loss_cell_weights
from .provenance import save


def evaluate(panel, eps, scales, output):
    dates = panel['dates'].astype('datetime64[D]')
    issues = panel['issuance_dates'].astype('datetime64[D]')
    locations = np.asarray(eps[0]['locations'])
    loc_indices = np.array([list(panel['locations']).index(loc) for loc in locations])
    targets = np.asarray([e['target_dates'] for e in eps], dtype='datetime64[D]')
    issuances = np.asarray([e['issuance'] for e in eps], dtype='datetime64[D]')
    wi, ti = np.searchsorted(issues, issuances), np.searchsorted(dates, targets)
    if not np.array_equal(issues[wi], issuances) or not np.array_equal(dates[ti], targets):
        raise ValueError('Baseline episode dates must match the vintage panel')
    n, h = targets.shape
    l = len(locations)
    age = ((issuances[:, None] - targets).astype(int) - 4) // 7
    rows = dict(issuance=np.repeat(issuances[:, None], h*l, axis=1).ravel().astype(str),
                boundary=np.repeat(targets[..., None], l, axis=2).ravel().astype(str),
                age=np.repeat(age[..., None], l, axis=2).ravel(),
                location=np.tile(np.arange(l), n*h), locations=locations, lag=0)
    output_values, statuses, supports = [], [], []
    for c in range(len(panel['target_names'])):
        asof = panel['asof_targets'][..., c][..., loc_indices]
        past = ti[..., None] + np.arange(-11, 1)
        history = asof[wi[:, None, None], past.clip(0)]
        history = np.where((past >= 0)[..., None], history, np.nan)
        history = history.transpose(0, 1, 3, 2).reshape(-1, 12)
        last = np.where(np.isfinite(history), np.arange(12), -1).max(1)
        anchor = np.where(last >= 0, history[np.arange(len(history)), last.clip(0)], 0.)
        rows['baseline_history'] = history
        point, status, support = predictions(panel, asof, rows, anchor, integer=c < 3)
        if c >= 3:
            point = np.minimum(point, 1.)
        output_values.append(point.reshape(n, h, l))
        statuses.append(status.reshape(n, h, l))
        supports.append(support.reshape(n, h, l))
    point = np.stack(output_values, axis=2)
    status, support = np.stack(statuses, axis=2), np.stack(supports, axis=2)
    truth = np.stack([e['target_values'] for e in eps])
    mask = np.stack([e['target_available'] for e in eps]).astype(bool)
    weights = loss_cell_weights(eps)
    score = float((np.where(mask, np.abs(point-truth), 0.) * weights / scales).sum())
    np.savez_compressed(output / 'baselinenowcast.npz', prediction=point, status=status, support=support,
                        truth=truth, mask=mask, target_dates=targets, issuance=issuances, locations=locations)
    scores = json.loads((output / 'nowcast_scores.json').read_text())
    scores.update(baselinenowcast_normalized_mae=score,
                  baselinenowcast_point_crps_ratio=scores['normalized_crps']/score if score > 0 else None,
                  baselinenowcast_status_counts={s: int(((status == s) & mask).sum()) for s in np.unique(status)},
                  baseline='baselinenowcast_point_v2; deterministic CRPS equals MAE; no uncertainty model')
    save(output / 'nowcast_scores.json', scores)
