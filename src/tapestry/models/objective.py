"""B0 native-unit loss normalization and explicit season/target/location weights."""
from datetime import date

import numpy as np

from tapestry.model_data.finalized import season

TARGET_WEIGHTS = (1., 1., 1., .5, .5, .5)
US_WEIGHT = .20
SCALE_MIN_WEEKS = 26
SCALE_FLOORS = (1., 1., 1., .001, .001, .001)
LOSS_DEFINITION = ('Native-unit fair CRPS / training channel-location Q95; equal seasons by target date; '
                   'within season normalize available channel weights; states/DC 80% equally, US 20%. '
                   'Q95 pools toward channel Q95 below 26 observed weeks; floors 1 admission/.001 ED. '
                   'Absent geography groups are renormalized over available groups. '
                   'Training normalization is a surrogate, not ensemble-relative WIS.')


def loss_scales(panel):
    """[C,L] native Q95 from unique permitted weeks [T,C,value_or_mask,L].

    Below 26 observed weeks use alpha*local + (1-alpha)*pooled, alpha=n/26.
    A floor protects constant-zero series. Input transforms are irrelevant here.
    """
    scales = np.empty((6, panel.shape[-1]), dtype=float)
    for c, floor in enumerate(SCALE_FLOORS):
        values, valid = panel[:, c, 0], panel[:, c, 1].astype(bool)
        pooled = float(np.quantile(values[valid], .95)) if valid.any() else floor
        for l in range(panel.shape[-1]):
            observed = values[:, l][valid[:, l]]
            local = float(np.quantile(observed, .95)) if observed.size else pooled
            alpha = min(observed.size / SCALE_MIN_WEEKS, 1.)
            scales[c, l] = max(alpha * local + (1 - alpha) * pooled, floor)
    return scales.tolist()


def loss_cell_weights(episodes, channel_weights=TARGET_WEIGHTS):
    """Fixed [N,H,C,L] coefficients summing to one over the whole partition.

    Group by target-date season, then eligible channels, then geography and
    location. Average valid dates/horizons within each location. Computing this
    once avoids random minibatch missingness changing the objective.
    """
    mask = np.stack([e['Y'][:, :, 1, :].astype(bool) for e in episodes])
    dates = np.array([[season(date.fromisoformat(day)) for day in e['target_dates']] for e in episodes])
    locations = np.asarray(episodes[0]['locations'])
    channel_weights = np.asarray(channel_weights, dtype=float)
    if channel_weights.shape != (6,) or np.any(channel_weights < 0) or not np.isfinite(channel_weights).all():
        raise ValueError('Need six finite nonnegative channel weights')
    result = np.zeros(mask.shape, dtype=np.float32)
    seasons_used = 0
    for label in np.unique(dates):
        selected = mask & (dates == label)[:, :, None, None]
        counts = selected.sum(axis=(0, 1))
        available = counts.any(axis=1)
        weights = channel_weights * available
        if not weights.sum():
            continue
        weights = weights / weights.sum()
        seasons_used += 1
        for c, weight in enumerate(weights):
            if not weight:
                continue
            states = (locations != 'US') & (counts[c] > 0)
            national = (locations == 'US') & (counts[c] > 0)
            groups = [(states, 1 - US_WEIGHT), (national, US_WEIGHT)]
            group_total = sum(w for locs, w in groups if locs.any())
            for locs, group_weight in groups:
                if locs.any():
                    per_location = np.zeros(len(locations))
                    per_location[locs] = weight * group_weight / group_total / locs.sum() / counts[c, locs]
                    result[:, :, c] += selected[:, :, c] * per_location
    if not seasons_used:
        raise ValueError('No supervised cells with positive objective weight')
    return result / seasons_used
