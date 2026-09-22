"""Native-unit loss normalization and explicit season/target/location weights.

Unchanged fair-CRPS loss weighting (docs/design/restructure-2026-unified.md §6:
"No change to the fair-CRPS loss ... only where that logic lives").
"""
import numpy as np

from tapestry.dataset.build import CHANNELS
from tapestry.dataset.cv import season
from tapestry.evaluation.totals import ADMISSIONS_WEIGHT, ED_WEIGHT, US_SCORE_WEIGHT

# Loss defaults = the score's default weights (defined once in evaluation/totals.py).
# They are separate choices: the loss's channel weights are a scenario option
# (`loss_weights`), the score's weights rank-time options.
TARGET_WEIGHTS = (ADMISSIONS_WEIGHT,) * 3 + (ED_WEIGHT,) * 3  # CHANNELS order: 3 admissions, 3 ED
US_WEIGHT = US_SCORE_WEIGHT
SCALE_MIN_WEEKS = 26
SCALE_FLOORS = (1., 1., 1., .001, .001, .001)
LOSS_DEFINITION = ('Native-unit fair CRPS / training channel-location Q95; equal seasons by target date; '
                   'within season normalize available channel weights; states/DC 80% equally, US 20%. '
                   'Q95 pools toward channel Q95 below 26 observed weeks; floors 1 admission/.001 ED. '
                   'Absent geography groups are renormalized over available groups. '
                   'Training normalization is a surrogate, not ensemble-relative WIS.')

LOSS_WEIGHTS = {
    'influenza_first': [1, .1, .1, .1, .1, .1],
    'balanced_admissions': [1, 1, 1, .1, .1, .1],
    'flu_only': [1, 0, 0, 0, 0, 0],
    'objective': list(TARGET_WEIGHTS),
}


def loss_scales(panel):
    """[C,L] native Q95 from unique permitted weeks [T,C,value_or_mask,L].

    Below 26 observed weeks use alpha*local + (1-alpha)*pooled, alpha=n/26.
    A floor protects constant-zero series. Input transforms are irrelevant here.
    """
    scales = np.empty((len(CHANNELS), panel.shape[-1]), dtype=float)
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
    location. Average valid dates/horizons within each location.
    """
    mask = np.stack([e['Y'][:, :, 1, :].astype(bool) for e in episodes])
    dates = np.array([[season(day) for day in e['target_dates']] for e in episodes])
    locations = np.asarray(episodes[0]['locations'])
    channel_weights = np.asarray(channel_weights, dtype=float)
    if channel_weights.shape != (len(CHANNELS),) or np.any(channel_weights < 0) or not np.isfinite(channel_weights).all():
        raise ValueError(f'Need {len(CHANNELS)} finite nonnegative channel weights')
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
