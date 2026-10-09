"""Interval-width calibration fitted on held-out weeks inside the training seasons.

One spread factor s per flu target and horizon rescales every quantile around the
median in native units: q' = median + s (q - median), then counts are clipped at 0 and
rounded as in `training.evaluate`, and ED proportions are clipped to [0, 1]. s is chosen on
a log grid from 0.5 to 4 to minimize native-unit WIS on the calibration forecasts. Each
location's WIS is divided by its own uncalibrated (s = 1) WIS, mirroring the
location-relative Hub score. States/DC share 80% of the weight equally and the US gets 20%.
Center shifts are not fitted."""
import numpy as np
from .quantiles import LEVELS
from .standard import quantile_scores

GRID = np.exp(np.linspace(np.log(.5), np.log(4), 61))
FLU_CHANNELS = (0, 3)


def rescale(q, c, s):
    """q [levels, ...] of channel c rescaled about its median by s."""
    k = len(LEVELS) // 2
    x = q[k][None] + s * (q - q[k][None])
    if c < 3:
        return np.floor(np.clip(x, 0, None) + .5)
    return np.clip(x, 0, 1)


def apply(q, factors):
    """q [levels, n, horizon, channel, location]; factors {channel: [s per horizon]}."""
    out = q.copy()
    for c, per_horizon in factors.items():
        for h, s in enumerate(per_horizon):
            out[:, :, h, int(c)] = rescale(q[:, :, h, int(c)], int(c), s)
    return out


def fit(q, y, mask, locations):
    """Spread factors from calibration forecasts q [levels, n, h, C, L], truth y/mask [n, h, C, L]."""
    locations = [str(v) for v in locations]
    n_states = sum(loc != 'US' for loc in locations)
    weight = np.array([.2 if loc == 'US' else .8 / n_states for loc in locations])
    factors, record = {}, {}
    for c in FLU_CHANNELS:
        base = np.full(len(locations), np.nan)
        for li in range(len(locations)):
            ok = mask[:, :, c, li].astype(bool)
            if ok.any():
                base[li] = quantile_scores(q[:, :, :, c, li][:, ok].T, y[:, :, c, li][ok]).wis.mean()
        factors[c], record[c] = [], []
        for h in range(q.shape[2]):
            usable = [li for li in range(len(locations)) if base[li] > 0 and mask[:, h, c, li].any()]
            if not usable:
                factors[c].append(1.)
                record[c].append(dict(horizon=h, cells=0, factor=1.))
                continue
            w = weight[usable] / weight[usable].sum()
            objective = []
            for s in GRID:
                total = 0.
                for wi, li in zip(w, usable):
                    ok = mask[:, h, c, li].astype(bool)
                    pred = rescale(q[:, ok, h, c, li], c, s)
                    total += wi * quantile_scores(pred.T, y[ok, h, c, li]).wis.mean() / base[li]
                objective.append(total)
            best = int(np.argmin(objective))
            one = int(np.argmin(abs(GRID - 1)))
            factors[c].append(float(GRID[best]))
            record[c].append(dict(horizon=h, cells=int(mask[:, h, c, usable].sum()), factor=float(GRID[best]),
                                  objective=float(objective[best]), objective_at_one=float(objective[one])))
    return factors, record
