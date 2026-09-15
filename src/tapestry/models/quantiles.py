"""The hub's 23-quantile grid for saved predictions and evaluation."""
import numpy as np

LEVELS = np.array([.01, .025, .05, .10, .15, .20, .25, .30, .35, .40, .45, .50,
                   .55, .60, .65, .70, .75, .80, .85, .90, .95, .975, .99])


def select_quantiles(values, levels):
    """Select exact saved levels, including from historical archives."""
    levels = np.asarray(levels)
    indices = []
    for level in LEVELS:
        matches = np.flatnonzero(np.isclose(levels, level, rtol=0, atol=1e-10))
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one saved quantile at {level}')
        indices.append(matches[0])
    return values[indices]
