"""The shared five-quantile grid for saved predictions and evaluation."""
import numpy as np

LEVELS = np.array([.025, .25, .5, .75, .975])


def select_quantiles(values, levels):
    """Select exact saved levels, including from historical 23-level archives."""
    levels = np.asarray(levels)
    indices = []
    for level in LEVELS:
        matches = np.flatnonzero(np.isclose(levels, level, rtol=0, atol=1e-10))
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one saved quantile at {level}')
        indices.append(matches[0])
    return values[indices]
