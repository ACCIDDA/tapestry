"""The hub's 23-quantile grid for saved predictions and evaluation.

Moved unchanged from `models/quantiles.py`.
"""
import numpy as np

LEVELS = np.array([.01, .025, .05, .10, .15, .20, .25, .30, .35, .40, .45, .50,
                   .55, .60, .65, .70, .75, .80, .85, .90, .95, .975, .99])

