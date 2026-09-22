"""A small synthetic panel in the `dataset.build` layout, shared by dataset and planner tests."""
from datetime import date, timedelta
import json

import numpy as np
import pytest

from tapestry.dataset.build import (CHANNELS, STATE_COVARIATE_NAMES, NATIONAL_COVARIATE_NAMES, overlay_dates,
                                    wednesdays)

LOCATIONS = ('NC', 'US')
R, D = 2, 16


def code(day):
    """A distinct positive number per reference week, so any value can be traced back to its week."""
    return float((date.fromisoformat(str(day)) - date(2023, 1, 7)).days // 7 + 1)


def synthetic_panel(n_weeks=3 * 52 + 10):
    """Every value encodes its reference week (and channel/covariate/location), as-of = truth + .5."""
    first = date(2023, 9, 2)
    dates = [(first + timedelta(weeks=i)).isoformat() for i in range(n_weeks)]
    weeks = np.array([code(d) for d in dates])
    targets = (weeks[:, None, None] + 1000 * np.arange(1, len(CHANNELS) + 1)[None, None, :]
               + 100000 * np.arange(len(LOCATIONS))[None, :, None]).astype(np.float32)
    covariates = (weeks[:, None, None] + 1000 * np.arange(1, len(STATE_COVARIATE_NAMES) + 1)[None, None, :]
                  + 100000 * np.arange(len(LOCATIONS))[None, :, None]).astype(np.float32)
    national = (weeks[:, None] + 50000.).astype(np.float32)
    issuances = wednesdays(dates[0], dates[-1])
    lookup = {d: i for i, d in enumerate(dates)}

    def overlay(truth, depth):
        out = np.full((len(issuances), depth, *truth.shape[1:]), np.nan, np.float32)
        for w, issuance in enumerate(issuances):
            for j, day in enumerate(overlay_dates(issuance, depth)):
                if day in lookup:
                    out[w, j] = truth[lookup[day]] + .5
        return out

    asof_targets = overlay(targets, R)
    asof_targets[::3, -1] = np.nan  # some latest weeks not yet reported: fall back to truth, known-final
    return dict(dates=np.array(dates, dtype='datetime64[D]'), locations=np.array(LOCATIONS),
                target_names=np.array(CHANNELS), targets=targets,
                covariate_names=np.array(STATE_COVARIATE_NAMES), covariates=covariates,
                covariate_mask=np.ones_like(covariates, dtype=bool),
                covariate_national_names=np.array(NATIONAL_COVARIATE_NAMES), covariates_national=national,
                issuance_dates=np.array(issuances, dtype='datetime64[D]'), asof_targets=asof_targets,
                asof_covariates=overlay(covariates, D), asof_covariates_national=overlay(national, D),
                metadata=json.dumps(dict(asof_target_weeks=R, asof_covariate_weeks=D)))


@pytest.fixture
def panel():
    return synthetic_panel()
