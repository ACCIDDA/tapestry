"""A small synthetic panel in the `dataset.build` layout, shared by dataset and planner tests."""
from datetime import date, timedelta
import json

import numpy as np
import pytest

from tapestry.dataset.build import (CHANNELS, STATE_COVARIATE_NAMES, NATIONAL_COVARIATE_NAMES, visible_weeks,
                                    wednesdays, deadline, utc, HUBS)

LOCATIONS = ('NC', 'US')


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
    visible = visible_weeks(dates, issuances)

    def asof(truth):
        shape = visible.shape + (1,) * (truth.ndim - 1)
        return np.where(visible.reshape(shape), truth[None] + .5, np.nan).astype(np.float32)

    asof_targets = asof(targets)
    for w in range(0, len(issuances), 3):  # some latest weeks not yet reported: fall back to truth, known-final
        if visible[w].any():
            asof_targets[w, visible[w].sum() - 1] = np.nan
    # Other Hubs' own-deadline rows (holiday extensions differ): as-of = truth + .25 there.
    hubs = {}
    for hub in HUBS[1:]:
        rows = np.array([w for w, i in enumerate(issuances) if deadline(i, hub) != deadline(i, HUBS[0])], int)
        hubs[f'hub_{hub}_issuances'] = rows
        for name, truth in (('targets', targets), ('covariates', covariates), ('covariates_national', national)):
            hubs[f'hub_{hub}_asof_{name}'] = (asof(truth)[rows] - .25).astype(np.float32)
    return dict(dates=np.array(dates, dtype='datetime64[D]'), locations=np.array(LOCATIONS),
                target_names=np.array(CHANNELS), targets=targets,
                covariate_names=np.array(STATE_COVARIATE_NAMES), covariates=covariates,
                covariate_national_names=np.array(NATIONAL_COVARIATE_NAMES), covariates_national=national,
                issuance_dates=np.array(issuances, dtype='datetime64[D]'), asof_targets=asof_targets,
                asof_covariates=asof(covariates), asof_covariates_national=asof(national),
                hub_names=np.array(HUBS), issuance_cutoffs_utc=np.array([[utc(deadline(i, h)) for i in issuances] for h in HUBS]),
                **hubs, metadata=json.dumps(dict(truth_day=dates[-1])))


@pytest.fixture
def panel():
    return synthetic_panel()
