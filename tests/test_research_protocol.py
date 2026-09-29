"""Scientific leakage and data-policy checks for complete-history B2 learning."""
import numpy as np
from tapestry.dataset.cv import fold, week_roles, SEASONS, season
from tapestry.model.scenario import Scenario
from tapestry.experiment.training import model_options
from test_dataset import perturbed


def recipe():
    return Scenario(input_mode='vintaged', asof_weeks=12, training_inputs='finalized',
                    input_normalization='b0', validation_calendar='b0', epochs=300, patience=30,
                    covariate_set='kinsa+ilinet', supplied_final=True)


def test_complete_training_and_vintage_evaluation(panel):
    s = recipe()
    f = fold(panel, s, '2025-2026')
    for e in f.train:
        assert np.array_equal(e['known_final'], e['available'])
        for i, day in enumerate(e['context_dates']):
            if day not in panel['dates'].astype(str):
                continue
            t = list(panel['dates'].astype(str)).index(day)
            a = e['available'][i]
            np.testing.assert_array_equal(e['values'][i][a], panel['targets'][t].T[a])
    for e in f.score:
        assert not e['known_final'].any()
        w = list(panel['issuance_dates'].astype(str)).index(e['issuance'])
        for i, day in enumerate(e['context_dates']):
            if day not in panel['dates'].astype(str):
                continue
            t = list(panel['dates'].astype(str)).index(day)
            expected = panel['asof_targets'][w, t].T
            np.testing.assert_array_equal(e['available'][i], np.isfinite(expected))
            np.testing.assert_array_equal(e['values'][i], np.nan_to_num(expected))
    # Kinsa is final in fitting, genuinely vintage at evaluation, and broadcast.
    for episodes, vintage in [(f.train, False), (f.score, True)]:
        for e in episodes:
            for i, day in enumerate(e['context_dates']):
                if day not in panel['dates'].astype(str):
                    continue
                t = list(panel['dates'].astype(str)).index(day)
                if vintage:
                    w = list(panel['issuance_dates'].astype(str)).index(e['issuance'])
                    expected = panel['asof_covariates_national'][w, t, 0]
                else:
                    expected = panel['covariates_national'][t, 0]
                if e['covariates'][i, 0, 1].any():
                    np.testing.assert_array_equal(e['covariates'][i, 0, 0], np.repeat(expected, 2))


def test_heldout_and_validation_values_cannot_change_fit_or_scales(panel):
    s = recipe()
    for held in SEASONS:
        roles = week_roles(panel['dates'], s, held)
        hidden = set(panel['dates'].astype(str)[roles != 'fit'])
        a = fold(panel, s, held, inner=True)
        b = fold(perturbed(panel, hidden), s, held, inner=True)
        for x, y in zip(a.train, b.train):
            for key in ('values', 'available', 'known_final', 'target_values', 'target_available', 'covariates'):
                np.testing.assert_array_equal(x[key], y[key])
        pop = {'NC': 1e7, 'US': 3.3e8}
        assert model_options(a.train, s, pop) == model_options(b.train, s, pop)
        assert all(not e['known_final'].any() for e in a.validation)


def test_b0_calendar_counts_from_first_stored_week(panel):
    s = recipe()
    dates = panel['dates'].astype(str)
    roles = week_roles(dates, s, '2025-2026')
    for label in ('2023-2024', '2024-2025'):
        ids = np.flatnonzero(np.array([season(d) for d in dates]) == label)
        expected = (np.arange(len(ids)) % 16 >= 4) & (np.arange(len(ids)) % 16 < 7)
        np.testing.assert_array_equal(roles[ids] == 'validation', expected)
