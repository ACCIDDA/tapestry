"""Scientific checks for the assumed publication lags and fold isolation."""
from dataclasses import replace
import numpy as np
from tapestry.dataset.episodes import episodes
from tapestry.dataset.cv import fold, season, week_roles
from tapestry.dataset.build import covariate_names_for
from tapestry.model.scenario import Scenario


def test_schedule_uses_final_values_at_correct_observation_dates(panel):
    names = covariate_names_for('inpatient+outpatient+kinsa+ilinet+clinical_lab+flusurv')
    final = episodes(panel, 12, 'finalized', names)
    scheduled = episodes(panel, 12, 'scheduled_final', names)
    by_date = {e['context_dates'][-1]: e for e in final}
    for e in scheduled:
        baseline = by_date[e['context_dates'][-1]]
        np.testing.assert_array_equal(e['available'], baseline['available'])
        np.testing.assert_array_equal(e['values'], baseline['values'])
        np.testing.assert_array_equal(e['known_final'], e['available'])
        np.testing.assert_array_equal(e['Y'], baseline['Y'])
        for k, name in enumerate(names):
            if name in ('ilinet_ili', 'clinical_lab_flu_pct_positive', 'flusurv_flu_rate'):
                assert not e['covariates'][-1, k].any()
                np.testing.assert_array_equal(e['covariates'][:-1, k], baseline['covariates'][:-1, k])
            else:
                np.testing.assert_array_equal(e['covariates'][:, k], baseline['covariates'][:, k])


def test_two_fold_schedule_excludes_heldout_values_from_fit(panel):
    from conftest import synthetic_panel
    panel = synthetic_panel(n_weeks=4 * 52 + 10)
    panel['dates'] = panel['dates'] - np.timedelta64(364, 'D')
    panel['issuance_dates'] = panel['issuance_dates'] - np.timedelta64(364, 'D')
    s = Scenario(input_mode='scheduled_final', evaluation_seasons='recent_two', patience=2, epochs=4,
                 covariate_set='kinsa+ilinet')
    assert s.scored_seasons == ('2025-2026', '2024-2025')
    for held in s.scored_seasons:
        roles = week_roles(panel['dates'], s, held)
        changed = dict(panel)
        for name in ('targets', 'covariates', 'covariates_national'):
            changed[name] = panel[name].copy()
            changed[name][~np.isin(roles, ['fit'])] = 999
        a, b = fold(panel, s, held, inner=True), fold(changed, s, held, inner=True)
        assert len(a.train) == len(b.train)
        for x, y in zip(a.train, b.train):
            for name in ('values', 'available', 'known_final', 'covariates', 'Y'):
                np.testing.assert_array_equal(x[name], y[name])


def test_multiscale_slopes_curvature_and_masked_values():
    import torch
    from tapestry.model.covariates import multiscale_features
    t = torch.arange(-11., 1.)
    x = (2 + 3*t + .5*t*t)[None, :, None, None]
    mask = torch.ones_like(x, dtype=torch.bool)
    features = multiscale_features(x, mask).reshape(3, 6)
    torch.testing.assert_close(features[:, 2], torch.ones(3), atol=2e-5, rtol=2e-5)
    linear = (2 + 3*t)[None, :, None, None]
    mask[:, -1] = False
    changed = linear.clone()
    changed[~mask] = 1e9
    a = multiscale_features(linear, mask).reshape(3, 6)
    b = multiscale_features(changed, mask).reshape(3, 6)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a[:, 1], torch.full((3,), 3.))
    torch.testing.assert_close(multiscale_features(changed, torch.zeros_like(mask)), torch.zeros((1, 1, 18)))
