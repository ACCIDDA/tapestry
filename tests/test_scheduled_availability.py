"""Scientific checks for the assumed publication lags and fold isolation."""
import numpy as np
from tapestry.dataset.episodes import episodes
from tapestry.dataset.cv import fold, week_roles
from tapestry.dataset.build import covariate_names_for
from tapestry.model.scenario import Scenario


def test_schedule_uses_final_values_at_correct_observation_dates(panel):
    from tapestry.dataset.episodes import select_covariates
    names = covariate_names_for('inpatient+outpatient+kinsa+ilinet+clinical_lab+flusurv')
    dates = list(panel['dates'].astype(str))
    locations = list(panel['locations'].astype(str))
    values, available = select_covariates(panel['covariates'], panel['covariates_national'], panel['covariate_names'],
                                          panel['covariate_national_names'], locations, names)
    checked = 0
    for e in episodes(panel, 12, 'scheduled_final', names):
        np.testing.assert_array_equal(e['known_final'], e['available'])
        for i, day in enumerate(e['context_dates']):
            if day not in dates:
                assert not e['available'][i].any()
                continue
            t = dates.index(day)
            truth = np.moveaxis(panel['targets'][t], -1, -2)
            np.testing.assert_array_equal(e['available'][i], ~np.isnan(truth))  # all six targets through T-0
            np.testing.assert_array_equal(e['values'][i], np.nan_to_num(truth))
            for k, name in enumerate(names):
                lagged = name in ('ilinet_ili', 'clinical_lab_flu_pct_positive', 'flusurv_flu_rate') and i == len(e['context_dates']) - 1
                expected = available[t, k] & ~lagged
                np.testing.assert_array_equal(e['covariates'][i, k, 1].astype(bool), expected)
                np.testing.assert_array_equal(e['covariates'][i, k, 0], np.where(expected, values[t, k], 0))
        checked += 1
    assert checked > 100


def test_two_fold_schedule_excludes_heldout_values_from_fit():
    from conftest import synthetic_panel
    panel = synthetic_panel(n_weeks=4 * 52 + 10)
    panel['dates'] = panel['dates'] - np.timedelta64(364, 'D')
    panel['issuance_dates'] = panel['issuance_dates'] - np.timedelta64(364, 'D')
    s = Scenario(evaluation_seasons='recent_two', patience=2, epochs=4,
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


def test_reported_inputs_use_reports_and_star_only_unarchived_cells(panel):
    """Standard evaluation inputs: reports where archived, finalized only where nothing was, labels final."""
    names = covariate_names_for('inpatient+kinsa+ilinet')
    final = {e['context_dates'][-1]: e for e in episodes(panel, 12, 'scheduled_final', names)}
    dates = list(panel['dates'].astype(str))
    reported = episodes(panel, 12, 'reported', names)
    assert reported and any(e['filled'].any() for e in reported)
    for e in reported:
        base = final.get(e['context_dates'][-1])
        if base is not None:  # finalized episodes stop once labels leave the calendar
            np.testing.assert_array_equal(e['Y'], base['Y'])  # labels are never reports
        w = list(panel['issuance_dates'].astype(str)).index(e['issuance'])
        for i, day in enumerate(e['context_dates']):
            if day not in dates:
                continue
            asof = np.moveaxis(panel['asof_targets'][w, dates.index(day)], -1, -2)
            truth = np.moveaxis(panel['targets'][dates.index(day)], -1, -2)
            np.testing.assert_array_equal(e['filled'][i], np.isnan(asof))
            np.testing.assert_array_equal(e['values'][i], np.where(np.isnan(asof), truth, asof))
        np.testing.assert_array_equal(e['known_final'], e['available'])
        for k, name in enumerate(names):
            if name == 'ilinet_ili':  # T-1 source: newest week never available, even if archived
                assert not e['covariates'][-1, k, 1].any()
        assert not (e['covariates_filled'] & ~e['covariates'][..., 1, :].astype(bool)).any()


def test_hub_deadlines_follow_each_hub_holiday_schedule(panel):
    from tapestry.dataset.build import deadline, for_hub
    eastern = lambda hub, wednesday: deadline(wednesday, hub).strftime('%Y-%m-%d %H:%M %Z')
    assert eastern('flusight', '2024-12-25') == '2024-12-26 23:00 EST'
    assert eastern('covid', '2024-12-25') == '2024-12-26 23:00 EST'
    assert eastern('rsv', '2024-12-25') == '2024-12-25 23:00 EST'  # no RSV Hub in 2024-25
    assert eastern('flusight', '2025-12-24') == '2025-12-30 23:00 EST'
    assert eastern('covid', '2025-12-24') == '2025-12-29 23:00 EST'
    assert eastern('flusight', '2025-12-31') == '2026-01-05 23:00 EST'
    assert eastern('rsv', '2025-12-31') == '2026-01-04 23:00 EST'
    assert eastern('covid', '2025-10-15') == '2025-10-15 23:00 EDT'
    covid = for_hub(panel, 'covid')
    rows = panel['hub_covid_issuances']
    assert len(rows)
    others = np.setdiff1d(np.arange(len(panel['issuance_dates'])), rows)
    np.testing.assert_array_equal(covid['asof_targets'][others], panel['asof_targets'][others])
    np.testing.assert_array_equal(covid['asof_targets'][rows], panel['hub_covid_asof_targets'])
    assert not np.array_equal(covid['asof_targets'][rows], panel['asof_targets'][rows], equal_nan=True)


def test_instant_cutoffs_treat_date_labels_as_whole_days():
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from tapestry.dataset.extract import cutoff_time, end_of_label_day, last_report_day
    deadline = datetime(2025, 12, 24, 23, tzinfo=ZoneInfo('America/New_York'))  # Thursday 04:00 UTC
    labels = end_of_label_day(np.array(['2025-12-24', '2025-12-25', '2025-12-25T03:00'], 'datetime64[ns]'))
    assert list(labels <= cutoff_time(deadline)) == [True, False, True]
    assert last_report_day(deadline) == '2025-12-24'
    assert cutoff_time('2025-12-24') == np.datetime64('2025-12-24T23:59:59.999999', 'ns')
