"""Baseline mathematics, unit invariance and future-vintage isolation."""
import numpy as np

from tapestry.model.baselinenowcast import delay_cdf, predictions, apply_delay


def test_delay_completion_preserves_signed_revisions_and_units():
    triangle = np.array([[10., 20., 15.], [20., 40., np.nan], [30., np.nan, np.nan]])
    np.testing.assert_allclose(delay_cdf(triangle), [2/3, 4/3, 1.])
    np.testing.assert_allclose(delay_cdf(triangle / 1000), delay_cdf(triangle))
    np.testing.assert_allclose(delay_cdf(triangle, integer=True), delay_cdf(triangle))
    assert delay_cdf(np.array([[0., np.nan]])) is None


def test_prediction_count_offset_fallback_and_future_vintages():
    dates = np.datetime64('2025-01-04') + np.arange(8)*np.timedelta64(7, 'D')
    issues = dates + np.timedelta64(4, 'D')
    asof = np.full((8, 8, 1), np.nan)
    for w in range(8):
        for t in range(w+1):
            asof[w, t, 0] = 10. if w == t else 20.
    panel = dict(dates=dates, issuance_dates=issues)
    rows = dict(issuance=np.array([str(issues[4])]*2), boundary=np.array([str(dates[4])]*2),
                age=np.array([0, 0]), location=np.array([0, 0]), lag=0,
                baseline_history=np.array([[10.], [np.nan]]))
    before = predictions(panel, asof, rows, np.array([10., 7.]), integer=True, max_delay=1)
    np.testing.assert_allclose(before[0], [20.5, 7.])
    assert list(before[1]) == ['estimated', 'missing_report']
    asof[5:] = 1e9
    after = predictions(panel, asof, rows, np.array([10., 7.]), integer=True, max_delay=1)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)


def test_count_expectation_applies_each_delay_sequentially():
    # Upstream: 10 + (10 + 1 - .5)/.5 * .25 = 15.25;
    # then 15.25 + (15.25 + 1 - .75)/.75 * .25 = 20 5/12.
    np.testing.assert_allclose(apply_delay(np.array([10.]), np.array([0]),
                                         np.array([.5, .75, 1.]), integer=True), [20 + 5/12])


def test_target_scoring_preserves_horizon_channel_location_alignment(tmp_path):
    import json
    from tapestry.experiment.nowcast_baseline import evaluate
    dates = np.datetime64('2025-01-04') + np.arange(16)*np.timedelta64(7, 'D')
    issues = dates + np.timedelta64(4, 'D')
    # Different values for every channel/location expose axis swaps.
    levels = np.arange(1, 13).reshape(2, 6) / 100
    asof = np.full((16, 16, 2, 6), np.nan)
    for w in range(16):
        asof[w, :w+1] = levels
    panel = dict(dates=dates, issuance_dates=issues, locations=np.array(['NC', 'US']),
                 target_names=np.array(list('abcdef')), asof_targets=asof)
    truth = np.stack([levels.T, levels.T])
    mask = np.ones_like(truth, dtype=bool)
    episode = dict(target_dates=dates[14:].astype(str).tolist(), issuance=str(issues[15]),
                   locations=('NC', 'US'), target_values=truth, target_available=mask,
                   Y=np.stack([truth, mask], axis=2))
    (tmp_path / 'nowcast_scores.json').write_text(json.dumps(dict(normalized_crps=0.)))
    evaluate(panel, [episode], np.ones((6, 2)), tmp_path)
    with np.load(tmp_path / 'baselinenowcast.npz') as saved:
        np.testing.assert_allclose(saved['prediction'][0], truth)
        assert (saved['status'] == 'estimated').all()
    scores = json.loads((tmp_path / 'nowcast_scores.json').read_text())
    assert scores['baselinenowcast_normalized_mae'] < 1e-12
