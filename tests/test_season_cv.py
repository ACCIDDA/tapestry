from datetime import date, timedelta

import numpy as np
import pytest
pytest.importorskip('torch')

from influpaintx.model_data import FinalizedDataset
from influpaintx.model_data.finalized import season
from influpaintx.models.season_cv import LEVELS, SEASONS, fold_data, wis


def test_holdout_cannot_change_fit_data_or_scales():
    days = tuple((date(2023, 9, 2) + timedelta(weeks=i)).isoformat() for i in range(157))
    panel = np.ones((157, 6, 2, 2), dtype=np.float32)
    ds = FinalizedDataset(panel, days, ('AL', 'US'), {})
    held_out = SEASONS[1]
    train, evaluation, scales = fold_data(ds, held_out)
    changed = panel.copy()
    for i, day in enumerate(days):
        if season(date.fromisoformat(day)) == held_out:
            changed[i, :, 0] = 12345
    other, _, other_scales = fold_data(FinalizedDataset(changed, days, ds.locations, {}), held_out)
    assert scales == other_scales
    assert len(train) == len(other)
    for a, b in zip(train, other):
        np.testing.assert_array_equal(a['X'], b['X'])
        np.testing.assert_array_equal(a['Y'], b['Y'])
        for i, day in enumerate(a['context_dates']):
            if season(date.fromisoformat(day)) == held_out:
                assert not a['X'][i].any()
        for i, day in enumerate(a['target_dates']):
            if season(date.fromisoformat(day)) == held_out:
                assert not a['Y'][i].any()
    origins = [date.fromisoformat(e['context_dates'][-1]) for e in evaluation]
    assert all((b - a).days == 7 for a, b in zip(origins, origins[1:]))
    for e in evaluation:
        assert e['Y'].shape == (4, 6, 2, 2)
        for i, day in enumerate(e['target_dates']):
            if season(date.fromisoformat(day)) != held_out:
                assert not e['Y'][i].any()


def test_wis_equals_independent_pinball_calculation():
    rng = np.random.default_rng(5)
    q = np.sort(rng.normal(size=(23, 10)), axis=0)
    y = rng.normal(size=10)
    error = y - q
    pinball = np.maximum(LEVELS[:, None] * error, (LEVELS[:, None] - 1) * error)
    np.testing.assert_allclose(wis(q, y), 2 * pinball.mean(0))
    # Degenerate last-value distributions have WIS equal to absolute error.
    flat = np.full((23, 1), 3.)
    np.testing.assert_allclose(wis(flat, np.array([5.])), 2)
