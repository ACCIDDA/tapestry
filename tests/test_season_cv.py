from collections import Counter
from datetime import date, timedelta

import numpy as np
import pytest
pytest.importorskip('torch')

from tapestry.model_data import FinalizedDataset
from tapestry.model_data.finalized import season
from tapestry.models.season_cv import LEVELS, SEASONS, fold_data, validation_split, wis

DAYS = tuple((date(2023, 9, 2) + timedelta(weeks=i)).isoformat() for i in range(157))


def test_holdout_cannot_change_fit_data_or_scales():
    panel = np.ones((157, 6, 2, 2), dtype=np.float32)
    ds = FinalizedDataset(panel, DAYS, ('AL', 'US'), {})
    held_out = SEASONS[1]
    train, evaluation, scales = fold_data(ds, held_out)
    changed = panel.copy()
    for i, day in enumerate(DAYS):
        if season(date.fromisoformat(day)) == held_out:
            changed[i, :, 0] = 12345
    other, _, other_scales = fold_data(FinalizedDataset(changed, DAYS, ds.locations, {}), held_out)
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


@pytest.mark.parametrize('held_out', SEASONS)
def test_validation_weeks_are_hidden_from_the_inner_fit(held_out):
    panel = np.ones((157, 6, 2, 2), dtype=np.float32)
    ds = FinalizedDataset(panel, DAYS, ('AL', 'US'), {})
    inner, validation, scales, info = validation_split(ds, held_out)
    hidden = set(info['validation_weeks'])
    training = [s for s in SEASONS if s != held_out]
    labels = [season(date.fromisoformat(day)) for day in DAYS]
    for label in SEASONS:
        share = sum(season(date.fromisoformat(d)) == label for d in hidden) / labels.count(label)
        assert share == 0 if label == held_out else .15 < share < .2
    # Hidden weeks come in runs of at most three consecutive weeks.
    ordered = sorted(date.fromisoformat(d) for d in hidden)
    runs = [1]
    for a, b in zip(ordered, ordered[1:]):
        runs[-1:] = [runs[-1] + 1] if (b - a).days == 7 else [runs[-1], 1]
    assert max(runs) == 3 and info['validation_target_weeks'] == len(hidden)
    changed = panel.copy()
    for i, day in enumerate(DAYS):
        if day in hidden or season(date.fromisoformat(day)) not in training:
            changed[i, :, 0] = 12345
    other_inner, other_validation, other_scales, _ = validation_split(FinalizedDataset(changed, DAYS, ds.locations, {}), held_out)
    assert scales == other_scales and len(inner) == len(other_inner)
    for a, b in zip(inner, other_inner):
        np.testing.assert_array_equal(a['X'], b['X'])
        np.testing.assert_array_equal(a['Y'], b['Y'])
        for dates, values in ((a['context_dates'], a['X']), (a['target_dates'], a['Y'])):
            for day, value in zip(dates, values):
                if day in hidden:
                    assert not value.any()
    # Validation scores exactly the hidden weeks, each at all four horizons.
    scored = Counter(day for e in validation for day, value in zip(e['target_dates'], e['Y']) if value[:, 1].any())
    assert set(scored) == hidden and set(scored.values()) == {4}
    assert any(not np.array_equal(a['Y'], b['Y']) for a, b in zip(validation, other_validation))


def test_wis_equals_independent_pinball_calculation():
    rng = np.random.default_rng(5)
    q = np.sort(rng.normal(size=(len(LEVELS), 10)), axis=0)
    y = rng.normal(size=10)
    error = y - q
    pinball = np.maximum(LEVELS[:, None] * error, (LEVELS[:, None] - 1) * error)
    np.testing.assert_allclose(wis(q, y), 2 * pinball.mean(0))
    # Degenerate last-value distributions have WIS equal to absolute error.
    flat = np.full((len(LEVELS), 1), 3.)
    np.testing.assert_allclose(wis(flat, np.array([5.])), 2)
