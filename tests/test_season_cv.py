from collections import Counter
from datetime import date, timedelta

import numpy as np
import pytest
torch = pytest.importorskip('torch')

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


def test_early_stopping_restores_the_best_validation_epoch():
    from types import SimpleNamespace
    from tapestry.models.run import fit, validation_draws, validation_loss
    rng = np.random.default_rng(3)
    panel = np.ones((157, 6, 2, 3), dtype=np.float32)
    panel[:, :3, 0] = rng.uniform(10, 100, size=(157, 3, 3))
    panel[:, 3:, 0] = rng.uniform(.01, .05, size=(157, 3, 3))
    ds = FinalizedDataset(panel, DAYS, ('AL', 'NC', 'US'), {})
    inner, validation, scales, _ = validation_split(ds, SEASONS[2])
    args = SimpleNamespace(lookback=8, horizons=[1, 2, 3, 4], width=8, device='cpu', lr=.01, epochs=8,
                           batch_size=16, members=4, seed=3, patience=2, loss_weights='objective',
                           count_transform='raw', ed_transform='linear', geography=False, dynamics=False,
                           encoder='mlp', spatial='none', heads='shared', decoder='residual2', noise='local', latent=4)
    torch.manual_seed(3)
    model, record = fit(inner, scales, args, validation=validation)
    epochs = len(record['validation_loss'])
    assert len(record['loss']) == epochs <= args.epochs
    assert record['best_epoch'] == 1 + int(np.argmin(record['validation_loss']))
    assert epochs == args.epochs or epochs - record['best_epoch'] == args.patience
    restored = validation_loss(model, validation, args, validation_draws(model, validation, args))
    assert restored == pytest.approx(min(record['validation_loss']), rel=1e-5)
    with pytest.raises(ValueError, match='patience'):
        fit(inner, scales, SimpleNamespace(**{**vars(args), 'patience': 0}), validation=validation)


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


def test_select_hub_quantiles_from_saved_levels():
    from tapestry.models.quantiles import select_quantiles
    assert len(LEVELS) == 23 and np.allclose(LEVELS + LEVELS[::-1], 1)
    values = np.arange(23 * 2).reshape(23, 2)
    np.testing.assert_array_equal(select_quantiles(values[::-1], LEVELS[::-1]), values)
    with pytest.raises(ValueError, match='exactly one saved quantile'):
        select_quantiles(values[:-2], LEVELS[:-2])
    with pytest.raises(ValueError, match='exactly one saved quantile'):
        select_quantiles(np.vstack([values, values[1]]), np.append(LEVELS, .025))
