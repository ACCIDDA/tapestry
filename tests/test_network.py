import pytest

torch = pytest.importorskip('torch')
from tapestry.model.network import COUNT_TRANSFORMS, Model, fair_crps, draw_dropout


def test_fair_crps_matches_pairwise_and_excludes_nan_labels():
    samples = torch.tensor([1., 2., 5.]).reshape(3, 1, 1, 1, 1).expand(-1, 1, 1, 2, 1).clone().requires_grad_()
    truth = torch.tensor([3., float('nan')]).reshape(1, 1, 2, 1)
    mask = torch.tensor([1., 0.]).reshape(1, 1, 2, 1)
    score = fair_crps(samples, truth, mask)
    expected = (1 + 2 + 2) / 3 - (1 + 4 + 1 + 3 + 4 + 3) / 12
    assert score[0].item() == pytest.approx(expected)
    assert score[1].item() == 0
    score.sum().backward()
    assert not samples.grad[:, :, :, 1].any()


def test_count_and_ed_transforms_invert_exactly():
    from tapestry.model.network import invert_counts, transform_counts, transform_proportions
    counts = torch.tensor([0., 3., 250., 40000.])
    population = torch.tensor([5e5, 5e5, 5e6, 3.3e8])
    for transform in COUNT_TRANSFORMS:
        values = transform_counts(counts, population, transform)
        assert (values >= 0).all()
        assert torch.allclose(invert_counts(values, population, transform), counts, rtol=1e-4, atol=1e-3)
    proportions = torch.tensor([.0001, .003, .02, .4])
    assert torch.allclose(torch.sigmoid(transform_proportions(proportions, 'logit')), proportions, rtol=1e-5)
    assert torch.allclose(transform_proportions(proportions, 'fourth_root').pow(4), proportions, rtol=1e-5)
    assert torch.equal(transform_proportions(proportions, 'linear'), proportions)


def test_model_forward_accepts_packed_x_or_values_available():
    torch.manual_seed(0)
    model = Model(lookback=4, width=8, latent=4)
    x = torch.rand(2, 4, 6, 2, 1)
    calendar = torch.zeros(2, 3)
    packed = model(x, calendar, members=3)
    values, available = x[:, :, :, 0], x[:, :, :, 1].bool()
    wrapped = model(values=values, available=available, calendar=calendar, members=3, vintaged=False)
    assert packed.shape == wrapped.shape
    assert packed.isfinite().all() and wrapped.isfinite().all()


def test_supplied_final_defaults_to_available_when_not_vintaged():
    """`vintaged=False` must treat every visible cell as known-final."""
    torch.manual_seed(1)
    model = Model(lookback=3, width=8, latent=4, supplied_final=True)
    values = torch.rand(1, 3, 6, 1)
    available = torch.ones_like(values, dtype=torch.bool)
    calendar = torch.zeros(1, 3)
    z = torch.randn(2, 1, 4)
    explicit_final = model(values=values, available=available, known_final=available, calendar=calendar,
                           z=z, vintaged=True)
    implicit_final = model(values=values, available=available, calendar=calendar, z=z, vintaged=False)
    torch.testing.assert_close(explicit_final, implicit_final)


def test_draw_dropout_only_hides_available_cells():
    import numpy as np
    a = np.ones((10, 12, 6, 2), dtype=bool)
    a[:, :4, 5] = False
    for scenario in ('natural', 'recent', 'gap', 'outage'):
        d = draw_dropout(a, np.random.default_rng(3), scenario=scenario)
        assert not (d & ~a).any()
        if scenario == 'natural':
            assert not d.any()
        else:
            assert d.any()
