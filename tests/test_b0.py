import pytest

torch = pytest.importorskip('torch')
from tapestry.models import fair_crps
from tapestry.models.b0 import COUNT_TRANSFORMS


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
    from tapestry.models.b0 import invert_counts, transform_counts, transform_proportions
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
