import pytest

torch = pytest.importorskip('torch')
from influpaintx.models import B0, fair_crps


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


def test_stochastic_outputs_gradients_and_location_equivariance():
    torch.manual_seed(1)
    model = B0(width=16, scale=[100, 200, 50, .05, .05, .05])
    x = torch.ones(2, 8, 6, 2, 3)
    x[:, :, 3:, 0] = .02
    cal = torch.zeros(2, 2)
    z = torch.randn(4, 2, 16, requires_grad=True)
    samples = model(x, cal, z=z)
    assert samples.shape == (4, 2, 4, 6, 3)
    assert torch.isfinite(samples).all() and (samples >= 0).all()
    assert (samples[:, :, :, 3:] <= 1).all()
    assert samples.std(0).mean() > 0
    samples.sum().backward()
    assert z.grad.abs().sum() > 0
    assert model.modulate.weight.grad.abs().sum() > 0
    model.eval()
    assert torch.allclose(model(x.flip(-1), cal, z=z), samples.flip(-1))
    # Masked input placeholders must not change predictions.
    x[:, 0, :, 1] = 0
    a = model(x, cal, z=z)
    x[:, 0, :, 0] = float('nan')
    assert torch.allclose(model(x, cal, z=z), a)


@pytest.mark.parametrize('transform', ['sqrt', 'fourth_root'])
def test_rate_inversion_population_order_and_checkpoint(transform):
    pop = {'01': 100000., 'US': 10000000.}
    model = B0(lookback=12, width=8, scale=[100, 200, 50, .05, .05, .05],
               count_transform=transform, populations=pop, geography=True,
               input_scale=[2, 3, 4, .05, .05, .05])
    # A zero residual must invert to the last admission COUNT at each support.
    torch.nn.init.zeros_(model.decoder[-1].weight)
    torch.nn.init.zeros_(model.decoder[-1].bias)
    x = torch.ones(1, 12, 6, 2, 2)
    x[:, :, :3, 0] = torch.tensor([10., 1000.])
    x[:, :, 3:, 0] = .02
    z = torch.randn(3, 1, 16)
    out = model(x, torch.zeros(1, 2), z=z, locations=['01', 'US'])
    assert torch.allclose(out[:, :, :, :3], torch.tensor([10., 1000.]).expand(3, 1, 4, 3, 2), rtol=1e-5)
    assert torch.allclose(model(x.flip(-1), torch.zeros(1, 2), z=z, locations=['US', '01']), out.flip(-1))
    clone = B0(**model.config)
    clone.load_state_dict(model.state_dict())
    assert torch.allclose(clone(x[..., :1], torch.zeros(1, 2), z=z, locations=['01']), out[..., :1])
    out.sum().backward()
    assert torch.isfinite(model.decoder[-1].bias.grad).all()


def test_dynamics_gaps_and_calendar():
    from influpaintx.models.b0 import recent_dynamics
    from influpaintx.models.run import calendar
    values = torch.tensor([1., 2., 4.]).reshape(1, 3, 1, 1)
    mask = torch.ones_like(values)
    assert torch.allclose(recent_dynamics(values, mask).flatten(), torch.tensor([2., 1., 0., 1., 1.]))
    mask[:, -1] = 0
    assert torch.allclose(recent_dynamics(values * mask, mask).flatten(), torch.tensor([0., 0., 1/3, 0., 0.]))
    assert recent_dynamics(values * 0, mask * 0)[0, 2, 0] == 1
    cal = calendar(['2024-12-25', '2025-01-01'], True)
    assert cal[0, 2] == 0
    assert cal[1, 2] == pytest.approx(1/26)


@pytest.mark.parametrize('lookback', [8, 12, 26])
def test_experiment_training_and_masked_input(lookback):
    model = B0(lookback=lookback, width=8, count_transform='fourth_root',
               populations={'US': 10000000}, geography=True, dynamics=True)
    x = torch.ones(1, lookback, 6, 2, 1)
    x[:, :, 3:, 0] = .02
    x[:, -2, :, 1] = 0
    z = torch.randn(2, 1, 16)
    cal = torch.zeros(1, 3)
    out = model(x, cal, z=z, locations=['US'])
    x[:, -2, :, 0] = float('nan')
    assert torch.allclose(out, model(x, cal, z=z, locations=['US']))
    loss = fair_crps(out, torch.ones(1, 4, 6, 1), torch.ones(1, 4, 6, 1)).sum()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_input_scalers_ignore_masked_values_and_weights_are_independent(tmp_path):
    import numpy as np
    from types import SimpleNamespace
    from influpaintx.models.experiments import model_options, LOSS_WEIGHTS
    population = tmp_path / 'population.csv'
    population.write_text('location,population\nUS,1000000\n')
    x = np.ones((3, 6, 2, 1), dtype='float32')
    x[0, :, 0] = np.nan
    x[0, :, 1] = 0
    episode = {'locations': ('US',), 'context_dates': ('a', 'b', 'c'), 'X': x}
    args = SimpleNamespace(count_transform='sqrt', geography=True, dynamics=False, population_file=str(population))
    a = model_options([episode], args)
    args.loss_weights = 'balanced_admissions'
    b = model_options([episode, episode], args)
    assert a == b
    assert a['input_scale'][:3] == pytest.approx([.1 ** .5] * 3)
    assert LOSS_WEIGHTS['flu_only'] == [1, 0, 0, 0, 0, 0]
