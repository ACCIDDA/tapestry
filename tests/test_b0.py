import pytest

torch = pytest.importorskip('torch')
from tapestry.models import B0, fair_crps
from tapestry.models.b0 import COUNT_TRANSFORMS, ED_TRANSFORMS


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


@pytest.mark.parametrize('count_transform', COUNT_TRANSFORMS)
@pytest.mark.parametrize('ed_transform', ED_TRANSFORMS)
def test_zero_residual_returns_last_valid_observation_and_outputs_stay_in_support(count_transform, ed_transform):
    pop = {'01': 100000., 'US': 10000000.}
    model = B0(width=8, scale=[100, 200, 50, .05, .05, .05], count_transform=count_transform,
               ed_transform=ed_transform, populations=pop, geography=True,
               input_scale=[2, 3, 4, .5, .5, .5], input_offset=[0, 0, 0, -4, -4, -4])
    torch.nn.init.zeros_(model.decoder[-1].weight)
    torch.nn.init.zeros_(model.decoder[-1].bias)
    x = torch.ones(1, 8, 6, 2, 2)
    x[:, :, :3, 0] = torch.tensor([10., 1000.])
    x[:, :, 3:, 0] = torch.tensor([.02, .3])
    x[:, -2, :3, 0] = torch.tensor([12., 1100.])
    x[:, -2, 3:, 0] = torch.tensor([.03, .25])
    # The latest week is missing, so the anchor is the second-to-last week.
    x[:, -1, :, 1] = 0
    x[:, -1, :, 0] = float('nan')
    cal, locations = torch.zeros(1, 2), ['01', 'US']
    z = torch.randn(3, 1, 16)
    out = model(x, cal, z=z, locations=locations)
    assert torch.allclose(out[:, :, :, :3], torch.tensor([12., 1100.]).expand(3, 1, 4, 3, 2), rtol=1e-3)
    assert torch.allclose(out[:, :, :, 3:], torch.tensor([.03, .25]).expand(3, 1, 4, 3, 2), rtol=1e-3)
    clone = B0(**model.config)
    clone.load_state_dict(model.state_dict())
    assert torch.equal(clone(x, cal, z=z, locations=locations), out)
    # Large residuals keep admissions nonnegative and ED visits proportions.
    torch.nn.init.normal_(model.decoder[-1].weight, std=5)
    wild = model(x, cal, z=torch.randn(64, 1, 16) * 3, locations=locations)
    assert torch.isfinite(wild).all() and (wild >= 0).all() and (wild[:, :, :, 3:] <= 1).all()


def test_rate_inversion_population_order_and_checkpoint():
    pop = {'01': 100000., 'US': 10000000.}
    model = B0(lookback=12, width=8, scale=[100, 200, 50, .05, .05, .05],
               count_transform='fourth_root', populations=pop, geography=True,
               input_scale=[2, 3, 4, .05, .05, .05])
    x = torch.ones(1, 12, 6, 2, 2)
    x[:, :, :3, 0] = torch.tensor([10., 1000.])
    x[:, :, 3:, 0] = .02
    z = torch.randn(3, 1, 16)
    out = model(x, torch.zeros(1, 2), z=z, locations=['01', 'US'])
    assert torch.allclose(model(x.flip(-1), torch.zeros(1, 2), z=z, locations=['US', '01']), out.flip(-1))
    clone = B0(**model.config)
    clone.load_state_dict(model.state_dict())
    assert torch.allclose(clone(x[..., :1], torch.zeros(1, 2), z=z, locations=['01']), out[..., :1])
    out.sum().backward()
    assert torch.isfinite(model.decoder[-1].bias.grad).all()


def test_dynamics_gaps_and_calendar():
    from tapestry.models.b0 import recent_dynamics
    from tapestry.models.run import calendar
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


def write_population(tmp_path):
    population = tmp_path / 'population.csv'
    population.write_text('location,population\nUS,1000000\n')
    return population


def test_input_scalers_ignore_masked_values_and_weights_are_independent(tmp_path):
    import numpy as np
    from types import SimpleNamespace
    from tapestry.models.experiments import model_options, LOSS_WEIGHTS
    x = np.ones((3, 6, 2, 1), dtype='float32')
    x[0, :, 0] = np.nan
    x[0, :, 1] = 0
    episode = {'locations': ('US',), 'context_dates': ('a', 'b', 'c'), 'X': x}
    args = SimpleNamespace(count_transform='sqrt', geography=True, dynamics=False, population_file=str(write_population(tmp_path)))
    a = model_options([episode], args)
    args.loss_weights = 'balanced_admissions'
    b = model_options([episode, episode], args)
    assert a == b
    assert a['input_scale'][:3] == pytest.approx([.1 ** .5] * 3)
    assert 'input_offset' not in a
    assert LOSS_WEIGHTS['flu_only'] == [1, 0, 0, 0, 0, 0]
    assert LOSS_WEIGHTS['objective'] == [1, 1, 1, .5, .5, .5]


def test_logit_ed_inputs_center_on_training_values_and_log1p_counts_scale(tmp_path):
    import numpy as np
    from types import SimpleNamespace
    from tapestry.models.experiments import model_options
    x = np.ones((4, 6, 2, 1), dtype='float32')
    x[:, :3, 0, 0] = np.array([10., 20., 40., 80.])[:, None]
    x[:, 3:, 0, 0] = np.array([.01, .02, .04, .9])[:, None]
    x[3, :, 0] = np.nan  # masked week never contributes
    x[3, :, 1] = 0
    episode = {'locations': ('US',), 'context_dates': ('a', 'b', 'c', 'd'), 'X': x}
    args = SimpleNamespace(count_transform='log1p', ed_transform='logit', geography=False, dynamics=False,
                           population_file=str(write_population(tmp_path)))
    options = model_options([episode], args)
    p = np.array([.01, .02, .04])
    logits = np.log(p / (1 - p))
    assert options['input_offset'] == pytest.approx([0, 0, 0] + [logits.mean()] * 3, rel=1e-5)
    assert options['input_scale'][3:] == pytest.approx([logits.std()] * 3, rel=1e-5)
    rates = np.log1p(np.array([10., 20., 40.]) / 10)
    assert options['input_scale'][:3] == pytest.approx([np.quantile(rates, .95)] * 3, rel=1e-5)


@pytest.mark.parametrize('encoder', ['mlp', 'conv'])
@pytest.mark.parametrize('heads', ['shared', 'state_us'])
@pytest.mark.parametrize('decoder', ['legacy', 'residual2'])
@pytest.mark.parametrize('spatial', ['none', 'attention'])
@pytest.mark.parametrize('noise', ['global', 'local'])
def test_architecture_gradients_masking_and_checkpoint(encoder, heads, decoder, spatial, noise):
    model = B0(lookback=12, width=8, latent=32, encoder=encoder, heads=heads, decoder=decoder,
               spatial=spatial, noise=noise)
    x = torch.ones(1, 12, 6, 2, 2)
    x[:, :, 3:, 0] = .02
    x[:, :, :3, 0, 1] = 50  # distinct locations, so attention has something to mix
    x[:, -2, :, 1] = 0
    z = torch.randn(3, 1, 32, requires_grad=True)
    local_z = torch.randn(3, 1, 2, 4)
    cal = torch.zeros(1, 2)
    out = model(x, cal, z=z, local_z=local_z, locations=['AL', 'US'])
    assert out.shape == (3, 1, 4, 6, 2)
    assert torch.isfinite(out).all() and (out >= 0).all()
    assert (out[:, :, :, 3:] <= 1).all()
    x[:, -2, :, 0] = float('nan')
    assert torch.allclose(out, model(x, cal, z=z, local_z=local_z, locations=['AL', 'US']))
    assert torch.allclose(out.flip(-1), model(x.flip(-1), cal, z=z, local_z=local_z.flip(2), locations=['US', 'AL']), atol=1e-6)
    clone = B0(**model.config)
    clone.load_state_dict(model.state_dict())
    assert torch.equal(out, clone(x, cal, z=z, local_z=local_z, locations=['AL', 'US']))
    out.sum().backward()
    assert z.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    if decoder == 'residual2':
        assert all(layer.weight.grad.abs().sum() > 0 for layer in model.decoder.modulations)
    assert len(model.local_noise_scales()) == (0 if noise == 'global' else 1 + (heads == 'state_us'))


@pytest.mark.parametrize('spatial', ['none', 'attention'])
def test_spatial_attention_mixes_locations_within_an_episode_only(spatial):
    torch.manual_seed(0)
    model = B0(width=8, spatial=spatial).eval()
    x = torch.rand(2, 8, 6, 2, 3)
    x[:, :, :, 1] = 1
    x[:, :, 3:, 0] *= .1
    z, cal = torch.randn(2, 2, 16), torch.zeros(2, 2)
    out = model(x, cal, z=z)
    changed = x.clone()
    changed[1, :, :, 0, 2] += 5  # another location's history, episode 1 only
    other = model(changed, cal, z=z)
    assert torch.allclose(other[:, 0], out[:, 0], rtol=1e-5, atol=1e-7)
    moved = not torch.allclose(other[:, 1, :, :, 0], out[:, 1, :, :, 0], rtol=1e-5, atol=1e-7)
    assert moved == (spatial == 'attention')


@pytest.mark.parametrize('decoder', ['legacy', 'residual2'])
@pytest.mark.parametrize('heads', ['shared', 'state_us'])
def test_local_noise_moves_only_its_location_and_learns_its_magnitude(decoder, heads):
    torch.manual_seed(0)
    model = B0(width=8, decoder=decoder, heads=heads, noise='local')
    x = torch.ones(1, 8, 6, 2, 2)
    x[:, :, 3:, 0] = .02
    cal, locations = torch.zeros(1, 2), ['AL', 'US']
    z, local_z = torch.randn(2, 1, 16), torch.randn(2, 1, 2, 4)
    out = model(x, cal, z=z, local_z=local_z, locations=locations)
    changed = local_z.clone()
    changed[:, :, 1] += 1
    other = model(x, cal, z=z, local_z=changed, locations=locations)
    assert torch.allclose(other[..., 0], out[..., 0])
    assert (other[..., 1] - out[..., 1]).abs().amin() > 0  # every member, horizon, and channel
    with pytest.raises(ValueError, match='Local latent'):
        model(x, cal, z=z, local_z=torch.randn(2, 1, 2, 4, 4), locations=locations)
    assert model.local_noise_scales() and all(v == pytest.approx(1) for v in model.local_noise_scales().values())
    out.sum().backward()
    assert all(p.grad.abs() > 0 for name, p in model.named_parameters() if name.endswith('local_scale'))
    # Without local noise the same draws reproduce the global-only initialization.
    torch.manual_seed(0)
    plain = B0(width=8, decoder=decoder, heads=heads)
    assert plain.local_noise_scales() == {}
    shared = {k: v for k, v in plain.state_dict().items()}
    assert all(torch.equal(model.state_dict()[k], v) for k, v in shared.items() if not k.startswith('spatial'))


@pytest.mark.parametrize('decoder', ['legacy', 'residual2'])
@pytest.mark.parametrize('noise', ['global', 'local'])
def test_separate_heads_receive_only_their_support_gradients(decoder, noise):
    model = B0(width=8, heads='state_us', decoder=decoder, noise=noise)
    x = torch.ones(1, 8, 6, 2, 2)
    x[:, :, 3:, 0] = .02
    with pytest.raises(ValueError, match='location IDs'):
        model(x, torch.zeros(1, 2))
    state = [p for name, p in model.named_parameters() if name.startswith(('decoder', 'local_'))]
    us = [p for name, p in model.named_parameters() if name.startswith(('us_decoder', 'us_local_'))]
    out = model(x, torch.zeros(1, 2), members=2, locations=['AL', 'US'])
    out[..., 0].sum().backward()
    assert sum(p.grad.abs().sum() for p in state) > 0
    assert all(p.grad is None or not p.grad.any() for p in us)
    model.zero_grad()
    model(x, torch.zeros(1, 2), members=2, locations=['AL', 'US'])[..., 1].sum().backward()
    assert all(p.grad is None or not p.grad.any() for p in state)
    assert sum(p.grad.abs().sum() for p in us) > 0
