"""Scientific information-flow checks for the covariate/spatial comparison."""
import pytest

torch = pytest.importorskip('torch')
from chromantis.model.network import Model, PooledMessage, SpatialBlock


@pytest.mark.parametrize('kind', ['national_broadcast', 'gated_pool'])
def test_pooled_messages_exclude_missing_senders(kind):
    torch.manual_seed(12)
    block = PooledMessage(8, kind)
    context = torch.randn(1, 3, 8)
    observed = torch.tensor([[True, False, True]])
    changed = context.clone()
    changed[:, 1] += 100
    result = block(context, observed, ['NC', 'CA', 'US'])
    altered = block(changed, observed, ['NC', 'CA', 'US'])
    torch.testing.assert_close(result[:, [0, 2]], altered[:, [0, 2]])
    torch.testing.assert_close(block(context, torch.zeros_like(observed), ['NC', 'CA', 'US']), context)


def test_attention_excludes_missing_senders_and_has_no_all_missing_message():
    torch.manual_seed(4)
    block = SpatialBlock(8)
    context = torch.randn(2, 3, 8, requires_grad=True)
    observed = torch.tensor([[True, False, True], [False, False, False]])
    result = block(context, observed)
    gradient, = torch.autograd.grad(result[0, 0].sum(), context, retain_graph=True)
    assert not gradient[0, 1].any()
    torch.testing.assert_close(result[1], context[1])
    result.square().sum().backward()
    assert context.grad.isfinite().all()


@pytest.mark.parametrize('spatial', ['target_spatial', 'joint_location_target'])
def test_covariate_at_remote_location_reaches_local_forecast(spatial):
    torch.manual_seed(31)
    names=('fa','ca','ra','fe','ce','re');units=('count',)*3+('proportion',)*3
    groups=('flu','covid','rsv')*2
    model = Model(lookback=3, width=8, latent=4, spatial=spatial, covariate_names=['kinsa_ili'],
                  input_names=names,input_units=units,input_groups=groups,target_names=names,
                  target_units=units,target_groups=groups,target_input_indices=range(6))
    values = torch.rand(1, 3, 6, 2)
    covariates = torch.zeros(1, 3, 1, 2, 2)
    covariates[:, :, :, 0, 1] = 2
    covariates[:, :, :, 1, 1] = 1
    covariates.requires_grad_()
    result = model(values=values, available=torch.ones_like(values, dtype=torch.bool),
                   calendar=torch.zeros(1, 3), z=torch.randn(2, 1, 4), locations=['NC', 'US'],
                   covariates=covariates)
    gradient, = torch.autograd.grad(result[..., 0].sum(), covariates)
    assert gradient[:, :, :, 0, 1].abs().sum() > 0
    assert not gradient[:, :, :, 0, 0].any()
