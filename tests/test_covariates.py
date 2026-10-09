"""Covariate reduction must respect reporting masks and temporal direction."""
import torch
from chromantis.model.covariates import CovariateEncoder, trailing_mean


def test_smoothing_is_causal_and_ignores_unpublished_values():
    x = torch.tensor([1., 900., 3., 7.]).reshape(1, 4, 1, 1)
    mask = torch.tensor([True, False, True, True]).reshape_as(x)
    smooth, visible = trailing_mean(x, mask)
    torch.testing.assert_close(smooth.flatten(), torch.tensor([1., 1., 2., 5.]))
    assert visible.all()
    changed = x.clone()
    changed[:, 3] = 9999
    torch.testing.assert_close(trailing_mean(changed, mask)[0][:, :3], smooth[:, :3])


def test_summary_slope_and_report_age_use_observed_times():
    x = torch.tensor([1., 900., 5., 900.]).reshape(1, 4, 1, 1)
    mask = torch.tensor([True, False, True, False]).reshape_as(x)
    result = CovariateEncoder(4, 'summary')(x, mask).flatten()
    # Last, recent mean, regression slope per week, SD, fraction observed, normalized age.
    torch.testing.assert_close(result, torch.tensor([5., 5., 2., 2., .5, .25]))


def test_compact_representations_ignore_missing_values_and_preserve_source_order():
    for kind in ('summary', 'shared'):
        model = CovariateEncoder(4, kind)
        x = torch.randn(2, 4, 3, 2)
        mask = torch.rand_like(x) > .4
        mask[:, :, 1] = False
        a = model(x, mask).reshape(2, 2, 3, 6)
        altered = torch.where(mask, x, 1e9)
        torch.testing.assert_close(model(altered, mask).reshape_as(a), a)
        torch.testing.assert_close(a[:, :, 1, :5], torch.zeros_like(a[:, :, 1, :5]))
        assert (a[:, :, 1, 5] == 1).all()
        order = [2, 0, 1]
        b = model(x[:, :, order], mask[:, :, order]).reshape_as(a)
        torch.testing.assert_close(b, a[:, :, order])


def test_national_pool_keeps_native_us_separate_and_excludes_unobserved_states():
    from chromantis.model.network import pooled_context
    context = torch.tensor([[[2., 4.], [4., 8.], [900., 900.], [20., 30.]]])
    visible = torch.tensor([[True, True, False, True]])
    result = pooled_context(context, visible, ['NC', 'CA', 'TX', 'US'])
    torch.testing.assert_close(result, torch.tensor([[3., 6., 20., 30.]]))
    torch.testing.assert_close(pooled_context(context, torch.zeros_like(visible), ['NC', 'CA', 'TX', 'US']),
                               torch.zeros_like(result))


def test_national_covariate_broadcast_preserves_missingness_without_us_row():
    import numpy as np
    from chromantis.dataset.episodes import select_covariates
    values, mask = select_covariates(np.zeros((2, 2, 0)), np.array([[3.], [np.nan]]), [],
                                     ['kinsa_ili'], ['NC', 'CA'], ['kinsa_ili'])
    np.testing.assert_array_equal(values[:, 0], [[3., 3.], [0., 0.]])
    np.testing.assert_array_equal(mask[:, 0], [[True, True], [False, False]])
