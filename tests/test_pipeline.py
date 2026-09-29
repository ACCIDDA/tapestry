"""Scientific alignment, leakage and uncertainty at the nowcast/forecast boundary."""
import numpy as np
import torch
import pytest

from tapestry.dataset import cv
from tapestry.dataset.build import CHANNELS
from tapestry.experiment.two_stage import cross_fit_keep, nowcast_training
from tapestry.model.pipeline import reconstruct, forecast_histories, CovariateHistory, HistorySamples
from tapestry.model.scenario import Scenario


def test_nowcast_labels_match_completed_weeks_and_inputs_never_use_truth(panel):
    scenario = Scenario(task='nowcast', input_mode='vintaged', lookback=4, nowcast_weeks=2)
    keep = np.ones(len(panel['dates']), dtype=bool)
    eps = nowcast_training(panel, scenario, keep)
    changed = dict(panel, targets=panel['targets'] + 999)
    other = nowcast_training(changed, scenario, keep)
    for e, b in zip(eps, other):
        assert e['target_dates'] == e['context_dates'][-2:]
        np.testing.assert_array_equal(e['values'], b['values'])
        np.testing.assert_array_equal(e['available'], b['available'])
        assert not e['known_final'].any()
        ids = [list(panel['dates'].astype(str)).index(d) for d in e['target_dates'] if d >= str(panel['dates'][0])]
        np.testing.assert_array_equal(e['target_values'][-len(ids):], panel['targets'][ids].swapaxes(-1, -2))


def test_cross_fit_excludes_boundary_labels_and_outer_fold(panel):
    s = Scenario(task='pipeline', input_mode='vintaged', lookback=4)
    roles = cv.week_roles(panel['dates'], s, '2024-2025')
    keep = np.isin(roles, ['fit', 'validation'])
    batch = cv.fold(panel, s, '2024-2025').train
    selected = [e for e in batch if cv.season(e['context_dates'][-1]) == '2023-2024']
    fitting = cross_fit_keep(panel, keep, '2023-2024', selected, s.nowcast_weeks)
    visible = set(panel['dates'][fitting].astype(str))
    excluded = {d for e in selected for d in (*e['context_dates'][-2:], *e['target_dates'])}
    assert not visible & excluded
    assert all(cv.season(d) == '2025-2026' for d in visible)
    assert visible


class DrawNowcaster:
    config = {'horizons': [-1, 0], 'lookback': 3, 'location_ids': ['NC', 'US'], 'covariate_names': []}

    def __call__(self, *, values, members, **kwargs):
        m = torch.arange(members, dtype=values.dtype)
        return m[:, None, None, None, None].expand(members, len(values), 2, 6, 2)


class EchoForecaster:
    config = {'horizons': [1], 'lookback': 4, 'location_ids': ['NC', 'US'], 'covariate_names': []}

    def __call__(self, *, values, available, known_final, estimated, members, **kwargs):
        assert members == 1
        assert not known_final[:, -2:].any()
        assert estimated[:, -2:].all()
        assert available[:, -2:].all()
        return values[:, -1:][None]


def test_handoff_preserves_joint_draws_masks_and_native_values():
    values = torch.full((2, 4, 6, 2), 120.)
    available = torch.ones_like(values, dtype=torch.bool)
    available[:, 0] = False
    histories = reconstruct(DrawNowcaster(), values=values, available=available,
                            known_final=available, locations=('NC', 'US'), members=3,
                            context_dates=(('2025-01-04', '2025-01-11', '2025-01-18', '2025-01-25'),) * 2,
                            issuances=('2025-01-29',) * 2)
    assert histories.targets == CHANNELS
    assert not histories.available[:, 0].any()
    assert not histories.estimated[:, :2].any()
    torch.testing.assert_close(histories.values[:, :, 1], torch.full((3, 2, 6, 2), 120.))
    out = forecast_histories(EchoForecaster(), histories)
    assert out.shape == (3, 2, 1, 6, 2)
    for i in range(3):
        torch.testing.assert_close(out[i], torch.full_like(out[i], float(i)))


@pytest.mark.parametrize('task', ['nowcast', 'pipeline'])
def test_validation_and_heldout_truth_cannot_change_training(panel, task):
    s = Scenario(task=task, input_mode='vintaged', lookback=4, epochs=2, patience=1)
    roles = cv.week_roles(panel['dates'], s, '2024-2025')
    excluded = roles != 'fit'
    changed = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in panel.items()}
    for name in ('targets', 'covariates', 'covariates_national'):
        changed[name][excluded] += 999
    for name in ('asof_targets', 'asof_covariates', 'asof_covariates_national'):
        changed[name][:, excluded] += 999
    a = cv.fold(panel, s, '2024-2025', inner=True)
    b = cv.fold(changed, s, '2024-2025', inner=True)
    assert len(a.train) == len(b.train)
    for left, right in zip(a.train, b.train):
        for key in ('values', 'available', 'target_values', 'target_available', 'known_final'):
            np.testing.assert_array_equal(left[key], right[key])
    hidden = set(panel['dates'][roles == 'validation'].astype(str))
    for e in a.validation:
        assert {d for d, mask in zip(e['target_dates'], e['target_available']) if mask.any()} <= hidden


def test_dated_covariates_are_reordered_and_misalignment_is_rejected():
    dates = (('2025-01-04', '2025-01-11', '2025-01-18', '2025-01-25'),)
    issuances = ('2025-01-29',)
    values = torch.ones(2, 1, 4, 6, 2)
    available = torch.ones_like(values[0], dtype=torch.bool)
    estimated = torch.zeros_like(available)
    estimated[:, -2:] = True
    histories = HistorySamples(values, available, ~estimated, estimated, ('NC', 'US'), dates, issuances)
    cov = torch.zeros(1, 4, 2, 2, 2)
    cov[:, :, 0, 0] = 12
    cov[:, :, 1, 0] = 34
    cov[:, :, :, 1] = 1

    class NamedForecaster(EchoForecaster):
        config = dict(EchoForecaster.config, lookback=2, covariate_names=['b', 'a'])
        def __call__(self, *, covariates, **kwargs):
            assert covariates.shape[1] == 2
            assert (covariates[:, :, 0, 0] == 34).all()
            assert (covariates[:, :, 1, 0] == 12).all()
            return super().__call__(**kwargs)

    named = CovariateHistory(cov, ('a', 'b'), ('NC', 'US'), dates, issuances)
    forecast_histories(NamedForecaster(), histories, covariates=named)
    shifted = CovariateHistory(cov, ('a', 'b'), ('NC', 'US'),
                              (('2025-01-11', '2025-01-18', '2025-01-25', '2025-02-01'),), ('2025-02-05',))
    with pytest.raises(ValueError, match='do not match'):
        forecast_histories(NamedForecaster(), histories, covariates=shifted)
    with pytest.raises(ValueError, match='Calendar'):
        forecast_histories(NamedForecaster(), histories, calendar=torch.zeros(1, 3), covariates=named)
    with pytest.raises(ValueError, match='Wednesday'):
        HistorySamples(values, available, ~estimated, estimated, ('NC', 'US'), dates, ('2025-02-05',))
