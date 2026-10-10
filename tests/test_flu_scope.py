"""Scientific guards for named flu inputs, targets, and correction units."""
import numpy as np
import torch

from chromantis.dataset.cv import training_seasons
from chromantis.dataset.episodes import episodes
from chromantis.model.network import Model
from chromantis.model.revision_tree import TrajectoryNowcaster
from chromantis.model.scenario import Scenario
from chromantis.problem import Problem


FLU = Problem.load('problems/us-flu-short-term.json')
ALL_NAMES = tuple(signal.name for signal in FLU.dataset.signals)


def test_ed_correction_preserves_proportions_and_other_inputs():
    values = np.ones((8, 2, 4), np.float32)
    values[:, 1] = .02
    e = dict(values=values, available=np.ones_like(values, dtype=bool),
             known_final=np.zeros_like(values, dtype=bool), input_units=FLU.target_units,
             context_dates=tuple(f'2025-01-{day:02d}' for day in range(1, 9)))
    truth = dict(e, values=values.copy())
    truth['values'][:, 1] = .04
    model = TrajectoryNowcaster(2, 10, 1, 'phase')
    model.input_channels = [0, 1]
    model.channels = [1]
    model.fit([(e, truth)] * 30)
    changed = model.apply(e)
    assert np.array_equal(changed['values'][:, 0], values[:, 0])
    assert np.array_equal(changed['values'][:-2, 1], values[:-2, 1])
    assert np.all(changed['values'][-2:, 1] > .02)
    assert np.all(changed['values'][:, 1] <= 1)


def test_two_season_extension_is_problem_scoped():
    assert training_seasons(FLU, Scenario(training_window='last2'), '2025-2026') == ('2023-2024', '2024-2025')
    assert training_seasons(FLU, Scenario(training_window='last2'), '2024-2025') == ('2023-2024', '2025-2026')


def test_operational_episode_keeps_unknown_future_labels_masked():
    panel = dict(dates=np.array(['2026-09-19', '2026-09-26', '2026-10-03']), locations=np.array(['US']),
                 issuance_dates=np.array(['2026-10-07']), target_names=np.array(ALL_NAMES),
                 targets=np.ones((3, 1, 6), np.float32), asof_targets=np.ones((1, 3, 1, 6), np.float32),
                 covariate_names=np.array([]), covariates=np.zeros((3, 1, 0), np.float32),
                 covariate_national_names=np.array([]), covariates_national=np.zeros((3, 0), np.float32))
    args = (panel, FLU, FLU.input_names('target'), 2, 'reported')
    assert episodes(*args) == []
    episode, = episodes(*args, require_labels=False)
    assert episode['context_dates'][-1] == '2026-10-03'
    assert episode['target_dates'] == ('2026-10-10', '2026-10-17', '2026-10-24', '2026-10-31')
    assert not episode['target_available'].any()


def test_target_input_model_has_no_other_pathogen_values_or_masks():
    kwargs = dict(lookback=8, width=16, latent=4, decoder='quantile', geography=False,
                  input_names=FLU.targets, input_units=FLU.target_units, input_groups=FLU.target_groups,
                  target_names=FLU.targets, target_units=FLU.target_units, target_groups=FLU.target_groups,
                  target_input_indices=(0, 1), scale=np.ones((2, 3)))
    model = Model(**kwargs)
    x = torch.rand(2, 8, 2, 3)
    mask = torch.ones_like(x, dtype=torch.bool)
    result = model(values=x, available=mask, calendar=torch.zeros(2, 3))
    assert result.shape == (23, 2, 4, 2, 3)
    assert FLU.target_weights('objective') == (1., .5)


def test_named_flu_covariate_sets_do_not_add_other_pathogens():
    assert FLU.covariate_names('ww_flu') == ('nwss_flu_wval_like',)
    assert FLU.covariate_names('outpatient_flu') == ('outpatient_flu',)
    assert FLU.covariate_names('kinsa+ww_flu') == ('kinsa_ili', 'nwss_flu_wval_like')
