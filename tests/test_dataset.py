"""Leakage and alignment of episodes cut from the panel (`dataset.episodes`, `dataset.cv`)."""
from datetime import date, timedelta

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from conftest import LOCATIONS, code, synthetic_panel
from chromantis.dataset.build import STATE_COVARIATE_NAMES, NATIONAL_COVARIATE_NAMES, decode, encode
from chromantis.dataset.cv import fold, season, week_roles
from chromantis.dataset.episodes import episodes, select_covariates
from chromantis.experiment.training import model_options, unique_truth
from chromantis.model.objective import loss_cell_weights, loss_scales
from chromantis.model.scenario import Scenario
from chromantis.problem import Problem

PROBLEM = Problem.load('problems/us-respiratory-all-short-term.json')
SEASONS = PROBLEM.folds

# Training inputs follow `history_source`: finalized scheduled histories, or actual Wednesday reports.
SCENARIOS = [Scenario(lookback=6, epochs=3, patience=1),
             Scenario(lookback=6, epochs=3, patience=1, covariate_set='inpatient+kinsa'),
             Scenario(lookback=6, epochs=3, patience=1, history_source='reported', covariate_set='inpatient+kinsa'),
             Scenario(lookback=6, epochs=3, patience=1, history_source='reported', covariate_set='inpatient',
                      validation_weeks=2, validation_spacing=10, validation_offset=1)]
POPULATIONS = {'NC': 1e7, 'US': 3.3e8}


def test_season_boundary_and_53_week_year():
    assert season(date(2023, 8, 5)) == '2023-2024'
    assert season(date(2023, 7, 29)) == '2022-2023'
    assert season(date(2021, 1, 2)) == '2020-2021'


def weeks_of(values, available):
    """Reference-week codes of every available value (values encode their week below 1000)."""
    return set((np.floor(values[available]) % 1000).astype(int).tolist())


def perturbed(panel, weeks):
    """The same panel with every value of `weeks` changed, wherever it is stored."""
    other = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in panel.items()}
    rows = np.array([str(d) in weeks for d in panel['dates']])
    for name in ('targets', 'covariates', 'covariates_national'):
        other[name][rows] += 777
    for name in ('asof_targets', 'asof_covariates', 'asof_covariates_national'):
        other[name][:, rows] += 777  # by reference week
    return other


@pytest.mark.parametrize('scenario', SCENARIOS)
@pytest.mark.parametrize('held_out', SEASONS)
@pytest.mark.parametrize('inner', [False, True])
def test_training_never_sees_held_out_or_validation_weeks(panel, scenario, held_out, inner):
    dates = np.array([str(d) for d in panel['dates']])
    labels = np.array([season(d) for d in dates])
    hidden = (labels == held_out) | ~np.isin(labels, SEASONS)
    if inner:
        hidden |= week_roles(dates, PROBLEM, scenario, held_out) == 'validation'
    hidden_weeks = set(dates[hidden])
    hidden_codes = {int(code(d)) for d in hidden_weeks}
    a = fold(panel, PROBLEM, scenario, held_out, inner)
    b = fold(perturbed(panel, hidden_weeks), PROBLEM, scenario, held_out, inner)
    # No hidden week reaches a training input or label, directly...
    for e in a.train:
        assert not weeks_of(e['values'], e['available']) & hidden_codes
        assert not weeks_of(e['target_values'], e['target_available']) & hidden_codes
        if 'covariates' in e:
            c = e['covariates']
            assert not weeks_of(c[..., 0, :], c[..., 1, :].astype(bool)) & hidden_codes
    # ...nor, through any path, the loss scales, loss weights or covariate standardization.
    assert len(a.train) == len(b.train)
    for x, y in zip(a.train, b.train):
        for key in ('values', 'available', 'known_final', 'target_values', 'target_available', 'covariates'):
            if key in x:
                np.testing.assert_array_equal(x[key], y[key])
    assert model_options(a.train, PROBLEM, scenario, POPULATIONS) == model_options(b.train, PROBLEM, scenario, POPULATIONS)
    assert loss_scales(unique_truth(a.train), PROBLEM.target_units) == loss_scales(unique_truth(b.train), PROBLEM.target_units)
    weights = PROBLEM.target_weights(scenario.loss_weights)
    np.testing.assert_array_equal(loss_cell_weights(a.train, weights), loss_cell_weights(b.train, weights))
    if inner:
        hidden_validation = set(dates[week_roles(dates, PROBLEM, scenario, held_out) == 'validation'])
        assert a.validation
        for e in a.validation:
            assert {d for d, m in zip(e['target_dates'], e['target_available'].any(axis=(1, 2))) if m} <= hidden_validation
        # Validation episodes (inputs, labels, availability, covariates) never see held-out or unused weeks.
        outside = set(dates[(labels == held_out) | ~np.isin(labels, SEASONS)])
        c = fold(perturbed(panel, outside), PROBLEM, scenario, held_out, inner)
        assert len(a.validation) == len(c.validation)
        for x, y in zip(a.validation, c.validation):
            for key in ('values', 'available', 'known_final', 'target_values', 'target_available', 'covariates'):
                if key in x:
                    np.testing.assert_array_equal(x[key], y[key])
    else:
        assert a.score[0]['context_dates'][-1] == dates[labels == held_out][0]  # origins from the season start
        for e in a.score:
            assert season(e['context_dates'][-1]) == held_out
            assert all(season(d) == held_out for d, m in zip(e['target_dates'], e['target_available'].any(axis=(1, 2))) if m)


@pytest.mark.parametrize('R', [0, 2, 6])
def test_vintaged_episodes_take_as_of_values_only_where_the_issuance_saw_them(panel, R):
    lookback, names = 6, ('inpatient_flu', 'kinsa_ili')
    start = str(panel['dates'][0])
    nc, us = LOCATIONS.index('NC'), LOCATIONS.index('US')
    finalized = {e['context_dates'][-1]: e for e in episodes(panel, PROBLEM, PROBLEM.input_names('all'), lookback, 'scheduled_final', names)}
    assert min(finalized) == start  # context before the calendar is padding, not a missing origin
    vintaged = episodes(panel, PROBLEM, PROBLEM.input_names('all'), lookback, 'vintaged', names, asof_weeks=R)
    unpublished = 0
    for e in vintaged:
        issuance = date.fromisoformat(e['issuance'])
        end = (issuance - timedelta(days=4)).isoformat()
        assert e['context_dates'][-1] == end
        assert e['target_dates'] == tuple((issuance + timedelta(days=3 + 7 * h)).isoformat() for h in range(4))
        for i, day in enumerate(e['context_dates']):
            value, available, final = e['values'][i, 0, nc], e['available'][i, 0, nc], e['known_final'][i, 0, nc]
            if day < start:
                assert not available
                continue
            truth = code(day) + 1000
            if i >= lookback - R:  # as-of window: exactly what the issuance saw, or nothing
                asof = panel['asof_targets'][list(panel['issuance_dates']).index(np.datetime64(e['issuance'])),
                                             list(map(str, panel['dates'])).index(day), nc, 0]
                assert not final
                if np.isnan(asof):
                    assert not available  # not published at the cutoff: unavailable, never truth-filled
                    unpublished += 1
                else:
                    assert available and value == truth + .5
            else:
                assert value == truth and final  # older week: truth, flagged final
            cov = e['covariates'][i]
            assert cov[0, 0, nc] == code(day) + 1000 + .5 and cov[0, 1, nc]  # inpatient_flu, as of the issuance
            assert cov[1, 0, us] == code(day) + 50000 + .5 and cov[1, 1, us] and cov[1, 1, nc]  # national Kinsa broadcast to all locations
            assert cov[1, 0, nc] == cov[1, 0, us]
        labelled = e['target_available'][:, 0, nc]
        np.testing.assert_array_equal(e['target_values'][labelled, 0, nc],
                                      [code(d) + 1000 for d, m in zip(e['target_dates'], labelled) if m])
        truth_episode = finalized.get(end)
        if truth_episode:
            np.testing.assert_array_equal(e['target_values'], truth_episode['target_values'])
    assert (unpublished > 0) == (R > 0) and len(vintaged) > 100


def test_sparse_as_of_storage_round_trips_exactly():
    panel = synthetic_panel()
    panel['asof_covariates'][5, 3, 0, 0] = np.nan  # visible in truth, not at this cutoff
    stored = encode(panel)
    assert stored['asof_targets_revised'].sum() < panel['asof_targets'].size
    restored = decode(stored)
    for name in ('asof_targets', 'asof_covariates', 'asof_covariates_national'):
        np.testing.assert_array_equal(restored[name], panel[name])


def test_covariates_are_read_from_the_correct_axis():
    panel = synthetic_panel()
    names = list(STATE_COVARIATE_NAMES[:2]) + list(NATIONAL_COVARIATE_NAMES)
    values, available = select_covariates(panel['covariates'], panel['covariates_national'], panel['covariate_names'],
                                          panel['covariate_national_names'], LOCATIONS, names)
    nc, us = LOCATIONS.index('NC'), LOCATIONS.index('US')
    for k in range(2):
        np.testing.assert_array_equal(values[:, k, nc], panel['covariates'][:, nc, k])
        np.testing.assert_array_equal(values[:, k, us], panel['covariates'][:, us, k])
    np.testing.assert_array_equal(values[:, 2, us], panel['covariates_national'][:, 0])
    # A national source (Kinsa) is broadcast to every location with its own availability (dataset/episodes.py).
    np.testing.assert_array_equal(values[:, 2, nc], values[:, 2, us])
    np.testing.assert_array_equal(available[:, 2, nc], available[:, 2, us])
