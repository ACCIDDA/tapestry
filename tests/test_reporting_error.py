"""Scientific invariants for the reporting-window bootstrap."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from tapestry.dataset.build import load
from tapestry.dataset import cv
from tapestry.dataset.reporting_error import ReportingErrors
from tapestry.model.scenario import Scenario

@pytest.fixture(scope='module')
def experiment():
    panel = load('data/processed/panel.npz')
    s = replace(Scenario.from_string(pd.read_csv('docs/experiments/b-2-t0/named_ranking.csv').iloc[1].config_id),
                supplied_final=False, reporting_augmentation='vintage')
    keep = cv.week_roles(panel['dates'], s, '2025-2026') == 'fit'
    bank = ReportingErrors(panel, s, '2025-2026', keep)
    fold = cv.fold(panel, s, '2025-2026', inner=True)
    return panel, s, keep, bank, fold


def test_labels_unchanged_and_fresh_queries(experiment):
    _, _, _, bank, fold = experiment
    rng = np.random.default_rng(42)
    a, b = bank.batch(fold.train, rng), bank.batch(fold.train, rng)
    assert any(not np.array_equal(x['values'], y['values']) for x,y in zip(a,b))
    for original, augmented in zip(fold.train, a):
        for key in ('target_values', 'target_available', 'Y', 'target_dates'):
            np.testing.assert_array_equal(original[key], augmented[key])
        assert not augmented['known_final'].any()
        assert not (augmented['available'] & ~original['available']).any()


def test_held_out_and_validation_values_do_not_enter_bank(experiment):
    panel, scenario, keep, bank, _ = experiment
    changed = dict(panel)
    for key in ('targets','covariates','covariates_national'):
        changed[key] = panel[key].copy()
        changed[key][~keep] = 99999
    for key in ('asof_targets','asof_covariates','asof_covariates_national'):
        changed[key] = panel[key].copy()
        changed[key][:, ~keep] = 77777
    other = ReportingErrors(changed, scenario, '2025-2026', keep)
    for key in ('features','errors','visible','support','cov_errors','cov_visible','cov_support'):
        np.testing.assert_array_equal(getattr(bank,key),getattr(other,key))
    assert bank.error_dates.isdisjoint(set(panel['dates'][~keep]))


def test_local_donor_preserves_age_signal_and_location_alignment(experiment):
    _, _, _, bank, fold = experiment
    episode = next(e for e in fold.train if e['available'].all())
    class SelectFirst:
        def random(self, size):
            return np.zeros(size)
    rng = SelectFirst()
    # Encode location identity and age in every donor; mixing locations or
    # reversing ages would change this expected result.
    original = bank.errors.copy()
    bank.errors[:] = np.arange(bank.errors.shape[-1])[None,None,None,:] / 1000
    bank.errors[:] += np.arange(bank.errors.shape[1])[None,:,None,None] / 100
    out = bank.draw(episode, rng)
    from tapestry.dataset.reporting_error import transport
    expected = transport(episode['values'], bank.errors[0], bank.target_floor(episode))
    bank.errors[:] = original
    expected[:,3:] = np.minimum(1,expected[:,3:])
    np.testing.assert_allclose(out['values'], np.where(out['available'], expected, 0))


def test_log_transport_recovers_observed_and_preserves_zero():
    from tapestry.dataset.reporting_error import transport
    final = np.array([0., 1., 100., .0001, .1])
    observed = np.array([2., 0., 80., .0003, .05])
    floor = np.array([1., 1., 5., .0001, .005])
    error = np.log((observed + floor) / (final + floor))
    np.testing.assert_allclose(transport(final, error, floor), observed, atol=1e-12)
    np.testing.assert_array_equal(transport(final, np.zeros(5), floor), final)


def test_value_only_changes_errors_not_native_masks(experiment):
    panel, scenario, keep, original_bank, fold = experiment
    bank = ReportingErrors(panel, replace(scenario, reporting_missingness=False), '2025-2026', keep)
    np.testing.assert_array_equal(bank.errors, original_bank.errors)
    np.testing.assert_array_equal(bank.features, original_bank.features)
    out = bank.batch(fold.train, np.random.default_rng(42))
    assert any(not np.array_equal(a['values'], b['values']) for a,b in zip(fold.train,out))
    for a,b in zip(fold.train,out):
        for name in ('target_values','target_available','available'):
            np.testing.assert_array_equal(a[name],b[name])
        np.testing.assert_array_equal(a['covariates'][...,1,:],b['covariates'][...,1,:])
