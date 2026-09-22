"""End-to-end smoke test: fit and evaluate one season-CV fold on synthetic arrays.

Guards the wiring between the array schema (`dataset.build`), episode
construction, and `Model` -- in particular the channel/location axis order,
which the on-disk arrays store as `[.., L, C]` but `Model` expects as
`[.., C, L]` (`experiment.planner._channel_first`).
"""
from datetime import date, timedelta
import json

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from tapestry.dataset.build import STATE_COVARIATE_NAMES, NATIONAL_COVARIATE_NAMES
from tapestry.model.scenario import Scenario
from tapestry.experiment import planner

CHANNELS = ('nhsn_flu_admissions', 'nhsn_covid_admissions', 'nhsn_rsv_admissions',
            'nssp_flu_proportion', 'nssp_covid_proportion', 'nssp_rsv_proportion')
LOCATIONS = ('NC', 'US')


@pytest.fixture
def finalized_dataset(tmp_path, monkeypatch):
    first = date(2023, 9, 2)
    n_weeks = 3 * 52 + 10
    dates = [(first + timedelta(weeks=i)).isoformat() for i in range(n_weeks)]
    rng = np.random.default_rng(0)
    targets = np.abs(rng.normal(10, 2, size=(n_weeks, len(LOCATIONS), 6))).astype(np.float32)
    known_final = np.ones_like(targets, dtype=bool)
    covariates = np.zeros((n_weeks, len(LOCATIONS), 0), np.float32)
    arrays = dict(dates=np.array(dates, dtype='datetime64[D]'), locations=np.array(LOCATIONS),
                 targets=targets, target_names=np.array(CHANNELS), known_final=known_final,
                 covariates=covariates, covariate_mask=covariates.astype(bool),
                 covariate_names=np.array([]), covariates_national=np.zeros((n_weeks, 0), np.float32),
                 covariate_national_names=np.array([]))
    processed = tmp_path / 'processed'
    processed.mkdir()
    np.savez_compressed(processed / 'finalized.npz', **arrays)
    locations_file = tmp_path / 'locations.csv'
    locations_file.write_text('abbreviation,population\nNC,10000000\nUS,330000000\n')
    monkeypatch.setattr(planner, 'LOCATIONS', str(locations_file))
    return processed


def test_fit_and_evaluate_one_fold(tmp_path, finalized_dataset):
    scenario = Scenario(lookback=4, width=8, latent=4, members=2, validation_members=2,
                        epochs=1, batch_size=4, geography=True, dynamics=False)
    output = tmp_path / 'eval_2024-2025'
    planner.fit(scenario, seed=42, held_out_season='2024-2025', eval_members=2, device='cpu',
               output=output, dataset_root=str(finalized_dataset))
    with np.load(output / 'forecasts.npz') as data:
        q = data['quantiles']
    assert q.shape[0] == 23 and q.shape[2:] == (4, 6, len(LOCATIONS))
    assert np.isfinite(q).all()


def test_fit_partition_bundles_independently_fitted_components(tmp_path, finalized_dataset):
    scenario = Scenario(lookback=4, width=8, latent=4, members=2, validation_members=2,
                        epochs=1, batch_size=4, geography=True, dynamics=False,
                        fit_partition='pathogen')
    output = tmp_path / 'eval_2024-2025'
    planner.fit(scenario, seed=42, held_out_season='2024-2025', eval_members=2, device='cpu',
               output=output, dataset_root=str(finalized_dataset))
    assert (output / 'forecasts.npz').is_file()


def test_early_stopping_selects_and_then_refits(tmp_path, finalized_dataset):
    scenario = Scenario(lookback=4, width=8, latent=4, members=2, validation_members=2,
                        epochs=3, patience=1, batch_size=4, geography=True, dynamics=False)
    output = tmp_path / 'eval_2024-2025'
    planner.fit(scenario, seed=42, held_out_season='2024-2025', eval_members=2, device='cpu',
               output=output, dataset_root=str(finalized_dataset))
    assert (output / 'model.pt').is_file()


@pytest.fixture
def vintaged_dataset(tmp_path, monkeypatch):
    """Built at lookback 6, one covariate/national value per index so mis-indexed
    axes (state vs. national, covariate vs. location) show up as wrong numbers
    rather than a shape mismatch or a crash."""
    built_lookback, horizons = 6, (1, 2, 3, 4)
    window = built_lookback + len(horizons)
    first_issuance = date(2023, 9, 6)  # a Wednesday
    n_issuances = 3 * 52 + 5
    issuances = [(first_issuance + timedelta(weeks=i)).isoformat() for i in range(n_issuances)]
    rng = np.random.default_rng(0)
    dates = np.empty((n_issuances, window), dtype='datetime64[D]')
    for i, issuance in enumerate(issuances):
        end_of_context = date.fromisoformat(issuance) - timedelta(days=4)
        window_dates = [(end_of_context - timedelta(weeks=w)).isoformat() for w in reversed(range(built_lookback))]
        window_dates += [(end_of_context + timedelta(weeks=h)).isoformat() for h in horizons]
        dates[i] = np.array(window_dates, dtype='datetime64[D]')
    targets = np.abs(rng.normal(10, 2, size=(n_issuances, window, len(LOCATIONS), 6))).astype(np.float32)
    known_final = np.ones_like(targets, dtype=bool)
    n_state, n_national = len(STATE_COVARIATE_NAMES), len(NATIONAL_COVARIATE_NAMES)
    # covariates[..., l, k] == 100*k + l, so reading covariate k at location l back
    # out as anything other than 100*k + l means an axis got swapped somewhere.
    covariates = (100 * np.arange(n_state)[None, None, None, :]
                  + np.arange(len(LOCATIONS))[None, None, :, None]).astype(np.float32)
    covariates = np.broadcast_to(covariates, (n_issuances, window, len(LOCATIONS), n_state)).copy()
    national = np.broadcast_to(1000 + np.arange(n_national, dtype=np.float32),
                                (n_issuances, window, n_national)).copy()
    arrays = dict(issuance_dates=np.array(issuances, dtype='datetime64[D]'), dates=dates,
                 locations=np.array(LOCATIONS), targets=targets, target_names=np.array(CHANNELS),
                 known_final=known_final, covariates=covariates, covariate_mask=np.ones_like(covariates, dtype=bool),
                 covariate_names=np.array(STATE_COVARIATE_NAMES), covariates_national=national,
                 covariate_national_names=np.array(NATIONAL_COVARIATE_NAMES),
                 metadata=json.dumps(dict(lookback=built_lookback)))
    processed = tmp_path / 'processed'
    processed.mkdir()
    np.savez_compressed(processed / 'vintaged.npz', **arrays)
    locations_file = tmp_path / 'locations.csv'
    locations_file.write_text('abbreviation,population\nNC,10000000\nUS,330000000\n')
    monkeypatch.setattr(planner, 'LOCATIONS', str(locations_file))
    return processed


def test_covariates_are_read_from_the_correct_axis(vintaged_dataset):
    arrays = planner.load_dataset(vintaged_dataset / 'vintaged.npz')
    names = list(STATE_COVARIATE_NAMES[:2]) + list(NATIONAL_COVARIATE_NAMES)
    values, available = planner.assemble_covariates(arrays, names)
    nc, us = LOCATIONS.index('NC'), LOCATIONS.index('US')
    for k, name in enumerate(STATE_COVARIATE_NAMES[:2]):
        j = list(STATE_COVARIATE_NAMES).index(name)
        assert np.allclose(values[..., k, nc], 100 * j + nc)
        assert np.allclose(values[..., k, us], 100 * j + us)
    national_k = len(STATE_COVARIATE_NAMES[:2])
    assert np.allclose(values[..., national_k, us], 1000)
    assert not available[..., national_k, nc].any()  # national-only: unavailable off the US column


def test_scenario_lookback_shorter_than_built_keeps_the_horizon_anchored(vintaged_dataset):
    arrays = planner.load_dataset(vintaged_dataset / 'vintaged.npz')
    built = list(planner.episodes_from_vintaged(arrays, lookback=6))
    short = list(planner.episodes_from_vintaged(arrays, lookback=4))
    assert built[0]['target_dates'] == short[0]['target_dates']
    assert built[0]['context_dates'][-4:] == short[0]['context_dates']


def test_scenario_lookback_longer_than_built_raises(vintaged_dataset):
    arrays = planner.load_dataset(vintaged_dataset / 'vintaged.npz')
    with pytest.raises(ValueError):
        list(planner.episodes_from_vintaged(arrays, lookback=8))


def test_fit_with_covariates_on_the_vintaged_dataset(tmp_path, vintaged_dataset):
    scenario = Scenario(lookback=4, width=8, latent=4, members=2, validation_members=2,
                        epochs=1, batch_size=4, geography=True, dynamics=False,
                        input_mode='vintaged', covariate_set='inpatient+kinsa', supplied_final=True)
    output = tmp_path / 'eval_2024-2025'
    planner.fit(scenario, seed=42, held_out_season='2024-2025', eval_members=2, device='cpu',
               output=output, dataset_root=str(vintaged_dataset))
    with np.load(output / 'forecasts.npz') as data:
        q = data['quantiles']
    assert np.isfinite(q).all()
