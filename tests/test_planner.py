"""End-to-end smoke test: fit and evaluate one season-CV fold on synthetic arrays.

Guards the wiring between the array schema (`dataset.build`), episode
construction, and `Model` -- in particular the channel/location axis order,
which the on-disk arrays store as `[.., L, C]` but `Model` expects as
`[.., C, L]` (`experiment.planner._channel_first`).
"""
from datetime import date, timedelta

import numpy as np
import pytest

torch = pytest.importorskip('torch')

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
