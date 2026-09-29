"""End-to-end smoke test: fit and evaluate one season-CV fold on a synthetic panel.

Guards the wiring between the panel schema (`dataset.build`), episode construction
and `Model` -- in particular the channel/location and covariate/location axis order,
which the on-disk arrays store as `[.., L, C]` but `Model` expects as `[.., C, L]`.
"""
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from conftest import LOCATIONS, synthetic_panel
from tapestry.dataset.build import STATE_COVARIATE_NAMES, NATIONAL_COVARIATE_NAMES, save
from tapestry.dataset.episodes import select_covariates
from tapestry.model.scenario import Scenario
from tapestry.experiment import fitting

SMALL = dict(lookback=4, width=8, latent=4, members=2, validation_members=2, epochs=1, batch_size=4,
             geography=True, dynamics=False)


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    path = tmp_path / 'panel.npz'
    save(synthetic_panel(), path)
    locations_file = tmp_path / 'locations.csv'
    locations_file.write_text('abbreviation,population\nNC,10000000\nUS,330000000\n')
    monkeypatch.setattr(fitting, 'LOCATIONS', str(locations_file))
    return path


@pytest.mark.parametrize('options', [dict(), dict(fit_partition='pathogen'), dict(epochs=3, patience=1),
                                     dict(input_mode='vintaged', covariate_set='inpatient+kinsa', supplied_final=True)])
def test_fit_and_evaluate_one_fold(tmp_path, dataset, options):
    scenario = Scenario(**{**SMALL, **options})
    output = tmp_path / 'eval_2024-2025'
    fitting.fit(scenario, seed=42, held_out_season='2024-2025', eval_members=2, device='cpu',
                output=output, dataset=dataset)
    with np.load(output / 'forecasts.npz') as data:
        q = data['quantiles']
    assert q.shape[0] == 23 and q.shape[2:] == (4, 6, len(LOCATIONS))
    assert np.isfinite(q).all()


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
    assert not available[:, 2, nc].any()  # national-only: unavailable off the US column
