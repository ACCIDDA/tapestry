import numpy as np
import torch

from tapestry.models.b1_run import covariate_scales
from tapestry.models.b2 import B2


def test_unseen_covariate_is_gated_at_inference():
    model = B2([0], {'NC': 1_000_000}, ['NC'], lookback=3, width=8,
        count_transform='fourth_root', ed_transform='logit', supplied_final=True,
        covariate_names=['observed', 'unseen'], covariate_offset=[[0], [0]],
        covariate_scale=[[1], [1]], covariate_trained=[[True], [False]])
    values = torch.ones(1, 3, 6, 1)
    available = torch.ones_like(values, dtype=torch.bool)
    known = torch.zeros_like(available)
    calendar = torch.zeros(1, 3)
    covariates = torch.ones(1, 3, 2, 2, 1)
    changed = covariates.clone()
    changed[:, :, 1, 0] = 1e6
    z = torch.zeros(2, 1, model.config['latent'])
    first = model(values, available, calendar, members=2, known_final=known,
                  covariates=covariates, z_future=z)
    second = model(values, available, calendar, members=2, known_final=known,
                   covariates=changed, z_future=z)
    assert torch.equal(first, second)


def test_covariate_statistics_use_observed_training_cells_only():
    c = np.zeros((2, 2, 2, 2, 1), np.float32)
    c[..., 0, 0, 0] = np.array([[1, 3], [5, 7]])
    c[..., 0, 1, 0] = True
    c[..., 1, 0, 0] = 999  # masked values must not enter statistics
    episodes = [{'C': row} for row in c]
    offset, scale, trained = covariate_scales(episodes)
    assert np.allclose(offset[0], [4]) and np.allclose(scale[0], [np.sqrt(5)])
    assert trained == [[True], [False]]


def test_wednesday_covariates_respect_release_cutoff_and_retractions():
    import pandas as pd
    from tapestry.model_data.b2 import _fill_revisions

    revisions = pd.DataFrame({
        'reference_time': ['2026-02-28'] * 3, 'geo_value': ['NC'] * 3,
        'report_time': ['2026-03-04', '2026-03-11', '2026-03-18'],
        'value': [10., np.nan, 30.],
    })
    origins = np.array(['2026-03-03', '2026-03-04', '2026-03-11', '2026-03-18'])
    dates = np.full((4, 1), '2026-02-28')
    final = np.full((4, 1, 1, 1), np.nan)
    wednesday = final.copy()
    _fill_revisions(revisions, origins, dates, ('NC',), final, wednesday, 0)
    np.testing.assert_allclose(wednesday[:, 0, 0, 0], [np.nan, 10., np.nan, 30.], equal_nan=True)
    np.testing.assert_allclose(final[:, 0, 0, 0], 30.)


def test_kinsa_week_is_a_complete_sunday_to_saturday_mean_at_each_report_date(tmp_path):
    import pandas as pd
    from tapestry.model_data.b2 import _kinsa_weekly

    snapshot = tmp_path / 'raw/pophive_kinsa_ili/snapshots/s1'
    snapshot.mkdir(parents=True)
    (tmp_path / 'raw/pophive_kinsa_ili/latest.json').write_text('{"snapshot_id": "s1"}')
    days = pd.date_range('2026-03-01', '2026-03-10')  # a Sunday through the next Tuesday
    daily = pd.DataFrame({'report_time': (days + pd.Timedelta(days=1)).strftime('%Y-%m-%dT10:00:00'),
                          'geo_type': 'nation', 'geo_value': 'US', 'reference_time': days.strftime('%Y-%m-%d'),
                          'kinsa_cough_cold_flu': np.arange(1., 11.)})
    revised = daily.iloc[[3]].assign(report_time='2026-03-20T10:00:00', kinsa_cough_cold_flu=13.)
    pd.concat([daily, revised]).to_csv(snapshot / 'archive.csv.gz', index=False)

    weekly, _ = _kinsa_weekly(tmp_path, '2026-12-31')
    # The partial week of 8-10 March has no value; the revision is a new report of week one.
    assert weekly.reference_time.tolist() == ['2026-03-07', '2026-03-07']
    assert weekly.report_time.tolist() == ['2026-03-08', '2026-03-20']
    np.testing.assert_allclose(weekly.value, [4., 4. + 9 / 7])
    assert len(_kinsa_weekly(tmp_path, '2026-03-10')[0]) == 1
