import numpy as np

from tapestry.models.season_cv import LEVELS


def test_export_maps_leads_and_channel_order(tmp_path):
    from tapestry.evaluation.hubs import export_b0, SEASONS
    from datetime import date, timedelta
    for i, label in enumerate(SEASONS):
        folder = tmp_path / f'eval_{label}'
        folder.mkdir()
        context = [date(2023, 10, 7), date(2024, 10, 5), date(2025, 10, 4)][i]
        targets = [(context + timedelta(weeks=h)).isoformat() for h in range(1, 5)]
        q = np.broadcast_to(np.arange(6)[None, None, None, :, None], (len(LEVELS), 1, 4, 6, 1)).copy()
        np.savez(folder / 'forecasts.npz', quantile_levels=LEVELS, quantiles=q,
                 context_end=[context.isoformat()], target_dates=[targets], locations=['NC'],
                 truth=np.zeros((1, 4, 6, 1)), mask=np.ones((1, 4, 6, 1), dtype=bool))
    frame = export_b0(tmp_path)[('2023-2024', 'wk inc flu prop ed visits')]
    assert frame.reference_date.unique().tolist() == ['2023-10-14']
    assert frame.horizon.tolist() == [0, 1, 2, 3]
    assert frame.location.tolist() == ['37'] * 4
    assert (frame['q0.5'] == 3).all()
    assert frame.target_end_date.tolist() == ['2023-10-14', '2023-10-21', '2023-10-28', '2023-11-04']
