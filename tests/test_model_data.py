from datetime import date

import numpy as np
import pytest

from tapestry.model_data import FinalizedDataset, build_dataset
from tapestry.model_data.finalized import season


def test_windows_masks_alignment():
    panel = np.zeros((3, 6, 2, 2), dtype=np.float32)
    panel[0, 0, :, 0] = (0, 1)  # A genuine observed zero.
    panel[1, 0, :, 1] = (20, 1)
    panel[2, 0, :, 1] = (30, 1)
    ds = FinalizedDataset(panel, ('2023-09-02', '2023-09-09', '2023-09-16'), ('AL', 'US'), {})
    result = ds.query('2023-09-02', locations=['US', 'AL'], target_end='2023-09-09')
    assert result['X'].shape == (8, 6, 2, 2)
    assert result['X'][-1, 0, :, 1].tolist() == [0, 1]
    assert not result['X'][:-1].any()
    assert result['Y'][0, 0, :, 0].tolist() == [20, 1]
    assert not result['Y'][1:].any()  # Held-out labels excluded.
    assert result['target_dates'][0] == '2023-09-09'


def test_season_boundary_and_53_week_year():
    assert season(date(2023, 8, 5)) == '2023-2024'
    assert season(date(2023, 7, 29)) == '2022-2023'
    assert season(date(2021, 1, 2)) == '2020-2021'


def test_builder_units_and_missing_values(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from tapestry.model_data import finalized as module

    rows = {
        'cdc_nhsn_final': [{'week': '2023-09-02', 'jurisdiction': 'AL',
                            'totalconfflunewadm': '0', 'totalconfc19newadm': '*', 'totalconfrsvnewadm': '-1'}],
        'cdc_nssp_trajectories': [{'week': '2023-09-02', 'geography': 'Alabama',
                                  'percent_visits_influenza': '2', 'percent_visits_covid': '101'}],
    }
    for key in rows:
        folder = tmp_path / 'raw' / key / 'snapshots' / 'test'
        folder.mkdir(parents=True)
        (folder / 'manifest.json').write_text(json.dumps({'snapshot_id': 'test'}))

    class Selected:
        def __init__(self, root):
            pass

        def selected_tables(self, *, dataset_key):
            yield SimpleNamespace(snapshot_id='test', source_path='test', event_date_column='week',
                                  geographic_resolutions=('state',), iter_rows=lambda: iter(rows[dataset_key]))

    monkeypatch.setattr(module, 'SelectedData', Selected)
    ds = build_dataset(tmp_path)
    al = ds.locations.index('AL')
    assert ds.panel[0, 0, :, al].tolist() == [0, 1]
    assert not ds.panel[0, 1:3, :, al].any()
    assert ds.panel[0, 3, 0, al] == pytest.approx(.02)
    assert not ds.panel[0, 4, :, al].any()
