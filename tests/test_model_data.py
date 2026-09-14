from datetime import date

import numpy as np
import pytest

from influpaintx.model_data import FinalizedDataset, build_dataset
from influpaintx.model_data.finalized import season


def test_windows_masks_alignment_and_roundtrip(tmp_path):
    panel = np.zeros((3, 6, 2, 2), dtype=np.float32)
    panel[0, 0, :, 0] = (0, 1)  # A genuine observed zero.
    panel[1, 0, :, 1] = (20, 1)
    panel[2, 0, :, 1] = (30, 1)
    ds = FinalizedDataset(panel, ('2023-09-02', '2023-09-09', '2023-09-16'), ('AL', 'US'), {})
    ds.save(tmp_path / 'panel.npz')
    ds = FinalizedDataset.load(tmp_path / 'panel.npz')
    result = ds.query('2023-09-02', locations=['US', 'AL'], target_end='2023-09-09')
    assert result['X'].shape == (8, 6, 2, 2)
    assert result['X'][-1, 0, :, 1].tolist() == [0, 1]
    assert not result['X'][:-1].any()
    assert result['Y'][0, 0, :, 0].tolist() == [20, 1]
    assert not result['Y'][1:].any()  # Held-out labels excluded.
    assert result['target_dates'][0] == '2023-09-09'
    assert ds.query('2023-09-09', lookback=12)['X'].shape[0] == 12
    with pytest.raises(ValueError):
        ds.query('2023-09-03')
    with pytest.raises(ValueError):
        ds.query('2023-09-02', horizons=[0])


def test_season_boundary_and_53_week_year():
    assert season(date(2023, 8, 5)) == '2023-2024'
    assert season(date(2023, 7, 29)) == '2022-2023'
    assert season(date(2021, 1, 2)) == '2020-2021'


def test_builder_units_missing_and_conflicts(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from influpaintx.model_data import finalized as module

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
    # The single supported build CLI must save the same panel and provenance.
    from influpaintx.model_data.cli import main
    output = tmp_path / 'processed' / 'canonical.npz'
    main(['build', '--data-root', str(tmp_path), '--output', str(output)])
    saved = FinalizedDataset.load(output)
    np.testing.assert_array_equal(saved.panel, ds.panel)
    assert json.loads(output.with_suffix('.json').read_text()) == saved.metadata
    rows['cdc_nhsn_final'].append(dict(rows['cdc_nhsn_final'][0], totalconfflunewadm='8'))
    with pytest.raises(ValueError, match='Conflicting'):
        build_dataset(tmp_path)


def test_cli_exports_query_and_season_batch(tmp_path, capsys):
    from influpaintx.model_data.cli import main

    panel = np.ones((3, 6, 2, 2), dtype=np.float32)
    dataset = tmp_path / 'dataset.npz'
    output = tmp_path / 'query.npz'
    FinalizedDataset(panel, ('2023-09-02', '2023-09-09', '2023-09-16'), ('AL', 'US'), {}).save(dataset)
    main(['query', '--dataset', str(dataset), '--context-end', '2023-09-02',
          '--locations', 'US', '--lookback', '12', '--output', str(output)])
    with np.load(output, allow_pickle=False) as result:
        assert result['X'].shape == (12, 6, 2, 1)
        assert result['Y'].shape == (4, 6, 2, 1)
        assert result['locations'].tolist() == ['US']
        assert result['target_dates'][0] == '2023-09-09'
    main(['windows', '--dataset', str(dataset), '--season', '2023-2024',
          '--target-end', '2023-09-09', '--output', str(output)])
    with np.load(output, allow_pickle=False) as result:
        assert result['X'].shape == (1, 8, 6, 2, 2)
        assert not result['Y'][0, 1:].any()
    with pytest.raises(SystemExit):
        main(['query', '--dataset', str(dataset), '--context-end', '2023-09-02', '--output', str(dataset)])
    assert FinalizedDataset.load(dataset).panel.shape == panel.shape
