import numpy as np
import pandas as pd
import pytest

from tapestry.evaluation.hubs import KEY, QCOLS, wide_quantiles
from tapestry.evaluation.compare import score_with_r
from tapestry.models.season_cv import LEVELS


def test_invalid_whole_quantile_tasks_are_excluded():
    rows = []
    for loc in ['01', '02', '04']:
        for q in LEVELS:
            rows.append(dict(reference_date='2025-01-04', target_end_date='2025-01-04',
                             location=loc, horizon=0, target='wk inc flu hosp', output_type_id=str(q), value=q * 100))
    df = pd.DataFrame(rows)
    df.loc[(df.location == '02') & (df.output_type_id == '0.5'), 'value'] = 999
    df = df[~((df.location == '04') & (df.output_type_id == '0.5'))]
    result, audit = wide_quantiles(df, 'wk inc flu hosp')
    assert result.location.tolist() == ['01']
    assert audit['invalid_or_incomplete_tasks'] == 2
    changed = df[df.location == '01'].iloc[[0]].copy()
    changed['value'] = 123
    result, audit = wide_quantiles(pd.concat([df, changed]), 'wk inc flu hosp')
    assert result.empty
    assert audit['conflicting_tasks'] == 1


def test_actual_scoringutils_matches_full_grid_pinball(tmp_path):
    import shutil
    if not shutil.which('Rscript'):
        pytest.skip('Rscript unavailable')
    values = np.linspace(0, 100, 23)
    rows = []
    for model in ['ours', 'ensemble']:
        row = dict(model=model, reference_date='2025-01-04', target_end_date='2025-01-04', location='01', horizon=0, observed=70)
        row.update(zip(QCOLS, values))
        rows.append(row)
    scores = score_with_r(pd.DataFrame(rows), tmp_path)
    error = 70 - values
    expected = 2 * np.maximum(LEVELS * error, (LEVELS - 1) * error).mean()
    np.testing.assert_allclose(scores.wis, expected)
    assert scores.location.tolist() == ['01', '01']
    assert set(scores.model) == {'ours', 'ensemble'}


def test_ensemble_support_and_complete_best_rule(tmp_path):
    import shutil
    if not shutil.which('Rscript'):
        pytest.skip('Rscript unavailable')
    from tapestry.evaluation.compare import compare_case
    target = 'wk inc flu hosp'
    cache = tmp_path / 'cache'
    folder = cache / 'flusight' / 'flu_hosp'
    folder.mkdir(parents=True)
    keys = pd.DataFrame({'reference_date': ['2025-01-04'] * 3,
                         'target_end_date': ['2025-01-04'] * 3,
                         'location': ['01', '02', '04'], 'horizon': [0] * 3})
    def predictions(locations, value):
        frame = keys[keys.location.isin(locations)].copy()
        frame[QCOLS] = value
        return frame
    predictions(['01', '02'], 10).to_parquet(folder / 'FluSight-ensemble.parquet', index=False)
    predictions(['01', '02'], 9).to_parquet(folder / 'full.parquet', index=False)
    predictions(['01'], 11).to_parquet(folder / 'partial.parquet', index=False)
    truth = keys[['target_end_date', 'location']].assign(target=target, observed=11)
    truth.to_parquet(cache / 'flusight' / 'truth.parquet', index=False)
    ours = predictions(['01', '02', '04'], 10).assign(b0_original_truth=11, b0_original_mask=True)
    result = compare_case(ours, 'flusight', '2024-2025', target, cache, tmp_path / 'output', 'ours', 'Rscript')
    assert result['n_units'] == 2  # Location 04 has no ensemble forecast.
    assert result['best'] == 'full'  # Partial's lower WIS cannot win through missingness.
    leaderboard = pd.read_csv(tmp_path / 'output' / result['directory'] / 'leaderboard.csv').set_index('model')
    assert leaderboard.loc['partial', 'coverage'] == .5
    assert not leaderboard.loc['partial', 'eligible_best']


def test_export_maps_leads_and_channel_order(tmp_path):
    from tapestry.evaluation.hubs import export_b0, SEASONS
    from datetime import date, timedelta
    for i, label in enumerate(SEASONS):
        folder = tmp_path / f'eval_{label}'
        folder.mkdir()
        context = [date(2023, 10, 7), date(2024, 10, 5), date(2025, 10, 4)][i]
        targets = [(context + timedelta(weeks=h)).isoformat() for h in range(1, 5)]
        q = np.broadcast_to(np.arange(6)[None, None, None, :, None], (23, 1, 4, 6, 1)).copy()
        np.savez(folder / 'forecasts.npz', quantile_levels=LEVELS, quantiles=q,
                 context_end=[context.isoformat()], target_dates=[targets], locations=['NC'],
                 truth=np.zeros((1, 4, 6, 1)), mask=np.ones((1, 4, 6, 1), dtype=bool))
    frame = export_b0(tmp_path)[('2023-2024', 'wk inc flu prop ed visits')]
    assert frame.reference_date.unique().tolist() == ['2023-10-14']
    assert frame.horizon.tolist() == [0, 1, 2, 3]
    assert frame.location.tolist() == ['37'] * 4
    assert (frame['q0.5'] == 3).all()
    assert frame.target_end_date.tolist() == ['2023-10-14', '2023-10-21', '2023-10-28', '2023-11-04']


def test_fallback_ranks_on_identical_tasks():
    from tapestry.evaluation.summary import select_best, METRICS
    rows = []
    for model, omitted in [('ours', None), ('ensemble', None), ('a', 0), ('b', 9)]:
        for i in range(10):
            if i == omitted:
                continue
            row = dict(model=model, reference_date=f'2025-01-{i+1:02}', target_end_date=f'2025-01-{i+1:02}', location='01', horizon=0)
            row.update({metric: 1. for metric in METRICS})
            row['wis'] = 2. if model == 'a' else 3.
            # A single non-common easy task must not change the ranking.
            if i == 0 and model == 'b':
                row['wis'] = 0
            rows.append(row)
    board, winner, units, summary, fallback = select_best(pd.DataFrame(rows), 'ours', 'ensemble')
    assert fallback and winner == 'a' and len(units) == 8
    assert all(row['n'] == 8 for row in summary)
